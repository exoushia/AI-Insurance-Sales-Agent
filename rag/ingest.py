"""
rag.ingest — RUN ONCE to build the vector store.

Chunks all three corpora into contextual, metadata-rich nodes and embeds them
into a persisted ChromaDB collection (text-embedding-3-small). Re-running is
idempotent: the collection is dropped and rebuilt from scratch.

Usage:
    python -m rag.ingest

Requires OPENAI_API_KEY (loaded via settings) for the embedding calls.
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from settings import AppConfig
from policy_feature_registry import REGISTRY, all_product_ids

from rag import config as C


# ---------------------------------------------------------------------------
# 1) POLICY WORDING  (data/policies/swasthya_SP0XX.txt)
# ---------------------------------------------------------------------------

# Split on clause / section markers and on ==== separator rules.
_CLAUSE_SPLIT_RE = re.compile(r"(?=Clause \d+\.\d+|\bSECTION \d+:)|\n={2,}\n")
# Pull the current SECTION header (e.g. "SECTION 3: ELIGIBILITY...") for context.
_SECTION_RE = re.compile(r"SECTION \d+:[^\n]*", re.IGNORECASE)
_CLAUSE_ID_RE = re.compile(r"Clause \d+\.\d+")


def _product_name(pid: str) -> str:
    entry = REGISTRY.get(pid, {})
    return entry.get("name", pid)


def build_policy_nodes(splitter: SentenceSplitter) -> list[TextNode]:
    """One+ node per clause/section of every product's wording document.

    Long clauses are token-split by `splitter` (numbers preserved). Each node's
    text is prefixed with a context header naming the product + section so the
    embedding (and the LLM later) always knows what it is reading.
    """
    nodes: list[TextNode] = []
    for pid in all_product_ids():
        filepath = os.path.join(C.POLICIES_DIR, f"swasthya_{pid}.txt")
        if not os.path.exists(filepath):
            continue
        name = _product_name(pid)
        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read()

        current_section = ""
        for block in _CLAUSE_SPLIT_RE.split(raw):
            if not block:
                continue
            block = block.strip()
            if len(block) < 50:
                continue

            sec_match = _SECTION_RE.search(block)
            if sec_match:
                current_section = sec_match.group(0).strip()
            clause_match = _CLAUSE_ID_RE.search(block)
            clause_id = clause_match.group(0) if clause_match else ""

            section_label = current_section or "General"
            header = f"Product {pid} — {name} — {section_label}"
            if clause_id:
                header += f" — {clause_id}"

            for piece in splitter.split_text(block):
                text = f"{header}:\n{piece.strip()}"
                nodes.append(TextNode(text=text, metadata={
                    C.META_SOURCE_TYPE: C.SOURCE_POLICY,
                    C.META_PRODUCT_ID: pid,
                    C.META_PRODUCT_NAME: name,
                    C.META_SECTION: section_label,
                    "clause": clause_id,
                }))
    return nodes


# ---------------------------------------------------------------------------
# 2) REGULATIONS  (data/policy_regulations_rag_ready.json)
# ---------------------------------------------------------------------------

def build_regulation_nodes(splitter: SentenceSplitter) -> list[TextNode]:
    """One+ node per IRDAI section (content >= 80 chars), with a citation header."""
    if not os.path.exists(C.REGULATIONS_PATH):
        return []
    with open(C.REGULATIONS_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    nodes: list[TextNode] = []
    for doc in corpus.get("documents", []):
        doc_name = (doc.get("document_name", "") or "")[:80]
        doc_type = doc.get("document_type", "")
        issuer = doc.get("issuer", "")
        for chapter in doc.get("chapters", []):
            chapter_title = chapter.get("chapter_title", "")
            for section in chapter.get("sections", []):
                content = (section.get("content", "") or "").strip()
                if len(content) < 80:
                    continue
                sec_num = section.get("section_number", "")
                sec_title = section.get("section_title", "")
                category = section.get("retrieval_category", "")
                header = (f"{doc_name} ({doc_type}, {issuer}) — {chapter_title} — "
                          f"§{sec_num} {sec_title}")
                for piece in splitter.split_text(content):
                    text = f"{header}:\n{piece.strip()}"
                    nodes.append(TextNode(text=text, metadata={
                        C.META_SOURCE_TYPE: C.SOURCE_REGULATION,
                        C.META_DOCUMENT: doc_name,
                        "document_type": doc_type,
                        "issuer": issuer,
                        "chapter": chapter_title[:80],
                        C.META_SECTION_NUMBER: sec_num,
                        C.META_SECTION_TITLE: sec_title,
                        C.META_CATEGORY: category,
                    }))
    return nodes


# ---------------------------------------------------------------------------
# 3) TREATMENT COSTS  (data/treatment_costs.json — Python literal)
# ---------------------------------------------------------------------------

def _load_treatment_costs() -> dict:
    """The file is `TREATMENT_COSTS = {...}` — a Python literal, not JSON."""
    try:
        with open(C.TREATMENT_COSTS_PATH, "r", encoding="utf-8") as f:
            text = f.read()
        brace = text.index("{")
        return ast.literal_eval(text[brace:])
    except (OSError, ValueError, SyntaxError):
        return {}


def _humanise(label: str) -> str:
    return label.replace("_", " ").strip()


def build_cost_nodes() -> list[TextNode]:
    """Render each cost leaf into a natural-language sentence keeping ₹ + commas
    (so the numeric guardrail can verify any figure the LLM repeats)."""
    costs = _load_treatment_costs()
    nodes: list[TextNode] = []

    for category, group in costs.items():
        cat_h = _humanise(category)
        if not isinstance(group, dict):
            continue
        for item, value in group.items():
            item_h = _humanise(item)
            if isinstance(value, dict):
                # Either {min,max} range or {setting: amount, ...}.
                if set(value.keys()) <= {"min", "max"} and value:
                    lo, hi = value.get("min"), value.get("max")
                    sentence = (f"{item_h} ({cat_h}) typically ranges from "
                                f"₹{lo:,} to ₹{hi:,}.")
                    nodes.append(_cost_node(sentence, cat_h, item_h, ""))
                else:
                    for setting, amount in value.items():
                        if not isinstance(amount, (int, float)):
                            continue
                        setting_h = _humanise(setting)
                        sentence = (f"{item_h} in a {setting_h} setting "
                                    f"({cat_h}) typically costs ₹{amount:,}.")
                        nodes.append(_cost_node(sentence, cat_h, item_h, setting_h))
            elif isinstance(value, (int, float)):
                sentence = f"{item_h} ({cat_h}) typically costs ₹{value:,}."
                nodes.append(_cost_node(sentence, cat_h, item_h, ""))
    return nodes


def _cost_node(text: str, category: str, procedure: str, setting: str) -> TextNode:
    return TextNode(text=text, metadata={
        C.META_SOURCE_TYPE: C.SOURCE_COST,
        C.META_CATEGORY: category,
        "procedure": procedure,
        "setting": setting,
    })


# ---------------------------------------------------------------------------
# BUILD + PERSIST
# ---------------------------------------------------------------------------

def ingest() -> dict[str, int]:
    """Build all nodes and (re)build the persisted Chroma collection."""
    config = AppConfig()
    if not config.openai_api_key:
        raise SystemExit(
            "OPENAI_API_KEY is required to embed the corpora. Set it in .env."
        )

    embed_model = OpenAIEmbedding(
        model=C.EMBED_MODEL, api_key=config.openai_api_key,
    )
    splitter = SentenceSplitter(chunk_size=C.CHUNK_SIZE, chunk_overlap=C.CHUNK_OVERLAP)

    print("Building nodes…")
    policy_nodes = build_policy_nodes(splitter)
    regulation_nodes = build_regulation_nodes(splitter)
    cost_nodes = build_cost_nodes()
    all_nodes = policy_nodes + regulation_nodes + cost_nodes
    counts = {
        "policy_wording": len(policy_nodes),
        "regulation": len(regulation_nodes),
        "treatment_cost": len(cost_nodes),
        "total": len(all_nodes),
    }
    for k, v in counts.items():
        print(f"  {k:16}: {v}")
    if not all_nodes:
        raise SystemExit("No nodes produced — check the data/ corpora paths.")

    os.makedirs(C.PERSIST_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=C.PERSIST_DIR)
    # Idempotent rebuild: drop any existing collection first.
    try:
        client.delete_collection(C.COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(C.COLLECTION_NAME)

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print(f"Embedding {len(all_nodes)} nodes with {C.EMBED_MODEL} → {C.PERSIST_DIR}…")
    VectorStoreIndex(
        all_nodes, storage_context=storage_context, embed_model=embed_model,
    )
    print(f"Done. Collection '{C.COLLECTION_NAME}' persisted "
          f"({collection.count()} vectors).")
    return counts


if __name__ == "__main__":
    ingest()
