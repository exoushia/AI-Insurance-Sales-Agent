"""
rag.index_store — conversation-time retriever (lazy singleton).

Loads the persisted Chroma collection once and exposes `query(...)` for M_09.
If the store has not been built (run `python -m rag.ingest`) or OpenAI is not
configured, `get_rag_index()` returns None and callers escalate.
"""

from __future__ import annotations

import os
import sys

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from settings import AppConfig
from rag import config as C

# Cached singletons (module-global) — built on first successful load.
_INDEX = None          # RagIndex instance
_LOAD_ATTEMPTED = False


class RagIndex:
    """Thin wrapper over a LlamaIndex VectorStoreIndex backed by Chroma."""

    def __init__(self, index) -> None:
        self._index = index

    def query(self, text: str, top_k: int = C.DEFAULT_TOP_K,
              product_id: str | None = None) -> list[dict]:
        """Return top-k hits as [{text, score, metadata}].

        When `product_id` is given, results are restricted to that product's
        wording (metadata filter); otherwise the whole corpus is searched.
        """
        if not text or not text.strip():
            return []

        filters = None
        if product_id:
            from llama_index.core.vector_stores import (
                MetadataFilter, MetadataFilters, FilterOperator,
            )
            filters = MetadataFilters(filters=[
                MetadataFilter(key=C.META_PRODUCT_ID, value=product_id,
                               operator=FilterOperator.EQ),
            ])

        retriever = self._index.as_retriever(similarity_top_k=top_k, filters=filters)
        results = retriever.retrieve(text)
        hits: list[dict] = []
        for node_with_score in results:
            node = node_with_score.node
            hits.append({
                "text": node.get_content(),
                "score": float(node_with_score.score or 0.0),
                "metadata": dict(node.metadata or {}),
            })
        return hits


def get_rag_index() -> RagIndex | None:
    """Lazily load and cache the persisted vector index. Returns None if the
    store is missing or OpenAI embeddings cannot be configured."""
    global _INDEX, _LOAD_ATTEMPTED
    if _INDEX is not None:
        return _INDEX
    if _LOAD_ATTEMPTED:
        return _INDEX   # previous attempt failed; don't retry every turn
    _LOAD_ATTEMPTED = True

    if not os.path.isdir(C.PERSIST_DIR):
        return None
    config = AppConfig()
    if not config.openai_api_key:
        return None

    try:
        import chromadb
        from llama_index.core import VectorStoreIndex
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.vector_stores.chroma import ChromaVectorStore

        client = chromadb.PersistentClient(path=C.PERSIST_DIR)
        collection = client.get_collection(C.COLLECTION_NAME)
        if collection.count() == 0:
            return None
        vector_store = ChromaVectorStore(chroma_collection=collection)
        embed_model = OpenAIEmbedding(
            model=C.EMBED_MODEL, api_key=config.openai_api_key,
        )
        index = VectorStoreIndex.from_vector_store(
            vector_store, embed_model=embed_model,
        )
        _INDEX = RagIndex(index)
    except Exception:
        _INDEX = None
    return _INDEX


# ---------------------------------------------------------------------------
# SMOKE TEST  (requires a built store + OPENAI_API_KEY)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    idx = get_rag_index()
    if idx is None:
        raise SystemExit(
            "No index loaded. Build it first: python -m rag.ingest "
            "(and ensure OPENAI_API_KEY is set)."
        )
    for q in ["free look period", "normal delivery cost in a private hospital",
              "maternity waiting period"]:
        print(f"\nQ: {q}")
        for h in idx.query(q, top_k=3):
            meta = h["metadata"]
            tag = meta.get(C.META_SOURCE_TYPE, "?")
            print(f"  [{tag} {h['score']:.3f}] {h['text'][:110].replace(chr(10), ' ')}")
    print("\nindex_store.py smoke test complete.")
