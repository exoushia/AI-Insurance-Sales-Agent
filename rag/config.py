"""
rag.config — shared constants for the vector retrieval layer.

Single source of truth for: where data lives, where the Chroma store persists,
which embedding model produces the vectors, and the metadata keys attached to
every node. Imported by both the run-once ingestion (ingest.py) and the
conversation-time retriever (index_store.py) so they can never drift apart.
"""

from __future__ import annotations

import os

# Package directory (AI-Insurance-Sales-Agent/) — parent of rag/.
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Corpora locations (read-only inputs to ingestion).
DATA_DIR = os.path.join(_PKG_DIR, "data")
POLICIES_DIR = os.path.join(DATA_DIR, "policies")
REGULATIONS_PATH = os.path.join(DATA_DIR, "policy_regulations_rag_ready.json")
TREATMENT_COSTS_PATH = os.path.join(DATA_DIR, "treatment_costs.json")

# Persisted Chroma vector store (created by ingestion, read by the retriever).
PERSIST_DIR = os.path.join(DATA_DIR, "vector_store")
COLLECTION_NAME = "swasthya_rag"

# Embedding model — explicit OpenAI text-embedding-3-small (1536-dim).
# Uses the same OPENAI_API_KEY as the rest of the app (loaded via settings).
EMBED_MODEL = "text-embedding-3-small"

# Chunking — token-aware sentence splitting for long clauses/sections.
# Generous size so monetary figures (₹ + commas) are never split mid-number.
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# Retrieval defaults.
DEFAULT_TOP_K = 6

# Node metadata keys (stable strings shared by ingestion + retrieval).
META_SOURCE_TYPE = "source_type"      # "policy_wording" | "regulation" | "treatment_cost"
META_PRODUCT_ID = "product_id"        # policy_wording only (e.g. "SP007")
META_PRODUCT_NAME = "product_name"
META_SECTION = "section"
META_DOCUMENT = "document"
META_SECTION_NUMBER = "section_number"
META_SECTION_TITLE = "section_title"
META_CATEGORY = "category"            # treatment_cost path

# source_type values.
SOURCE_POLICY = "policy_wording"
SOURCE_REGULATION = "regulation"
SOURCE_COST = "treatment_cost"
