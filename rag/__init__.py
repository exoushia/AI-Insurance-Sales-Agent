"""
rag — vector retrieval layer for the AI Insurance Sales Agent.

Houses the LlamaIndex + ChromaDB retrieval stack that powers M_09 (AgenticRAG):
  - config.py      : paths, collection name, embed model, metadata keys.
  - ingest.py      : RUN ONCE — chunk + embed all corpora into the Chroma store.
  - index_store.py : lazy singleton retriever queried at conversation time.
"""
