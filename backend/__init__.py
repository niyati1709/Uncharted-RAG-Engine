"""
DevDocs AI Backend Package
==========================
Public API for Streamlit Frontend and Integration Adapters:
- ingest_pdf(file_bytes, document_name, owner, document_id=None)
- list_documents(owner=None, limit=100)
- answer_question(query, filters=None, top_k=5)
"""

from backend.database import (
    get_database,
    close_database,
    ping_connection,
    save_document_metadata,
    save_chunks,
    retrieve_chunks,
    retrieve_chunks_by_keyword,
    list_documents,
    get_document,
    delete_document,
)

from backend.rag_pipeline import (
    ingest_pdf,
    answer_question,
    retrieve_hybrid_chunks,
    generate_answer,
    generate_query_embedding,
    reciprocal_rank_fusion,
)

__all__ = [
    "ingest_pdf",
    "list_documents",
    "answer_question",
    "get_database",
    "close_database",
    "ping_connection",
    "save_document_metadata",
    "save_chunks",
    "retrieve_chunks",
    "retrieve_chunks_by_keyword",
    "get_document",
    "delete_document",
    "retrieve_hybrid_chunks",
    "generate_answer",
    "generate_query_embedding",
    "reciprocal_rank_fusion",
]
