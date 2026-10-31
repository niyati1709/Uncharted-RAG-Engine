"""
backend_adapter.py - Thin Compatibility & Integration Layer for DevDocs AI
==========================================================================
Provides standard interface bridges for Member 1's Streamlit frontend (app.py)
and integration tests.

Exposes:
1. ingest_pdf(file_bytes, document_name, owner, document_id=None)
2. list_documents(owner=None, limit=100)
3. answer_question(query, filters=None, top_k=5)
"""

from typing import List, Dict, Any, Optional

from backend.rag_pipeline import (
    ingest_pdf as _backend_ingest_pdf,
    answer_question as _backend_answer_question,
)
from backend.database import (
    list_documents as _backend_list_documents,
    ping_connection as _backend_ping_connection,
)


def ingest_pdf(
    file_bytes: Any,
    document_name: str,
    owner: str = "default_owner",
    document_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Ingest uploaded PDF document bytes into MongoDB Atlas with 384-dim vector embeddings.
    
    Returns:
        {
            "document_id": str,
            "document_name": str,
            "owner": str,
            "total_chunks": int,
            "status": "success" | "error",
            "message": str
        }
    """
    return _backend_ingest_pdf(
        file_bytes=file_bytes,
        document_name=document_name,
        owner=owner,
        document_id=document_id
    )


def list_documents(
    owner: Optional[str] = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    List uploaded documents with metadata and chunk counts.
    
    Returns:
        List of dicts: [
            {
                "document_id": str,
                "document_name": str,
                "upload_date": str,
                "owner": str,
                "content_type": str,
                "chunk_count": int
            }
        ]
    """
    return _backend_list_documents(owner=owner, limit=limit)


def answer_question(
    query: str,
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Answer user query grounded strictly in retrieved documentation chunks.
    
    Returns:
        {
            "answer": str,
            "citations": [
                {
                    "source_id": str,
                    "document_id": str,
                    "document_name": str,
                    "page_number": int,
                    "text": str
                }
            ],
            "status": "answered" | "not_found" | "conflict" | "error"
        }
    """
    return _backend_answer_question(query=query, filters=filters, top_k=top_k)


# Convenience Aliases for compatibility
ask_question = answer_question
query_documents = answer_question
upload_pdf = ingest_pdf
get_documents = list_documents
check_health = _backend_ping_connection

__all__ = [
    "ingest_pdf",
    "list_documents",
    "answer_question",
    "ask_question",
    "query_documents",
    "upload_pdf",
    "get_documents",
    "check_health",
]

