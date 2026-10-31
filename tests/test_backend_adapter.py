"""
tests/test_backend_adapter.py - Integration & Adapter Tests for DevDocs AI
==========================================================================
Tests the three public functions:
1. ingest_pdf(file_bytes, document_name, owner, document_id=None)
2. list_documents(owner=None, limit=100)
3. answer_question(query, filters=None, top_k=5)
And verifies compatibility aliases.
"""

import io
import pytest
from unittest.mock import patch, MagicMock

import backend_adapter
from backend_adapter import ingest_pdf, list_documents, answer_question


# ---------------------------------------------------------------------------
# Test 1: list_documents
# ---------------------------------------------------------------------------

@patch("backend.database.get_database")
def test_list_documents_empty(mock_get_db):
    """Test list_documents returns empty list when collection is empty."""
    mock_db = MagicMock()
    mock_docs_col = MagicMock()
    mock_chunks_col = MagicMock()
    mock_db.__getitem__.side_effect = lambda name: mock_docs_col if name == "documents" else mock_chunks_col
    mock_get_db.return_value = mock_db

    mock_docs_col.find.return_value = []

    docs = list_documents()
    assert isinstance(docs, list)
    assert len(docs) == 0


@patch("backend.database.get_database")
def test_list_documents_filtering_and_counts(mock_get_db):
    """Test list_documents filters by owner and attaches chunk_count."""
    mock_db = MagicMock()
    mock_docs_col = MagicMock()
    mock_chunks_col = MagicMock()
    mock_db.__getitem__.side_effect = lambda name: mock_docs_col if name == "documents" else mock_chunks_col
    mock_get_db.return_value = mock_db

    mock_docs_col.find.return_value = [
        {
            "document_id": "doc_001",
            "document_name": "python_docs.pdf",
            "upload_date": "2026-10-31T10:00:00Z",
            "owner": "alice",
            "content_type": "application/pdf"
        }
    ]
    mock_chunks_col.count_documents.return_value = 14

    docs = list_documents(owner="alice", limit=10)

    assert len(docs) == 1
    d = docs[0]
    assert d["document_id"] == "doc_001"
    assert d["document_name"] == "python_docs.pdf"
    assert d["owner"] == "alice"
    assert d["chunk_count"] == 14

    # Verify find filter was passed
    mock_docs_col.find.assert_called_once()
    filter_arg = mock_docs_col.find.call_args[0][0]
    assert filter_arg == {"owner": "alice"}


# ---------------------------------------------------------------------------
# Test 2: ingest_pdf
# ---------------------------------------------------------------------------

def test_ingest_pdf_empty_bytes():
    """Test ingest_pdf handles empty or invalid bytes gracefully."""
    result = ingest_pdf(b"", "empty.pdf", owner="alice")
    assert result["status"] == "error"
    assert "Invalid or empty" in result["message"]
    assert result["total_chunks"] == 0


@patch("backend.rag_pipeline.save_document_metadata")
@patch("backend.rag_pipeline.save_chunks")
@patch("backend.rag_pipeline._extract_text_from_pdf_bytes")
def test_ingest_pdf_success(mock_extract, mock_save_chunks, mock_save_meta):
    """Test successful PDF ingestion flow."""
    mock_extract.return_value = [
        {"page_number": 1, "text": "Python is a dynamic programming language."},
        {"page_number": 2, "text": "Python supports multiple programming paradigms."}
    ]
    mock_save_meta.return_value = {}
    mock_save_chunks.return_value = {"status": "success"}

    dummy_pdf_bytes = b"%PDF-1.4 dummy binary content"

    result = ingest_pdf(
        file_bytes=dummy_pdf_bytes,
        document_name="python_intro.pdf",
        owner="developer_1",
        document_id="doc_custom_99"
    )

    assert result["status"] == "success"
    assert result["document_id"] == "doc_custom_99"
    assert result["document_name"] == "python_intro.pdf"
    assert result["owner"] == "developer_1"
    assert result["total_chunks"] >= 2
    assert "Successfully ingested" in result["message"]

    mock_save_meta.assert_called_once()
    mock_save_chunks.assert_called_once()


# ---------------------------------------------------------------------------
# Test 3: answer_question
# ---------------------------------------------------------------------------

@patch("backend.rag_pipeline.retrieve_hybrid_chunks")
@patch("backend.rag_pipeline.generate_answer")
def test_answer_question_passes_top_k(mock_generate, mock_retrieve):
    """Test answer_question forwards top_k and filters correctly."""
    mock_retrieve.return_value = [
        {
            "chunk_id": "c1",
            "document_id": "doc_1",
            "document_name": "guide.pdf",
            "page_number": 1,
            "text": "Decorators modify behavior."
        }
    ]
    mock_generate.return_value = {
        "answer": "Decorators modify behavior in Python.",
        "citations": [
            {
                "source_id": "c1",
                "document_id": "doc_1",
                "document_name": "guide.pdf",
                "page_number": 1,
                "text": "Decorators modify behavior."
            }
        ],
        "status": "answered"
    }

    result = answer_question(
        query="What do decorators do?",
        filters={"document_id": "doc_1"},
        top_k=3
    )

    assert result["status"] == "answered"
    assert "Decorators modify behavior" in result["answer"]
    assert len(result["citations"]) == 1
    cit = result["citations"][0]
    assert cit["source_id"] == "c1"
    assert cit["document_id"] == "doc_1"
    assert cit["page_number"] == 1

    # Verify top_k was passed to retrieve_hybrid_chunks
    mock_retrieve.assert_called_once_with(
        "What do decorators do?",
        filters={"document_id": "doc_1"},
        top_k=3
    )


def test_answer_question_empty_input():
    """Test answer_question handles empty query gracefully."""
    result = answer_question("   ")
    assert result["status"] == "error"
    assert result["citations"] == []


# ---------------------------------------------------------------------------
# Test 4: Compatibility Layer Aliases
# ---------------------------------------------------------------------------

def test_compatibility_aliases():
    """Verify that all convenience aliases point to the correct implementation."""
    assert backend_adapter.ask_question == backend_adapter.answer_question
    assert backend_adapter.query_documents == backend_adapter.answer_question
    assert backend_adapter.upload_pdf == backend_adapter.ingest_pdf
    assert backend_adapter.get_documents == backend_adapter.list_documents

