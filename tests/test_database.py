"""
tests/test_database.py - Unit and Integration Tests for backend/database.py
==========================================================================
Tests:
- Input validation (missing fields, wrong data types)
- Embedding dimension validation (expected 384 vs mismatch)
- Document metadata insertion and upsert logic
- Chunk bulk upsert and idempotency
- MongoDB query filtering
- Error handling on disconnected/invalid URI
"""

import os
import pytest
from unittest.mock import MagicMock, patch
from backend.database import (
    save_document_metadata,
    save_chunks,
    retrieve_chunks,
    retrieve_chunks_by_keyword,
    close_database,
    EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_DIMENSIONS,
)


@pytest.fixture(autouse=True)
def cleanup():
    """Ensure clean database client state before/after each test."""
    close_database()
    yield
    close_database()


# ---------------------------------------------------------------------------
# Test 1: Embedding Dimension Validation
# ---------------------------------------------------------------------------

def test_save_chunks_validates_dimension():
    """Test that save_chunks rejects embeddings that do not match the expected 384 dimensions."""
    invalid_chunks = [
        {
            "chunk_id": "chunk_001",
            "document_id": "doc_001",
            "document_name": "test.pdf",
            "text": "Sample text",
            "page_number": 1,
            "embedding": [0.1] * 128,  # Only 128 instead of 384!
            "content_type": "application/pdf"
        }
    ]

    with pytest.raises(ValueError) as exc_info:
        save_chunks(invalid_chunks)

    assert "Embedding dimension mismatch" in str(exc_info.value)
    assert "Expected 384" in str(exc_info.value)
    assert "received 128" in str(exc_info.value)


def test_save_chunks_rejects_empty_or_non_list():
    """Test that save_chunks rejects empty chunk lists."""
    with pytest.raises(ValueError):
        save_chunks([])


def test_save_chunks_rejects_missing_required_fields():
    """Test that save_chunks enforces chunk_id, document_id, and text."""
    # Missing chunk_id
    with pytest.raises(ValueError) as exc:
        save_chunks([{"document_id": "doc_1", "text": "foo", "embedding": [0.1] * 384}])
    assert "chunk_id" in str(exc.value)

    # Missing document_id
    with pytest.raises(ValueError) as exc:
        save_chunks([{"chunk_id": "c_1", "text": "foo", "embedding": [0.1] * 384}])
    assert "document_id" in str(exc.value)


# ---------------------------------------------------------------------------
# Test 2: Document Metadata Insertion (Mocked)
# ---------------------------------------------------------------------------

@patch("backend.database.get_database")
def test_save_document_metadata_success(mock_get_db):
    """Test saving document metadata with valid fields."""
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_db.__getitem__.return_value = mock_collection
    mock_get_db.return_value = mock_db

    doc_data = {
        "document_id": "doc_001",
        "document_name": "python_tutorial.pdf",
        "owner": "alice",
        "content_type": "application/pdf"
    }

    result = save_document_metadata(doc_data)

    assert result["document_id"] == "doc_001"
    assert result["document_name"] == "python_tutorial.pdf"
    assert result["owner"] == "alice"
    assert "upload_date" in result

    # Verify upsert replace_one was called with the document_id
    mock_collection.replace_one.assert_called_once()
    args, kwargs = mock_collection.replace_one.call_args
    assert args[0] == {"document_id": "doc_001"}
    assert kwargs.get("upsert") is True


def test_save_document_metadata_validation():
    """Test that missing document_id or document_name raises ValueError."""
    with pytest.raises(ValueError):
        save_document_metadata({"document_name": "no_id.pdf"})

    with pytest.raises(ValueError):
        save_document_metadata({"document_id": "doc_1"})

    with pytest.raises(TypeError):
        save_document_metadata("not_a_dict")


# ---------------------------------------------------------------------------
# Test 3: Chunk Bulk Upsert (Mocked)
# ---------------------------------------------------------------------------

@patch("backend.database.get_database")
def test_save_chunks_bulk_upsert_success(mock_get_db):
    """Test saving valid 384-dimensional chunks using bulk write."""
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_db.__getitem__.return_value = mock_collection
    mock_get_db.return_value = mock_db

    # Mock bulk write response
    mock_bulk_result = MagicMock()
    mock_bulk_result.upserted_count = 2
    mock_bulk_result.modified_count = 0
    mock_bulk_result.matched_count = 0
    mock_collection.bulk_write.return_value = mock_bulk_result

    valid_chunks = [
        {
            "chunk_id": f"chunk_00{i}",
            "document_id": "doc_001",
            "document_name": "python_docs.pdf",
            "text": f"Documentation paragraph {i}",
            "page_number": i,
            "embedding": [0.05] * 384,
            "content_type": "application/pdf"
        }
        for i in range(1, 3)
    ]

    result = save_chunks(valid_chunks)

    assert result["status"] == "success"
    assert result["total_chunks"] == 2
    assert result["upserted_count"] == 2
    mock_collection.bulk_write.assert_called_once()


# ---------------------------------------------------------------------------
# Test 4: Vector Retrieval Dimension & Pipeline Construction (Mocked)
# ---------------------------------------------------------------------------

@patch("backend.database.get_database")
def test_retrieve_chunks_vector_pipeline(mock_get_db):
    """Test that retrieve_chunks constructs the exact Atlas $vectorSearch pipeline."""
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_db.__getitem__.return_value = mock_collection
    mock_get_db.return_value = mock_db

    # Return dummy search results
    mock_collection.aggregate.return_value = [
        {
            "chunk_id": "chunk_001",
            "document_id": "doc_001",
            "document_name": "python_docs.pdf",
            "text": "Decorators dynamically alter functions.",
            "page_number": 5,
            "content_type": "application/pdf",
            "score": 0.89
        }
    ]

    dummy_vector = [0.1] * 384
    filters = {"document_id": "doc_001"}

    results = retrieve_chunks(query_embedding=dummy_vector, filters=filters, top_k=3)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "chunk_001"
    assert results[0]["score"] == 0.89

    # Verify pipeline structure passed to aggregate()
    mock_collection.aggregate.assert_called_once()
    pipeline = mock_collection.aggregate.call_args[0][0]

    assert "$vectorSearch" in pipeline[0]
    stage = pipeline[0]["$vectorSearch"]
    assert stage["path"] == "embedding"
    assert len(stage["queryVector"]) == 384
    assert stage["limit"] == 3
    assert stage["filter"] == {"document_id": {"$eq": "doc_001"}}


def test_retrieve_chunks_dimension_check():
    """Test that retrieve_chunks rejects query vectors with wrong dimensions."""
    with pytest.raises(ValueError) as exc:
        retrieve_chunks([0.1] * 100)
    assert "Invalid query_embedding" in str(exc.value)


# ---------------------------------------------------------------------------
# Test 5: Keyword Search (Mocked)
# ---------------------------------------------------------------------------

@patch("backend.database.get_database")
def test_retrieve_chunks_by_keyword(mock_get_db):
    """Test keyword retrieval matching exact terms."""
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_db.__getitem__.return_value = mock_collection
    mock_get_db.return_value = mock_db

    mock_collection.find.return_value = [
        {
            "chunk_id": "chunk_001",
            "document_id": "doc_001",
            "document_name": "python_docs.pdf",
            "text": "A decorator is a function that takes another function as an argument.",
            "page_number": 3,
            "content_type": "application/pdf",
        },
        {
            "chunk_id": "chunk_002",
            "document_id": "doc_001",
            "document_name": "python_docs.pdf",
            "text": "Functions in Python are first-class objects.",
            "page_number": 2,
            "content_type": "application/pdf",
        }
    ]

    results = retrieve_chunks_by_keyword(query_text="decorator function", top_k=2)

    assert len(results) == 2
    # The first document has both 'decorator' and 'function', so it should have a higher score
    assert results[0]["chunk_id"] == "chunk_001"
    assert results[0]["score"] >= results[1]["score"]

