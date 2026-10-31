"""
tests/test_rag_pipeline.py - Unit and Integration Tests for backend/rag_pipeline.py
===================================================================================
Tests:
- Reciprocal Rank Fusion (RRF) ranking algorithm
- Citation formatting and preservation
- Grounded answer generation
- Handling unanswerable questions (Case 1 & Case 2)
- Handling conflicting documents (Case 3)
- Frontend integration contract compliance (status, answer, citations format)
"""

import pytest
from unittest.mock import patch, MagicMock
from backend.rag_pipeline import (
    reciprocal_rank_fusion,
    format_context_for_prompt,
    generate_answer,
    answer_question,
)


# ---------------------------------------------------------------------------
# Test 1: Reciprocal Rank Fusion (RRF) Logic
# ---------------------------------------------------------------------------

def test_reciprocal_rank_fusion_prefers_consensus():
    """
    Test that a chunk returned in both Vector Search and Keyword Search
    ranks higher than a chunk appearing only in one.
    """
    vector_results = [
        {"chunk_id": "chunk_A", "text": "Semantic match A", "score": 0.95},
        {"chunk_id": "chunk_B", "text": "Semantic match B", "score": 0.85},
    ]

    keyword_results = [
        {"chunk_id": "chunk_B", "text": "Exact keyword match B", "score": 3.0},
        {"chunk_id": "chunk_C", "text": "Exact keyword match C", "score": 1.0},
    ]

    # chunk_B appears in both!
    fused = reciprocal_rank_fusion(vector_results, keyword_results, k=60, top_k=3)

    assert len(fused) == 3
    # chunk_B should be #1 because it got rank points from BOTH retrievers
    assert fused[0]["chunk_id"] == "chunk_B"
    assert "rrf_score" in fused[0]
    assert fused[0]["rrf_score"] > fused[1]["rrf_score"]


# ---------------------------------------------------------------------------
# Test 2: Context Formatting
# ---------------------------------------------------------------------------

def test_format_context_for_prompt():
    """Verify that chunks are formatted with clean source identifiers."""
    chunks = [
        {
            "chunk_id": "c1",
            "document_id": "doc_101",
            "document_name": "python_tour.pdf",
            "page_number": 4,
            "text": "Decorators wrap functions."
        }
    ]

    formatted = format_context_for_prompt(chunks)

    assert "[Source 1 | ID: c1 | Doc: python_tour.pdf | Page: 4]" in formatted
    assert "Decorators wrap functions." in formatted


# ---------------------------------------------------------------------------
# Test 3: Grounded Answer & Citations
# ---------------------------------------------------------------------------

@patch("backend.rag_pipeline.call_llm")
def test_generate_answer_grounded_success(mock_call_llm):
    """Test generating a grounded answer with preserved citations."""
    mock_call_llm.return_value = "Python decorators wrap a function and alter its behavior [Source 1]."

    retrieved_chunks = [
        {
            "chunk_id": "chunk_001",
            "document_id": "doc_001",
            "document_name": "python_guide.pdf",
            "page_number": 12,
            "text": "A decorator takes a function and returns a new function.",
            "score": 0.92,
            "rrf_score": 0.03
        }
    ]

    result = generate_answer("What is a decorator?", retrieved_chunks)

    assert result["status"] == "answered"
    assert "decorators wrap a function" in result["answer"]
    assert len(result["citations"]) == 1
    citation = result["citations"][0]
    assert citation["source_id"] == "chunk_001"
    assert citation["document_id"] == "doc_001"
    assert citation["document_name"] == "python_guide.pdf"
    assert citation["page_number"] == 12
    assert citation["text"] == "A decorator takes a function and returns a new function."


# ---------------------------------------------------------------------------
# Test 4: Unanswerable Questions Handling
# ---------------------------------------------------------------------------

def test_generate_answer_unanswerable_no_chunks():
    """Case 1: No relevant chunks found."""
    result = generate_answer("What is quantum computing?", [])

    assert result["status"] == "not_found"
    assert result["answer"] == "The answer is not available in the uploaded documents."
    assert result["citations"] == []


@patch("backend.rag_pipeline.call_llm")
def test_generate_answer_unanswerable_chunks_insufficient(mock_call_llm):
    """Case 2: Chunks exist but LLM detects insufficient evidence."""
    mock_call_llm.return_value = "The answer is not available in the uploaded documents."

    chunks = [
        {
            "chunk_id": "chunk_002",
            "document_id": "doc_001",
            "document_name": "python_guide.pdf",
            "page_number": 3,
            "text": "Python was created by Guido van Rossum.",
            "score": 0.85
        }
    ]

    result = generate_answer("What is the rust borrow checker?", chunks)

    assert result["status"] == "not_found"
    assert result["answer"] == "The answer is not available in the uploaded documents."
    assert result["citations"] == []


# ---------------------------------------------------------------------------
# Test 5: Conflicting Documents Handling
# ---------------------------------------------------------------------------

@patch("backend.rag_pipeline.call_llm")
def test_generate_answer_conflicting_documents(mock_call_llm):
    """Case 3: Two documents contain contradictory information."""
    mock_call_llm.return_value = (
        "There is a conflict between the documents. In python_v2.pdf [Source 1], "
        "print is a statement, whereas in python_v3.pdf [Source 2], print is a function."
    )

    conflicting_chunks = [
        {
            "chunk_id": "chunk_v2",
            "document_id": "doc_py2",
            "document_name": "python_v2.pdf",
            "page_number": 1,
            "text": "print is a statement: print 'hello'",
            "score": 0.90
        },
        {
            "chunk_id": "chunk_v3",
            "document_id": "doc_py3",
            "document_name": "python_v3.pdf",
            "page_number": 1,
            "text": "print is a function: print('hello')",
            "score": 0.89
        }
    ]

    result = generate_answer("Is print a function or statement?", conflicting_chunks)

    assert result["status"] == "conflict"
    assert "conflict between the documents" in result["answer"]
    assert len(result["citations"]) == 2
    doc_names = {c["document_name"] for c in result["citations"]}
    assert "python_v2.pdf" in doc_names
    assert "python_v3.pdf" in doc_names


# ---------------------------------------------------------------------------
# Test 6: Member 1 Frontend Contract Verification
# ---------------------------------------------------------------------------

@patch("backend.rag_pipeline.retrieve_hybrid_chunks")
@patch("backend.rag_pipeline.generate_answer")
def test_answer_question_contract(mock_gen_answer, mock_retrieve):
    """
    Verify the frontend contract:
    result = answer_question(...)
    Must contain keys: answer (str), citations (list), status ('answered' | 'not_found' | 'conflict' | 'error')
    """
    mock_retrieve.return_value = [{"chunk_id": "c1", "text": "sample"}]
    mock_gen_answer.return_value = {
        "answer": "This is a tested answer.",
        "citations": [{"source_id": "c1", "document_id": "d1", "document_name": "f.pdf", "page_number": 1, "text": "sample"}],
        "status": "answered"
    }

    result = answer_question("What is Python?")

    assert isinstance(result, dict)
    assert "answer" in result and isinstance(result["answer"], str)
    assert "citations" in result and isinstance(result["citations"], list)
    assert result["status"] in ["answered", "not_found", "conflict", "error"]


def test_answer_question_empty_query():
    """Verify error status on empty query input."""
    result = answer_question("")
    assert result["status"] == "error"
    assert result["citations"] == []

