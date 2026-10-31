"""
tests/test_evaluation.py - Unit Tests for backend/evaluation.py
==============================================================
Tests Hit Rate@K, Precision@K, Recall@K, and aggregate metric reporting.
"""

from backend.evaluation import (
    calculate_hit_rate_at_k,
    calculate_precision_at_k,
    calculate_recall_at_k,
    evaluate_retrieval,
)


def test_calculate_hit_rate_at_k():
    """Test Hit Rate@K returns 1.0 on hit, 0.0 on miss."""
    retrieved = ["chunk_A", "chunk_B", "chunk_C"]
    expected = ["chunk_B"]

    # Hit within top 2
    assert calculate_hit_rate_at_k(retrieved, expected, k=2) == 1.0

    # Miss when k=1 (chunk_B is at index 1 / rank 2)
    assert calculate_hit_rate_at_k(retrieved, expected, k=1) == 0.0

    # Miss when not in retrieved
    assert calculate_hit_rate_at_k(retrieved, ["chunk_Z"], k=5) == 0.0


def test_calculate_precision_at_k():
    """Test Precision@K calculates relevant / k."""
    retrieved = ["chunk_A", "chunk_B", "chunk_C", "chunk_D", "chunk_E"]
    expected = ["chunk_A", "chunk_C"]

    # In top 5: chunk_A and chunk_C are relevant -> 2 / 5 = 0.4
    assert calculate_precision_at_k(retrieved, expected, k=5) == 0.4

    # In top 2: only chunk_A is relevant -> 1 / 2 = 0.5
    assert calculate_precision_at_k(retrieved, expected, k=2) == 0.5

    # In top 1: chunk_A is relevant -> 1 / 1 = 1.0
    assert calculate_precision_at_k(retrieved, expected, k=1) == 1.0


def test_calculate_recall_at_k():
    """Test Recall@K calculates relevant / total_expected."""
    retrieved = ["chunk_A", "chunk_B", "chunk_C", "chunk_D", "chunk_E"]
    expected = ["chunk_A", "chunk_Z"]  # 2 expected chunks, but only chunk_A retrieved

    # 1 of 2 retrieved in top 5 -> 0.5
    assert calculate_recall_at_k(retrieved, expected, k=5) == 0.5

    # Both retrieved
    assert calculate_recall_at_k(retrieved, ["chunk_A", "chunk_B"], k=5) == 1.0


def test_evaluate_retrieval_aggregate():
    """Test full evaluation pipeline with mock retriever."""
    test_cases = [
        {
            "question": "What is Python?",
            "expected_chunk_ids": ["c1"]
        },
        {
            "question": "What is Rust?",
            "expected_chunk_ids": ["c99"]
        }
    ]

    def mock_retriever(query, top_k):
        # Query 1 finds c1; Query 2 returns other chunks
        if "Python" in query:
            return [{"chunk_id": "c1"}, {"chunk_id": "c2"}]
        return [{"chunk_id": "c3"}, {"chunk_id": "c4"}]

    report = evaluate_retrieval(test_cases=test_cases, retriever_fn=mock_retriever, k=2)

    assert report["total_queries"] == 2
    # Q1 hit (1.0), Q2 miss (0.0) -> mean hit rate = 0.5
    assert report["mean_hit_rate@k"] == 0.5
    # Q1 prec = 1/2 = 0.5, Q2 prec = 0/2 = 0.0 -> mean prec = 0.25
    assert report["mean_precision@k"] == 0.25

