"""
backend/evaluation.py - Retrieval Evaluation Metrics for DevDocs AI
===================================================================
Evaluates the retrieval quality of DevDocs AI using standard Information Retrieval (IR) metrics:
1. Hit Rate@K: Proportion of queries where at least one ground-truth relevant chunk was retrieved in the top K.
2. Precision@K: Fraction of retrieved top-K chunks that are relevant.
3. Recall@K: Fraction of all expected relevant chunks that were successfully retrieved in the top K.
"""

import logging
from typing import List, Dict, Any, Callable, Optional
from backend.rag_pipeline import retrieve_hybrid_chunks

logger = logging.getLogger("devdocs.evaluation")

# Ground-truth evaluation dataset for technical documentation
SAMPLE_TEST_CASES = [
    {
        "question": "How does Python handle exceptions?",
        "expected_document_ids": ["doc_001"],
        "expected_chunk_ids": ["chunk_001"]
    },
    {
        "question": "What is a Python decorator?",
        "expected_document_ids": ["doc_001"],
        "expected_chunk_ids": ["chunk_002"]
    },
    {
        "question": "How do generator functions and the yield keyword work?",
        "expected_document_ids": ["doc_002"],
        "expected_chunk_ids": ["chunk_005"]
    }
]


def calculate_hit_rate_at_k(retrieved_ids: List[str], expected_ids: List[str], k: int = 5) -> float:
    """
    Hit Rate@K: Returns 1.0 if at least one expected chunk is in top-K, otherwise 0.0.
    """
    if not expected_ids:
        return 0.0
    top_k_retrieved = set(retrieved_ids[:k])
    has_hit = bool(top_k_retrieved.intersection(set(expected_ids)))
    return 1.0 if has_hit else 0.0


def calculate_precision_at_k(retrieved_ids: List[str], expected_ids: List[str], k: int = 5) -> float:
    """
    Precision@K: (Number of relevant retrieved chunks in top K) / K
    """
    if k <= 0:
        return 0.0
    top_k_retrieved = set(retrieved_ids[:k])
    relevant_retrieved = top_k_retrieved.intersection(set(expected_ids))
    return len(relevant_retrieved) / float(k)


def calculate_recall_at_k(retrieved_ids: List[str], expected_ids: List[str], k: int = 5) -> float:
    """
    Recall@K: (Number of relevant retrieved chunks in top K) / (Total expected relevant chunks)
    """
    if not expected_ids:
        return 0.0
    top_k_retrieved = set(retrieved_ids[:k])
    relevant_retrieved = top_k_retrieved.intersection(set(expected_ids))
    return len(relevant_retrieved) / float(len(expected_ids))


def evaluate_retrieval(
    test_cases: List[Dict[str, Any]] = SAMPLE_TEST_CASES,
    retriever_fn: Optional[Callable[[str, int], List[Dict[str, Any]]]] = None,
    k: int = 5
) -> Dict[str, Any]:
    """
    Run evaluation across all test cases and return aggregate metrics.
    """
    if retriever_fn is None:
        def default_retriever(query: str, top_k: int):
            return retrieve_hybrid_chunks(query, top_k=top_k)
        retriever_fn = default_retriever

    hit_rates = []
    precisions = []
    recalls = []
    query_details = []

    for test_case in test_cases:
        query = test_case["question"]
        expected_chunks = test_case.get("expected_chunk_ids", [])

        # Retrieve chunks using hybrid search
        results = retriever_fn(query, k)
        retrieved_chunk_ids = [c.get("chunk_id", "") for c in results]

        # Calculate metrics
        hit = calculate_hit_rate_at_k(retrieved_chunk_ids, expected_chunks, k=k)
        prec = calculate_precision_at_k(retrieved_chunk_ids, expected_chunks, k=k)
        rec = calculate_recall_at_k(retrieved_chunk_ids, expected_chunks, k=k)

        hit_rates.append(hit)
        precisions.append(prec)
        recalls.append(rec)

        query_details.append({
            "question": query,
            "expected_chunk_ids": expected_chunks,
            "retrieved_chunk_ids": retrieved_chunk_ids[:k],
            "hit_rate": hit,
            "precision": round(prec, 4),
            "recall": round(rec, 4)
        })

    avg_hit_rate = sum(hit_rates) / max(len(hit_rates), 1)
    avg_precision = sum(precisions) / max(len(precisions), 1)
    avg_recall = sum(recalls) / max(len(recalls), 1)

    summary = {
        "k": k,
        "total_queries": len(test_cases),
        "mean_hit_rate@k": round(avg_hit_rate, 4),
        "mean_precision@k": round(avg_precision, 4),
        "mean_recall@k": round(avg_recall, 4),
        "details": query_details
    }

    return summary


def print_evaluation_report(report: Dict[str, Any]):
    """Print a clean CLI table report of retrieval performance."""
    k = report["k"]
    print("\n" + "=" * 65)
    print(f"       DevDocs AI - Retrieval Evaluation Report (K={k})")
    print("=" * 65)
    print(f"Total Test Queries : {report['total_queries']}")
    print(f"Mean Hit Rate@{k}   : {report['mean_hit_rate@k'] * 100:.1f}%")
    print(f"Mean Precision@{k}  : {report['mean_precision@k'] * 100:.1f}%")
    print(f"Mean Recall@{k}     : {report['mean_recall@k'] * 100:.1f}%")
    print("-" * 65)
    for idx, d in enumerate(report["details"], start=1):
        status = "HIT" if d["hit_rate"] > 0 else "MISS"
        print(f"[{status}] Q{idx}: {d['question']}")
        print(f"      Expected: {d['expected_chunk_ids']}")
        print(f"      Retrieved: {d['retrieved_chunk_ids']}")
        print(f"      Precision@{k}: {d['precision']:.2f} | Recall@{k}: {d['recall']:.2f}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    report = evaluate_retrieval()
    print_evaluation_report(report)

