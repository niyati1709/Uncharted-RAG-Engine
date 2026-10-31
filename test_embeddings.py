"""
DevDocs AI - Embedding Generation Test Suite
Comprehensive tests for backend/embeddings.py.
"""

import math
from typing import Any, Dict, List
import numpy as np

from backend.embeddings import (
    get_embedding_model,
    get_embedding_dimension,
    generate_embeddings,
    cosine_similarity,
    embed_query,
    EMBEDDING_DIMENSION,
)


def test_model_loading_and_caching():
    """Verify that the model loads properly and is cached as a singleton."""
    print("\n--- Test: Model Loading & Caching ---")
    model_1 = get_embedding_model()
    assert model_1 is not None, "Model should not be None."

    model_2 = get_embedding_model()
    assert model_1 is model_2, "get_embedding_model must return the cached singleton instance."
    print("  [PASS] Model loaded and caching verified (same instance returned).")


def test_embedding_dimension():
    """Verify that get_embedding_dimension returns 384."""
    print("\n--- Test: Embedding Dimension Constant ---")
    dim = get_embedding_dimension()
    assert dim == 384, f"Expected dimension 384, got: {dim}"
    assert dim == EMBEDDING_DIMENSION, "Dimension mismatch with constant."
    print("  [PASS] Embedding dimension correctly verified as 384.")


def test_single_and_multiple_chunk_embeddings():
    """Verify single and multi-chunk embedding generation and properties."""
    print("\n--- Test: Embedding Generation (Single & Multiple Chunks) ---")
    chunks = [
        {
            "document_id": "doc_01",
            "document_name": "api_guide.pdf",
            "chunk_id": "doc_01_p1_c1",
            "text": "FastAPI is a modern, fast web framework for building APIs with Python.",
            "page_number": 1,
            "content_type": "text",
            "source_id": "api_guide.pdf"
        },
        {
            "document_id": "doc_01",
            "document_name": "api_guide.pdf",
            "chunk_id": "doc_01_p1_c2",
            "text": "Authentication in FastAPI can be implemented using OAuth2 with Password and Bearer tokens.",
            "page_number": 1,
            "content_type": "text",
            "source_id": "api_guide.pdf"
        }
    ]

    # Test single chunk
    single_res = generate_embeddings([chunks[0]])
    assert len(single_res) == 1, "Expected 1 output chunk for single input."
    emb_single = single_res[0]["embedding"]

    assert isinstance(emb_single, list), f"Expected list, got {type(emb_single)}"
    assert len(emb_single) == 384, f"Expected 384 floats, got {len(emb_single)}"
    assert all(isinstance(val, float) for val in emb_single), "All elements must be Python floats."
    assert all(math.isfinite(val) for val in emb_single), "All embedding values must be finite."

    # Test multiple chunks
    multi_res = generate_embeddings(chunks)
    assert len(multi_res) == 2, "Expected 2 output chunks for multi input."

    for idx, c in enumerate(multi_res):
        emb = c["embedding"]
        assert isinstance(emb, list), f"Chunk {idx} embedding must be a list."
        assert len(emb) == 384, f"Chunk {idx} embedding must have length 384."
        assert all(isinstance(v, float) for v in emb), f"Chunk {idx} contains non-float values."
        assert all(math.isfinite(v) for v in emb), f"Chunk {idx} contains non-finite values."

        # Verify normalized vector (L2 norm ~ 1.0)
        norm = np.linalg.norm(np.array(emb))
        assert abs(norm - 1.0) < 1e-4, f"Chunk {idx} embedding should be normalized, got norm {norm}"

    print("  [PASS] Single and multiple chunk embeddings generated with dimension 384, finite values, and normalization.")


def test_immutability_and_metadata_preservation():
    """Verify that original inputs are not mutated and all metadata is preserved."""
    print("\n--- Test: Immutability & Metadata Preservation ---")
    original_chunk = {
        "document_id": "doc_test",
        "document_name": "spec.pdf",
        "chunk_id": "doc_test_p2_c1",
        "text": "Docker containers provide isolated runtime environments for applications.",
        "page_number": 2,
        "content_type": "text",
        "source_id": "spec.pdf",
        "custom_tag": "infrastructure"
    }

    # Pass in a list containing original_chunk
    input_chunks = [original_chunk]
    result = generate_embeddings(input_chunks)

    # 1. Assert input dictionary was NOT modified
    assert "embedding" not in original_chunk, "Original input dictionary must not be mutated!"
    assert "embedding" not in input_chunks[0], "Input list element was mutated!"

    # 2. Assert output dictionary preserves all original keys plus 'embedding'
    res_chunk = result[0]
    for key, value in original_chunk.items():
        assert res_chunk[key] == value, f"Metadata field '{key}' was altered: {res_chunk[key]} != {value}"

    assert "embedding" in res_chunk, "Output chunk must contain 'embedding' key."
    print("  [PASS] Original input dictionaries remained unmodified and metadata was preserved.")


def test_empty_input_handling():
    """Verify that passing an empty list returns an empty list without error."""
    print("\n--- Test: Empty Input Handling ---")
    res = generate_embeddings([])
    assert res == [], f"Expected empty list [], got: {res}"
    print("  [PASS] Empty list input correctly returns [].")


def test_invalid_input_handling():
    """Verify that malformed inputs raise appropriate exceptions."""
    print("\n--- Test: Invalid Input Handling ---")
    # Non-list input
    try:
        generate_embeddings("not a list")
        assert False, "Should raise TypeError for string input"
    except TypeError:
        pass

    # Non-dict chunk item
    try:
        generate_embeddings(["not a dict"])
        assert False, "Should raise TypeError for non-dict item"
    except TypeError:
        pass

    # Missing text field
    try:
        generate_embeddings([{"document_id": "d1", "page_number": 1}])
        assert False, "Should raise ValueError for missing 'text'"
    except ValueError:
        pass

    # Empty text content
    try:
        generate_embeddings([{"text": "   "}])
        assert False, "Should raise ValueError for whitespace-only text"
    except ValueError:
        pass

    print("  [PASS] Invalid inputs correctly raise TypeError and ValueError.")


def test_cosine_similarity():
    """Verify cosine similarity calculation, boundary values, and zero vector handling."""
    print("\n--- Test: Cosine Similarity Calculations ---")
    # Identical vectors
    v1 = [1.0, 2.0, 3.0]
    sim_identical = cosine_similarity(v1, v1)
    assert abs(sim_identical - 1.0) < 1e-6, f"Expected 1.0 for identical vectors, got {sim_identical}"

    # Orthogonal vectors
    v_x = [1.0, 0.0]
    v_y = [0.0, 1.0]
    sim_ortho = cosine_similarity(v_x, v_y)
    assert abs(sim_ortho - 0.0) < 1e-6, f"Expected 0.0 for orthogonal vectors, got {sim_ortho}"

    # Opposite vectors
    v_neg = [-1.0, -2.0, -3.0]
    sim_opp = cosine_similarity(v1, v_neg)
    assert abs(sim_opp - (-1.0)) < 1e-6, f"Expected -1.0 for opposite vectors, got {sim_opp}"

    # Zero vector handling (should return 0.0 without division by zero crash)
    v_zero = [0.0, 0.0, 0.0]
    sim_zero = cosine_similarity(v1, v_zero)
    assert sim_zero == 0.0, f"Expected 0.0 for zero vector, got {sim_zero}"

    # Dimension mismatch
    try:
        cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])
        assert False, "Should raise ValueError on dimension mismatch"
    except ValueError:
        pass

    print("  [PASS] Cosine similarity correctly handles identical, orthogonal, opposite, and zero vectors.")


def test_semantic_similarity_between_texts():
    """Verify that related technical sentences have higher cosine similarity than unrelated ones."""
    print("\n--- Test: Semantic Similarity on Developer Documentation Texts ---")
    text_a = "Configuring a database connection pool in PostgreSQL and Python."
    text_b = "How to set up PostgreSQL database connection settings."
    text_unrelated = "Delicious Italian pizza recipe baked with fresh mozzarella and tomato sauce."

    emb_a = embed_query(text_a)
    emb_b = embed_query(text_b)
    emb_unrelated = embed_query(text_unrelated)

    sim_related = cosine_similarity(emb_a, emb_b)
    sim_unrelated = cosine_similarity(emb_a, emb_unrelated)

    print(f"    Similarity (DB config vs. DB setup)   : {sim_related:.4f}")
    print(f"    Similarity (DB config vs. Pizza recipe): {sim_unrelated:.4f}")

    assert sim_related > sim_unrelated, (
        f"Related texts ({sim_related:.4f}) should have higher similarity than unrelated ({sim_unrelated:.4f})"
    )
    assert sim_related > 0.65, f"Expected high similarity (>0.65) for closely related text, got {sim_related:.4f}"
    assert sim_unrelated < 0.35, f"Expected low similarity (<0.35) for pizza recipe, got {sim_unrelated:.4f}"

    print("  [PASS] Semantic similarity correctly discriminates related technical documentation from unrelated topics.")


def main():
    print("=" * 65)
    print("DevDocs AI - Module 3: Embedding Generation Test Suite")
    print("=" * 65)

    test_model_loading_and_caching()
    test_embedding_dimension()
    test_single_and_multiple_chunk_embeddings()
    test_immutability_and_metadata_preservation()
    test_empty_input_handling()
    test_invalid_input_handling()
    test_cosine_similarity()
    test_semantic_similarity_between_texts()

    print("\n" + "=" * 65)
    print("ALL MODULE 3 EMBEDDING TESTS PASSED SUCCESSFULLY.")
    print("=" * 65)


if __name__ == "__main__":
    main()
