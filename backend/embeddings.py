"""
DevDocs AI - Embedding Generation Module
Member 2 Responsibility: Document Processing and Embeddings

This module generates 384-dimensional dense vector embeddings using the
sentence-transformers/all-MiniLM-L6-v2 model and computes cosine similarity
with NumPy.
"""

from copy import deepcopy
from typing import Any, Dict, List, Optional, Union
import numpy as np

# Model configuration constants (DO NOT CHANGE)
MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION: int = 384

# Module-level model cache to avoid repeated initialization
_CACHED_MODEL: Any = None


def get_embedding_model():
    """
    Loads and caches the SentenceTransformer model.
    Ensures the model is loaded once and reused across all operations.

    Returns:
        SentenceTransformer: The loaded all-MiniLM-L6-v2 model instance.
    """
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _CACHED_MODEL = SentenceTransformer(MODEL_NAME)
    return _CACHED_MODEL


def get_embedding_dimension() -> int:
    """
    Returns the fixed embedding dimension produced by the model (384).

    Returns:
        int: The embedding dimension size (384).
    """
    return EMBEDDING_DIMENSION


def generate_embeddings(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generates 384-dimensional normalized vector embeddings for each chunk's text.

    Parameters:
        chunks (List[Dict[str, Any]]): List of chunk dictionaries (typically from
                                      backend.chunking.chunk_text). Each chunk must
                                      contain a non-empty 'text' string.

    Returns:
        List[Dict[str, Any]]: A new list of chunk dictionaries with an added 'embedding'
                              field containing a Python list of 384 floating-point numbers.
                              The original input dictionaries are not modified.

    Raises:
        TypeError: If chunks is not a list or if any element is not a dictionary.
        ValueError: If any chunk is missing the 'text' field or if 'text' is empty/whitespace.
    """
    # 1. Validate container type
    if not isinstance(chunks, list):
        raise TypeError(f"chunks must be a list of dictionaries, got: {type(chunks).__name__}")

    # Handle empty input gracefully
    if len(chunks) == 0:
        return []

    # 2. Validate individual chunk items and collect texts
    texts_to_embed: List[str] = []
    validated_chunks: List[Dict[str, Any]] = []

    for idx, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise TypeError(
                f"Chunk at index {idx} is invalid: expected dict, got {type(chunk).__name__}"
            )

        if "text" not in chunk:
            raise ValueError(f"Chunk at index {idx} is missing the required 'text' field.")

        raw_text = chunk["text"]
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise ValueError(
                f"Chunk at index {idx} has invalid or empty 'text' content."
            )

        # Deepcopy to preserve original input dictionaries unmodified
        validated_chunks.append(deepcopy(chunk))
        texts_to_embed.append(raw_text.strip())

    # 3. Batch encode using the cached model with normalization
    model = get_embedding_model()
    raw_embeddings = model.encode(
        texts_to_embed,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    # Convert to pure Python list of floats
    if hasattr(raw_embeddings, "tolist"):
        embeddings_list = raw_embeddings.tolist()
    else:
        embeddings_list = [np.asarray(e, dtype=np.float32).tolist() for e in raw_embeddings]

    # 4. Attach embeddings to cloned chunk dictionaries
    for idx, chunk_copy in enumerate(validated_chunks):
        embedding_floats = [float(val) for val in embeddings_list[idx]]

        # Ensure exact dimension
        if len(embedding_floats) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Generated embedding dimension {len(embedding_floats)} does not match expected {EMBEDDING_DIMENSION}."
            )

        chunk_copy["embedding"] = embedding_floats

    return validated_chunks


def cosine_similarity(
    vector_a: Union[List[float], np.ndarray],
    vector_b: Union[List[float], np.ndarray]
) -> float:
    """
    Computes cosine similarity between two vector embeddings using NumPy.

    Parameters:
        vector_a (List[float] or np.ndarray): First embedding vector.
        vector_b (List[float] or np.ndarray): Second embedding vector.

    Returns:
        float: Cosine similarity score between -1.0 and 1.0 (or 0.0 for zero vectors).

    Raises:
        TypeError: If inputs cannot be converted to numeric NumPy arrays.
        ValueError: If vectors are not 1-dimensional or if shapes do not match.
    """
    try:
        vec_a = np.asarray(vector_a, dtype=np.float64)
        vec_b = np.asarray(vector_b, dtype=np.float64)
    except Exception as err:
        raise TypeError(f"Failed to convert inputs to numerical arrays: {err}") from err

    if vec_a.ndim != 1 or vec_b.ndim != 1:
        raise ValueError(
            f"Input vectors must be 1-dimensional, got shapes {vec_a.shape} and {vec_b.shape}."
        )

    if vec_a.shape != vec_b.shape:
        raise ValueError(
            f"Vector dimensions do not match: {vec_a.shape} vs {vec_b.shape}."
        )

    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))

    # Safe handling of zero or near-zero vectors
    if norm_a == 0.0 or norm_b == 0.0 or not np.isfinite(norm_a) or not np.isfinite(norm_b):
        return 0.0

    similarity = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    # Clip bounds to compensate for floating-point imprecision
    return max(-1.0, min(1.0, similarity))


def embed_query(query: str) -> List[float]:
    """
    Generates a normalized embedding for a single user query string using the cached model.
    Prepared for subsequent query retrieval workflows.

    Parameters:
        query (str): The search query or developer question.

    Returns:
        List[float]: A 384-dimensional list of floating-point numbers.

    Raises:
        ValueError: If query is not a non-empty string.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")

    model = get_embedding_model()
    raw_emb = model.encode(query.strip(), normalize_embeddings=True, show_progress_bar=False)

    if hasattr(raw_emb, "tolist"):
        emb_list = raw_emb.tolist()
    else:
        emb_list = [float(x) for x in raw_emb]

    return [float(x) for x in emb_list]
