"""
DevDocs AI - Text Chunking Module
Member 2 Responsibility: Document Processing and Embeddings

This module divides extracted PDF text into smaller, overlapping word-based chunks
while preserving page-level metadata and citation information.
"""

from typing import Any, Dict, List


def chunk_text(
    pages: List[Dict[str, Any]],
    document_id: str,
    document_name: str,
    chunk_size: int = 180,
    overlap: int = 30
) -> List[Dict[str, Any]]:
    """
    Splits page-by-page text into word-based overlapping chunks.

    Parameters:
        pages (List[Dict[str, Any]]): List of page dictionaries, typically returned by
                                      backend.pdf_reader.extract_pdf. Each dictionary should have
                                      'page_number', 'text', and optionally 'source_file'.
        document_id (str): Unique identifier for the document.
        document_name (str): Human-readable name or filename of the document.
        chunk_size (int, optional): Number of words per chunk. Must be > 0. Defaults to 180.
        overlap (int, optional): Number of overlapping words between consecutive chunks.
                                 Must be >= 0 and strictly less than chunk_size. Defaults to 30.

    Returns:
        List[Dict[str, Any]]: A list of chunk dictionaries, each containing:
            - 'document_id' (str)
            - 'document_name' (str)
            - 'chunk_id' (str): Deterministic ID: '{document_id}_p{page_number}_c{chunk_number}'
            - 'text' (str): Text content of the chunk
            - 'page_number' (int): Originating page number (1-indexed)
            - 'content_type' (str): Content type label (e.g., 'text')
            - 'source_id' (str): Originating source file or document identifier

    Raises:
        ValueError: If chunk_size or overlap are invalid, if document_id/document_name
                    are empty strings, or if page dictionaries are malformed.
        TypeError: If pages is not a list.
    """
    # 1. Parameter Validations
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("document_id must be a non-empty string.")

    if not isinstance(document_name, str) or not document_name.strip():
        raise ValueError("document_name must be a non-empty string.")

    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive integer (> 0), got: {chunk_size}")

    if not isinstance(overlap, int) or isinstance(overlap, bool) or overlap < 0:
        raise ValueError(f"overlap must be a non-negative integer (>= 0), got: {overlap}")

    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be strictly less than chunk_size ({chunk_size})."
        )

    if not isinstance(pages, list):
        raise TypeError(f"pages must be a list of dictionaries, got: {type(pages).__name__}")

    step = chunk_size - overlap
    all_chunks: List[Dict[str, Any]] = []

    # 2. Iterate through each page independently to preserve accurate citations
    for page_index, page in enumerate(pages):
        if not isinstance(page, dict):
            raise ValueError(
                f"Malformed page at index {page_index}: expected dict, got {type(page).__name__}"
            )

        # Validate page number
        page_number = page.get("page_number")
        if page_number is None or not isinstance(page_number, int) or isinstance(page_number, bool):
            raise ValueError(
                f"Malformed page at index {page_index}: missing or invalid 'page_number' (got: {page_number})"
            )

        # Determine source_id (fallback to source_file, document_name, or document_id)
        source_id = str(page.get("source_file") or document_name or document_id)

        # Extract raw text
        raw_text = page.get("text")
        if raw_text is None:
            continue

        if not isinstance(raw_text, str):
            raw_text = str(raw_text)

        clean_text = raw_text.strip()
        # Requirement 8: Skip empty pages
        if not clean_text:
            continue

        words = clean_text.split()
        total_words = len(words)
        if total_words == 0:
            continue

        # 3. Create word-based chunks with sliding window
        chunk_num = 1
        start_idx = 0

        while start_idx < total_words:
            end_idx = min(start_idx + chunk_size, total_words)
            chunk_words = words[start_idx:end_idx]
            chunk_content = " ".join(chunk_words)

            # Deterministic chunk ID: {document_id}_p{page_number}_c{chunk_num}
            chunk_id = f"{document_id}_p{page_number}_c{chunk_num}"

            all_chunks.append({
                "document_id": document_id,
                "document_name": document_name,
                "chunk_id": chunk_id,
                "text": chunk_content,
                "page_number": page_number,
                "content_type": "text",
                "source_id": source_id,
            })

            chunk_num += 1

            # Stop once the last word of this page is included
            if end_idx >= total_words:
                break

            start_idx += step

    return all_chunks
