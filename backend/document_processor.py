"""
DevDocs AI - Document Processor Module
Member 2 Responsibility: Document Processing and Embeddings

This module integrates PDF extraction, text chunking, and embedding generation
into a unified pipeline function: process_document().
"""

from pathlib import Path
from typing import Any, Dict, List, Union

from backend.pdf_reader import extract_pdf, PDFProcessingError
from backend.chunking import chunk_text
from backend.embeddings import generate_embeddings, get_embedding_dimension


class DocumentProcessingError(Exception):
    """Custom exception raised when document processing fails."""
    pass


def process_document(
    file_path: Union[str, Path],
    document_id: str,
    chunk_size: int = 180,
    overlap: int = 30
) -> List[Dict[str, Any]]:
    """
    Executes the end-to-end document processing pipeline:
    1. Validates inputs and extracts document filename as document_name.
    2. Extracts page-by-page text using backend.pdf_reader.extract_pdf.
    3. Splits text into word-level chunks using backend.chunking.chunk_text.
    4. Generates dense vector embeddings using backend.embeddings.generate_embeddings.

    Parameters:
        file_path (str or Path): Path to the target PDF document.
        document_id (str): Unique document identifier.
        chunk_size (int, optional): Number of words per chunk. Defaults to 180.
        overlap (int, optional): Number of overlapping words between consecutive chunks. Defaults to 30.

    Returns:
        List[Dict[str, Any]]: List of fully processed chunk dictionaries containing:
            - 'document_id' (str)
            - 'document_name' (str)
            - 'chunk_id' (str)
            - 'text' (str)
            - 'page_number' (int)
            - 'source_id' (str)
            - 'content_type' (str)
            - 'embedding' (List[float]): 384-dimensional normalized vector.

    Raises:
        DocumentProcessingError: If file is missing, low-level extraction fails, or embedding generation fails.
        ValueError: If document_id, chunk_size, or overlap are invalid.
    """
    # 1. Input validations
    if not file_path:
        raise DocumentProcessingError("No file_path was provided.")

    path_obj = Path(file_path)
    if not path_obj.exists():
        raise DocumentProcessingError(f"Document file not found: '{file_path}'")

    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("document_id must be a non-empty string.")

    # 1. Extract the PDF filename as document_name
    document_name = path_obj.name

    # 2. Extract PDF text per page
    try:
        pages = extract_pdf(path_obj)
    except PDFProcessingError as err:
        raise DocumentProcessingError(f"PDF extraction failed for '{document_name}': {err}") from err
    except Exception as err:
        raise DocumentProcessingError(f"Unexpected error during PDF extraction: {err}") from err

    # 3. Chunk text per page
    try:
        chunks = chunk_text(
            pages=pages,
            document_id=document_id.strip(),
            document_name=document_name,
            chunk_size=chunk_size,
            overlap=overlap
        )
    except ValueError as err:
        raise ValueError(f"Invalid chunking configuration: {err}") from err
    except Exception as err:
        raise DocumentProcessingError(f"Unexpected error during text chunking: {err}") from err

    # Return empty list if no chunks were generated (e.g., empty or image-only PDF without text)
    if not chunks:
        return []

    # 4. Generate embeddings for all chunks
    try:
        embedded_chunks = generate_embeddings(chunks)
    except Exception as err:
        raise DocumentProcessingError(f"Embedding generation failed for '{document_name}': {err}") from err

    return embedded_chunks
