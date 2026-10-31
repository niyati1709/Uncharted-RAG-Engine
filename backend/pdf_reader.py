"""
DevDocs AI - PDF Reader Module
Member 2 Responsibility: Document Processing and Embeddings

This module provides functionality to extract text from PDF documents page-by-page
using the pypdf library, with robust validation and error handling.
"""

import os
from typing import Any, Dict, List, Union
from pathlib import Path
from pypdf import PdfReader
from pypdf.errors import PdfReadError, FileNotDecryptedError


class PDFProcessingError(Exception):
    """Custom exception raised when an error occurs during PDF validation or processing."""
    pass


def extract_pdf(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Extracts text from each page of a given PDF file.

    Parameters:
        file_path (str or Path): The path to the PDF file to be processed.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries representing each page.
            Each dictionary contains:
                - 'page_number' (int): The 1-indexed page number.
                - 'text' (str): Extracted text content from the page.
                - 'source_file' (str): The filename of the source PDF.
                - 'extraction_error' (str or None): Error message if page extraction failed, else None.

    Raises:
        PDFProcessingError: If the file does not exist, has an invalid extension,
                            is empty, is encrypted/password-protected, or cannot be parsed.
    """
    # 1. Validate file path input
    if not file_path:
        raise PDFProcessingError("No file path was provided.")

    path_obj = Path(file_path)

    # 2. Check if the file exists and is a regular file
    if not path_obj.exists():
        raise PDFProcessingError(f"File not found: '{file_path}'")

    if not path_obj.is_file():
        raise PDFProcessingError(f"Provided path is not a valid file: '{file_path}'")

    # 3. Check for valid PDF extension (.pdf)
    if path_obj.suffix.lower() != ".pdf":
        raise PDFProcessingError(
            f"Invalid file extension '{path_obj.suffix}'. Only .pdf files are supported."
        )

    # 4. Check for empty file (0 bytes)
    try:
        if path_obj.stat().st_size == 0:
            raise PDFProcessingError(f"The PDF file is empty (0 bytes): '{path_obj.name}'")
    except OSError as err:
        raise PDFProcessingError(f"Failed to access file metadata: {err}") from err

    # 5. Load and parse the PDF with pypdf
    try:
        reader = PdfReader(str(path_obj))
    except (PdfReadError, Exception) as err:
        raise PDFProcessingError(f"Failed to parse PDF '{path_obj.name}': {err}") from err

    # 6. Check for password protection / encryption
    if reader.is_encrypted:
        try:
            # Check if decrypting with an empty password succeeds
            decrypt_status = reader.decrypt("")
            if decrypt_status == 0:  # Decryption failed; requires password
                raise PDFProcessingError(
                    f"PDF file '{path_obj.name}' is password-protected and cannot be read."
                )
        except (FileNotDecryptedError, Exception) as err:
            raise PDFProcessingError(
                f"PDF file '{path_obj.name}' is encrypted or password-protected: {err}"
            ) from err

    # 7. Check if PDF contains any pages
    if not reader.pages or len(reader.pages) == 0:
        raise PDFProcessingError(f"The PDF file '{path_obj.name}' contains no pages.")

    # 8. Extract text from each page separately
    pages_data: List[Dict[str, Any]] = []
    source_filename = path_obj.name

    for page_idx, page in enumerate(reader.pages, start=1):
        try:
            extracted = page.extract_text()
            # If text is None (e.g. image-only page without OCR), default to empty string
            page_text = extracted.strip() if extracted else ""
            pages_data.append({
                "page_number": page_idx,
                "text": page_text,
                "source_file": source_filename,
                "extraction_error": None
            })
        except Exception as page_err:
            # Catch individual page extraction failures without crashing the entire extraction
            pages_data.append({
                "page_number": page_idx,
                "text": "",
                "source_file": source_filename,
                "extraction_error": f"Error extracting page {page_idx}: {page_err}"
            })

    return pages_data
