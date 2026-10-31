"""
DevDocs AI - Document Processor Test Suite
Tests for backend/document_processor.py.
"""

import math
import os
from pathlib import Path
from pypdf import PdfWriter

from backend.document_processor import (
    process_document,
    DocumentProcessingError,
)
from backend.embeddings import EMBEDDING_DIMENSION


PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_PDF = PROJECT_ROOT / "data" / "sample_doc.pdf"


def test_successful_processing():
    """Verify end-to-end processing of sample_doc.pdf with all expected fields."""
    print("\n--- Test: Successful Processing of Sample PDF ---")
    doc_id = "doc_devdocs_101"
    chunks = process_document(SAMPLE_PDF, document_id=doc_id)

    assert len(chunks) > 0, "Expected at least 1 chunk from sample_doc.pdf"
    print(f"  [INFO] Total chunks produced: {len(chunks)}")

    required_fields = [
        "document_id",
        "document_name",
        "chunk_id",
        "text",
        "page_number",
        "source_id",
        "content_type",
        "embedding",
    ]

    for idx, chunk in enumerate(chunks):
        # 1. Verify required fields exist
        for field in required_fields:
            assert field in chunk, f"Chunk {idx} is missing required field '{field}'"

        # 2. Verify document ID and document name
        assert chunk["document_id"] == doc_id, f"Expected doc_id '{doc_id}', got '{chunk['document_id']}'"
        assert chunk["document_name"] == "sample_doc.pdf", f"Expected 'sample_doc.pdf', got '{chunk['document_name']}'"

        # 3. Verify chunk metadata and IDs
        assert chunk["chunk_id"].startswith(f"{doc_id}_p"), f"Malformed chunk_id: {chunk['chunk_id']}"
        assert chunk["content_type"] == "text"
        assert chunk["source_id"] == "sample_doc.pdf"
        assert isinstance(chunk["page_number"], int) and chunk["page_number"] >= 1
        assert len(chunk["text"].strip()) > 0, "Chunk text should not be empty."

        # 4. Verify embedding dimension and finite values
        emb = chunk["embedding"]
        assert isinstance(emb, list), f"Chunk {idx} embedding must be a list."
        assert len(emb) == EMBEDDING_DIMENSION, f"Embedding dim {len(emb)} != {EMBEDDING_DIMENSION}"
        assert all(isinstance(v, float) for v in emb), f"Chunk {idx} contains non-float values in embedding."
        assert all(math.isfinite(v) for v in emb), f"Chunk {idx} contains non-finite values in embedding."

    print("  [PASS] Successfully verified fields, metadata, document ID, and 384-dim finite embeddings.")


def test_page_number_accuracy():
    """Verify that chunks accurately retain their source page numbers."""
    print("\n--- Test: Page Number Retention ---")
    chunks = process_document(SAMPLE_PDF, document_id="doc_page_check")

    page_numbers = [c["page_number"] for c in chunks]
    assert 1 in page_numbers, "Expected chunks for page 1"
    assert 2 in page_numbers, "Expected chunks for page 2"

    page_1_chunks = [c for c in chunks if c["page_number"] == 1]
    page_2_chunks = [c for c in chunks if c["page_number"] == 2]

    assert all("Page 1" in c["text"] or "Overview" in c["text"] for c in page_1_chunks)
    assert all("Page 2" in c["text"] or "Member 2" in c["text"] for c in page_2_chunks)
    print("  [PASS] Page numbers and citations correctly isolated.")


def test_configurable_chunk_size_and_overlap():
    """Verify that chunk_size and overlap parameters affect chunk generation."""
    print("\n--- Test: Configurable Chunk Size and Overlap ---")
    # Using small chunk_size to test configurability
    chunks_small = process_document(
        SAMPLE_PDF,
        document_id="doc_small_cfg",
        chunk_size=5,
        overlap=2
    )

    chunks_default = process_document(
        SAMPLE_PDF,
        document_id="doc_default_cfg",
        chunk_size=180,
        overlap=30
    )

    assert len(chunks_small) > len(chunks_default), (
        f"Smaller chunk_size (5) should yield more chunks than default (180). "
        f"Got {len(chunks_small)} vs {len(chunks_default)}"
    )
    print(f"  [PASS] Configurable chunking verified (small size: {len(chunks_small)} chunks, default size: {len(chunks_default)} chunks).")


def test_missing_pdf_handling():
    """Verify that missing PDF files raise DocumentProcessingError."""
    print("\n--- Test: Missing PDF Error Handling ---")
    missing_path = PROJECT_ROOT / "data" / "does_not_exist.pdf"
    try:
        process_document(missing_path, document_id="doc_missing")
        assert False, "Should raise DocumentProcessingError for missing file"
    except DocumentProcessingError as err:
        print(f"  [PASS] Correctly caught missing PDF error: {err}")


def test_empty_and_textless_pdf_handling():
    """Verify handling of 0-byte PDFs and textless/blank PDFs."""
    print("\n--- Test: Empty and Textless PDF Handling ---")
    # 1. Test 0-byte PDF (should raise DocumentProcessingError)
    empty_0byte_path = PROJECT_ROOT / "data" / "zero_bytes.pdf"
    with open(empty_0byte_path, "wb") as f:
        pass

    try:
        process_document(empty_0byte_path, document_id="doc_0byte")
        assert False, "Should raise DocumentProcessingError for 0-byte PDF"
    except DocumentProcessingError as err:
        print(f"  [PASS] Correctly caught 0-byte PDF error: {err}")
    finally:
        if empty_0byte_path.exists():
            os.remove(empty_0byte_path)

    # 2. Test blank page PDF (valid PDF structure, but no text content)
    blank_pdf_path = PROJECT_ROOT / "data" / "blank_page.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    with open(blank_pdf_path, "wb") as f:
        writer.write(f)

    try:
        res = process_document(blank_pdf_path, document_id="doc_blank")
        assert res == [], f"Expected empty list [] for blank textless PDF, got: {res}"
        print("  [PASS] Blank/textless PDF correctly handled returning empty chunk list [].")
    finally:
        if blank_pdf_path.exists():
            os.remove(blank_pdf_path)


def test_invalid_parameters():
    """Verify parameter validations for document_id and chunk configurations."""
    print("\n--- Test: Invalid Parameters Handling ---")
    # Empty document_id
    try:
        process_document(SAMPLE_PDF, document_id="")
        assert False, "Should raise ValueError for empty document_id"
    except ValueError:
        pass

    # Invalid chunk_size <= 0
    try:
        process_document(SAMPLE_PDF, document_id="valid_id", chunk_size=0)
        assert False, "Should raise ValueError for chunk_size=0"
    except ValueError:
        pass

    # Invalid overlap >= chunk_size
    try:
        process_document(SAMPLE_PDF, document_id="valid_id", chunk_size=50, overlap=50)
        assert False, "Should raise ValueError for overlap >= chunk_size"
    except ValueError:
        pass

    print("  [PASS] Parameter validations correctly enforced.")


def main():
    print("=" * 65)
    print("DevDocs AI - Document Processor Integration Test Suite")
    print("=" * 65)

    test_successful_processing()
    test_page_number_accuracy()
    test_configurable_chunk_size_and_overlap()
    test_missing_pdf_handling()
    test_empty_and_textless_pdf_handling()
    test_invalid_parameters()

    print("\n" + "=" * 65)
    print("ALL DOCUMENT PROCESSOR INTEGRATION TESTS PASSED SUCCESSFULLY.")
    print("=" * 65)


if __name__ == "__main__":
    main()
