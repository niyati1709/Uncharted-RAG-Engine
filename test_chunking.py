"""
DevDocs AI - Text Chunking Test Suite
Comprehensive unit and integration tests for backend/chunking.py.
"""

import sys
from pathlib import Path
from backend.chunking import chunk_text
from backend.pdf_reader import extract_pdf


def test_short_page():
    """Verify that a page with fewer words than chunk_size produces exactly 1 chunk."""
    text = "DevDocs AI is an intelligent assistant designed to accelerate developer workflows."
    pages = [{"page_number": 1, "text": text, "source_file": "overview.pdf"}]
    
    chunks = chunk_text(pages, document_id="doc_001", document_name="overview.pdf", chunk_size=50, overlap=10)
    
    assert len(chunks) == 1, f"Expected 1 chunk, got {len(chunks)}"
    c = chunks[0]
    assert c["document_id"] == "doc_001"
    assert c["document_name"] == "overview.pdf"
    assert c["chunk_id"] == "doc_001_p1_c1"
    assert c["page_number"] == 1
    assert c["content_type"] == "text"
    assert c["source_id"] == "overview.pdf"
    assert c["text"] == text
    print("  [PASS] Short page test passed (1 chunk generated correctly).")


def test_long_page_and_overlap():
    """Verify that a long page creates multiple overlapping chunks and preserves overlap words."""
    # Generate 100 words: "word_0 word_1 ... word_99"
    words = [f"word_{i}" for i in range(100)]
    text = " ".join(words)
    pages = [{"page_number": 1, "text": text, "source_file": "manual.pdf"}]
    
    chunk_size = 40
    overlap = 10
    step = chunk_size - overlap  # 30
    # Expected chunks:
    # Chunk 1: indices 0..40 (40 words)
    # Chunk 2: indices 30..70 (40 words)
    # Chunk 3: indices 60..100 (40 words)
    chunks = chunk_text(pages, document_id="doc_long", document_name="manual.pdf", chunk_size=chunk_size, overlap=overlap)
    
    assert len(chunks) == 3, f"Expected 3 chunks, got {len(chunks)}"
    assert chunks[0]["chunk_id"] == "doc_long_p1_c1"
    assert chunks[1]["chunk_id"] == "doc_long_p1_c2"
    assert chunks[2]["chunk_id"] == "doc_long_p1_c3"
    
    # Verify overlap between Chunk 1 and Chunk 2
    c1_words = chunks[0]["text"].split()
    c2_words = chunks[1]["text"].split()
    overlap_1_2 = c1_words[-overlap:]
    expected_overlap_1_2 = c2_words[:overlap]
    assert overlap_1_2 == expected_overlap_1_2, f"Overlap mismatch between c1 and c2: {overlap_1_2} != {expected_overlap_1_2}"
    
    # Verify overlap between Chunk 2 and Chunk 3
    c3_words = chunks[2]["text"].split()
    overlap_2_3 = c2_words[-overlap:]
    expected_overlap_2_3 = c3_words[:overlap]
    assert overlap_2_3 == expected_overlap_2_3, f"Overlap mismatch between c2 and c3: {overlap_2_3} != {expected_overlap_2_3}"
    
    print("  [PASS] Long page test passed (3 chunks, exact word overlap verified).")


def test_two_pages():
    """Verify that multiple pages are processed independently and preserve page citations."""
    pages = [
        {"page_number": 1, "text": "Page one first sentence. Page one second sentence.", "source_file": "guide.pdf"},
        {"page_number": 2, "text": "Page two architecture details. Page two conclusions.", "source_file": "guide.pdf"}
    ]
    
    chunks = chunk_text(pages, document_id="doc_two", document_name="guide.pdf", chunk_size=50, overlap=10)
    
    assert len(chunks) == 2, f"Expected 2 chunks, got {len(chunks)}"
    assert chunks[0]["page_number"] == 1
    assert chunks[0]["chunk_id"] == "doc_two_p1_c1"
    assert "Page one" in chunks[0]["text"]
    assert "Page two" not in chunks[0]["text"]
    
    assert chunks[1]["page_number"] == 2
    assert chunks[1]["chunk_id"] == "doc_two_p2_c1"
    assert "Page two" in chunks[1]["text"]
    assert "Page one" not in chunks[1]["text"]
    print("  [PASS] Two pages test passed (page boundaries and citations accurately preserved).")


def test_empty_page():
    """Verify that empty or whitespace-only pages are skipped."""
    pages = [
        {"page_number": 1, "text": "Valid content on page one.", "source_file": "mixed.pdf"},
        {"page_number": 2, "text": "   \n\t   ", "source_file": "mixed.pdf"},
        {"page_number": 3, "text": "", "source_file": "mixed.pdf"},
        {"page_number": 4, "text": "Valid content on page four.", "source_file": "mixed.pdf"}
    ]
    
    chunks = chunk_text(pages, document_id="doc_empty", document_name="mixed.pdf", chunk_size=50, overlap=10)
    
    assert len(chunks) == 2, f"Expected 2 chunks (pages 1 and 4), got {len(chunks)}"
    assert chunks[0]["page_number"] == 1
    assert chunks[0]["chunk_id"] == "doc_empty_p1_c1"
    assert chunks[1]["page_number"] == 4
    assert chunks[1]["chunk_id"] == "doc_empty_p4_c1"
    print("  [PASS] Empty page test passed (empty/whitespace pages successfully skipped).")


def test_invalid_chunk_size():
    """Verify that invalid chunk_size values raise ValueError."""
    pages = [{"page_number": 1, "text": "Some text", "source_file": "test.pdf"}]
    
    for invalid_size in [0, -10, "100", 1.5, None, False]:
        try:
            chunk_text(pages, "doc_id", "doc_name", chunk_size=invalid_size, overlap=0)
            assert False, f"Expected ValueError for chunk_size={invalid_size}"
        except ValueError:
            pass
    print("  [PASS] Invalid chunk_size validation passed (0, negative, and non-int rejected).")


def test_invalid_overlap():
    """Verify that negative overlap or overlap >= chunk_size raise ValueError."""
    pages = [{"page_number": 1, "text": "Some text", "source_file": "test.pdf"}]
    
    # overlap < 0
    for invalid_overlap in [-1, -50, "10", 2.5, None, False]:
        try:
            chunk_text(pages, "doc_id", "doc_name", chunk_size=100, overlap=invalid_overlap)
            assert False, f"Expected ValueError for overlap={invalid_overlap}"
        except ValueError:
            pass
            
    # overlap >= chunk_size
    for bad_overlap in [100, 150]:
        try:
            chunk_text(pages, "doc_id", "doc_name", chunk_size=100, overlap=bad_overlap)
            assert False, f"Expected ValueError for overlap={bad_overlap} >= chunk_size=100"
        except ValueError:
            pass
    print("  [PASS] Invalid overlap validation passed (negative and overlap >= chunk_size rejected).")


def test_integration_with_pdf_reader():
    """Verify chunk_text on actual output from backend/pdf_reader.py."""
    sample_pdf = Path(__file__).resolve().parent / "data" / "sample_doc.pdf"
    if not sample_pdf.exists():
        print("  [SKIP] Integration test skipped: sample_doc.pdf not found.")
        return
        
    pages = extract_pdf(sample_pdf)
    chunks = chunk_text(pages, document_id="devdocs_001", document_name="sample_doc.pdf", chunk_size=180, overlap=30)
    
    assert len(chunks) == 2, f"Expected 2 chunks from 2 pages, got {len(chunks)}"
    assert chunks[0]["page_number"] == 1
    assert chunks[0]["chunk_id"] == "devdocs_001_p1_c1"
    assert "Overview" in chunks[0]["text"]
    
    assert chunks[1]["page_number"] == 2
    assert chunks[1]["chunk_id"] == "devdocs_001_p2_c1"
    assert "Member 2" in chunks[1]["text"]
    print("  [PASS] Integration test with backend/pdf_reader.py passed.")


def main():
    print("=" * 60)
    print("DevDocs AI - Text Chunking Module Test Suite")
    print("=" * 60)
    
    test_short_page()
    test_long_page_and_overlap()
    test_two_pages()
    test_empty_page()
    test_invalid_chunk_size()
    test_invalid_overlap()
    test_integration_with_pdf_reader()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY.")
    print("=" * 60)


if __name__ == "__main__":
    main()
