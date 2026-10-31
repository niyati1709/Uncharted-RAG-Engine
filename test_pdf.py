"""
DevDocs AI - PDF Reader Test Script
Tests the extract_pdf function from backend/pdf_reader.py.
"""

from pathlib import Path
from backend.pdf_reader import extract_pdf, PDFProcessingError


def main():
    print("=" * 60)
    print("DevDocs AI - PDF Extraction Test")
    print("=" * 60)

    # Path to sample PDF in data/ directory
    project_root = Path(__file__).resolve().parent
    sample_pdf_path = project_root / "data" / "sample_doc.pdf"

    print(f"\n[INFO] Target PDF: {sample_pdf_path}")

    # 1. Test extraction on sample PDF
    try:
        pages = extract_pdf(sample_pdf_path)
        print(f"[SUCCESS] Extracted {len(pages)} page(s) successfully.\n")

        for page in pages:
            print("-" * 50)
            print(f"Page Number      : {page['page_number']}")
            print(f"Source File      : {page['source_file']}")
            print(f"Extraction Error : {page['extraction_error']}")
            print(f"Text Content:\n{page['text']}")
        print("-" * 50)

    except PDFProcessingError as error:
        print(f"[ERROR] PDF processing failed: {error}")
    except Exception as unexpected_error:
        print(f"[UNEXPECTED ERROR] Unexpected failure: {unexpected_error}")

    # 2. Test error handling behavior
    print("\n[INFO] Running error-handling validation checks:")

    # Check 1: Missing file
    missing_file = project_root / "data" / "missing_file.pdf"
    try:
        extract_pdf(missing_file)
        print("  [FAIL] Missing file was not caught.")
    except PDFProcessingError as err:
        print(f"  [PASS] Missing file caught: {err}")

    # Check 2: Invalid extension
    invalid_ext = project_root / "requirements.txt"
    try:
        extract_pdf(invalid_ext)
        print("  [FAIL] Invalid extension was not caught.")
    except PDFProcessingError as err:
        print(f"  [PASS] Invalid extension caught: {err}")

    print("\n" + "=" * 60)
    print("Test run completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
