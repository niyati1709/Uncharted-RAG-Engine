"""
backend_adapter.py - Integration Layer between DevDocs AI Frontend & Backend

This module serves as the contract and bridge between the Streamlit frontend
and the RAG backend (developed by Team Members 2 & 3).

Frontend Integration Contracts:
1. ingest_document(file_path: str, use_mock: bool = True) -> dict
2. list_documents(use_mock: bool = True) -> list
3. answer_question(question: str, use_mock: bool = True) -> dict

Strict Separation Rules:
- Mock Mode: Uses simulated responses and storage; explicitly flags responses as mock.
- Live Mode: Directs calls to `backend.rag_pipeline`. Never silently falls back to mock.
  If the backend is missing, incomplete, or fails, meaningful errors are raised or returned.
- Independence: This adapter contains zero Streamlit dependencies.
"""

import os
import sys
import time
import importlib
import hashlib
from typing import List, Dict, Any, Optional

REQUIRED_BACKEND_FUNCTIONS = ["ingest_document", "list_documents", "answer_question"]


# -----------------------------------------------------------------------------
# 1. CUSTOM BACKEND EXCEPTIONS
# -----------------------------------------------------------------------------
class BackendError(Exception):
    """Base exception for backend operations."""
    pass


class BackendUnavailableError(BackendError):
    """Raised when backend.rag_pipeline cannot be imported or found."""
    pass


class BackendIncompleteError(BackendError):
    """Raised when backend.rag_pipeline is missing required callable functions."""
    pass


class BackendExecutionError(BackendError):
    """Raised when an actual backend function encounters an unhandled exception."""
    pass


# -----------------------------------------------------------------------------
# 2. BACKEND DETECTION & VERIFICATION
# -----------------------------------------------------------------------------
def check_backend_status() -> Dict[str, Any]:
    """
    Checks whether `backend.rag_pipeline` exists, can be imported, and contains
    all three required callable functions:
      - ingest_document(file_path: str) -> dict
      - list_documents() -> list
      - answer_question(question: str) -> dict

    Returns:
        dict: {
            "available": bool,
            "module_imported": bool,
            "missing_functions": List[str],
            "error_message": Optional[str],
            "module": Optional[Any]
        }
    """
    try:
        # Dynamically import or reload module to support runtime additions
        if "backend.rag_pipeline" in sys.modules:
            module = sys.modules["backend.rag_pipeline"]
            if hasattr(module, "__file__") and module.__file__:
                try:
                    module = importlib.reload(module)
                except Exception:
                    pass
        else:
            module = importlib.import_module("backend.rag_pipeline")
    except (ImportError, ModuleNotFoundError) as e:
        return {
            "available": False,
            "module_imported": False,
            "missing_functions": list(REQUIRED_BACKEND_FUNCTIONS),
            "error_message": f"backend.rag_pipeline is not available: {str(e)}",
            "module": None
        }
    except Exception as e:
        return {
            "available": False,
            "module_imported": False,
            "missing_functions": list(REQUIRED_BACKEND_FUNCTIONS),
            "error_message": f"Failed to load backend.rag_pipeline: {str(e)}",
            "module": None
        }

    # Verify all required functions exist and are callable
    missing = []
    for fn in REQUIRED_BACKEND_FUNCTIONS:
        if not hasattr(module, fn) or not callable(getattr(module, fn)):
            missing.append(fn)

    if missing:
        return {
            "available": False,
            "module_imported": True,
            "missing_functions": missing,
            "error_message": f"backend.rag_pipeline is missing callable function(s): {', '.join(missing)}",
            "module": module
        }

    return {
        "available": True,
        "module_imported": True,
        "missing_functions": [],
        "error_message": None,
        "module": module
    }


def is_backend_available() -> bool:
    """Returns True ONLY when backend.rag_pipeline exists and has all 3 required functions."""
    return check_backend_status()["available"]


# -----------------------------------------------------------------------------
# 3. MOCK STORAGE MANAGEMENT (STARTS EMPTY BY DEFAULT)
# -----------------------------------------------------------------------------
# In-memory mock storage starts EMPTY by default.
# Simulated documents are explicitly marked with `is_mock=True` and `status="simulated_indexed"`.
_MOCK_DOCUMENTS: List[Dict[str, Any]] = []


def get_sample_demo_document() -> Dict[str, Any]:
    """Provides a sample simulated document for optional demo testing."""
    return {
        "document_id": "demo_doc_001",
        "filename": "fastapi_quickstart_guide.pdf (Demo)",
        "status": "simulated_indexed",
        "total_pages": 42,
        "total_chunks": 168,
        "is_mock": True
    }


def load_demo_documents() -> List[Dict[str, Any]]:
    """Loads a sample simulated document into mock store if empty."""
    global _MOCK_DOCUMENTS
    sample = get_sample_demo_document()
    if not any(d.get("document_id") == sample["document_id"] for d in _MOCK_DOCUMENTS):
        _MOCK_DOCUMENTS.append(sample)
    return list(_MOCK_DOCUMENTS)


def clear_mock_documents() -> None:
    """Resets mock document store to an empty list."""
    global _MOCK_DOCUMENTS
    _MOCK_DOCUMENTS = []


# -----------------------------------------------------------------------------
# 4. BACKEND INTEGRATION FUNCTIONS
# -----------------------------------------------------------------------------
def ingest_document(file_path: str, use_mock: bool = True) -> Dict[str, Any]:
    """
    Ingest a PDF document into the RAG vector store.

    Parameters:
        file_path (str): The local path to the saved PDF file.
        use_mock (bool): If True, uses mock pipeline. If False, calls live backend.

    Returns:
        dict: {
            "success": bool,
            "document_id": str,
            "filename": str,
            "status": str,
            "message": str,
            "is_mock": bool
        }
    """
    filename = os.path.basename(file_path)

    # 1. LIVE BACKEND MODE - Strict Execution
    if not use_mock:
        status_info = check_backend_status()
        if not status_info["available"]:
            return {
                "success": False,
                "document_id": "",
                "filename": filename,
                "status": "failed",
                "message": f"Live Backend Error: {status_info['error_message']}",
                "is_mock": False
            }

        real_rag = status_info["module"]
        try:
            result = real_rag.ingest_document(file_path)
            if not isinstance(result, dict):
                return {
                    "success": False,
                    "document_id": "",
                    "filename": filename,
                    "status": "failed",
                    "message": "Live backend returned invalid format (expected dict).",
                    "is_mock": False
                }
            result["is_mock"] = False
            return result
        except Exception as e:
            return {
                "success": False,
                "document_id": "",
                "filename": filename,
                "status": "failed",
                "message": f"Live backend execution error during ingestion: {str(e)}",
                "is_mock": False
            }

    # 2. MOCK MODE - Clearly Simulated
    time.sleep(0.5)

    # Compute content hash if file exists
    file_hash = ""
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
        except Exception:
            file_hash = ""

    # Check for duplicate in mock store by content hash or exact file path
    for doc in _MOCK_DOCUMENTS:
        if (file_hash and doc.get("file_hash") == file_hash) or doc.get("file_path") == file_path:
            return {
                "success": True,
                "document_id": doc["document_id"],
                "filename": doc["filename"],
                "status": "simulated_indexed",
                "message": f"Simulated document '{doc['filename']}' is already in mock storage.",
                "is_mock": True
            }

    # If same filename exists but content differs, assign a disambiguated display name
    display_name = filename
    existing_names = [d["filename"] for d in _MOCK_DOCUMENTS]
    if display_name in existing_names:
        base, ext = os.path.splitext(filename)
        count = sum(1 for d in existing_names if d.startswith(base))
        display_name = f"{base} ({count + 1}){ext}"

    estimated_pages = 15
    if os.path.exists(file_path):
        size_kb = os.path.getsize(file_path) / 1024
        estimated_pages = max(1, int(size_kb / 40))

    doc_id = f"mock_doc_{len(_MOCK_DOCUMENTS) + 1:03d}"
    chunks = estimated_pages * 4

    new_doc = {
        "document_id": doc_id,
        "filename": display_name,
        "file_path": file_path,
        "file_hash": file_hash,
        "status": "simulated_indexed",
        "total_pages": estimated_pages,
        "total_chunks": chunks,
        "is_mock": True
    }
    _MOCK_DOCUMENTS.append(new_doc)

    return {
        "success": True,
        "document_id": doc_id,
        "filename": display_name,
        "status": "simulated_indexed",
        "message": f"Document '{display_name}' processed in Mock Mode ({estimated_pages} pages, {chunks} chunks).",
        "is_mock": True
    }


def list_documents(use_mock: bool = True) -> List[Dict[str, Any]]:
    """
    Retrieve all documents indexed in the knowledge base.

    Parameters:
        use_mock (bool): If True, returns mock store. If False, calls live backend.

    Returns:
        list of dict: [
            {
                "document_id": "doc_001",
                "filename": "python_guide.pdf",
                "status": "indexed",
                "total_pages": 25,
                "total_chunks": 100,
                "is_mock": bool
            }
        ]

    Raises:
        BackendUnavailableError: If live mode is requested but backend module is missing.
        BackendIncompleteError: If live mode is requested but backend is missing functions.
        BackendExecutionError: If live backend raises an unhandled error.
    """
    # 1. LIVE BACKEND MODE - Strict Execution
    if not use_mock:
        status_info = check_backend_status()
        if not status_info["module_imported"]:
            raise BackendUnavailableError(status_info["error_message"])
        if status_info["missing_functions"]:
            raise BackendIncompleteError(status_info["error_message"])

        real_rag = status_info["module"]
        try:
            docs = real_rag.list_documents()
            if not isinstance(docs, list):
                raise BackendExecutionError("Live backend list_documents() did not return a list.")
            for doc in docs:
                if isinstance(doc, dict):
                    doc["is_mock"] = False
            return docs
        except BackendError:
            raise
        except Exception as e:
            raise BackendExecutionError(f"Live backend error in list_documents: {str(e)}") from e

    # 2. MOCK MODE - Return copy of in-memory simulated documents
    return [dict(d) for d in _MOCK_DOCUMENTS]


def answer_question(question: str, use_mock: bool = True) -> Dict[str, Any]:
    """
    Query the RAG pipeline with a developer question and obtain an answer with citations.

    Parameters:
        question (str): The natural language query from the developer.
        use_mock (bool): If True, returns simulated answer. If False, calls live backend.

    Returns:
        dict: {
            "answer": str,
            "sources": [
                {
                    "document_name": str,
                    "page": int,
                    "section": str,
                    "content": str
                }
            ],
            "success": bool,
            "is_mock": bool
        }
    """
    clean_q = question.strip() if question else ""
    if not clean_q:
        return {
            "answer": "Please provide a non-empty question to query the documentation.",
            "sources": [],
            "success": False,
            "is_mock": use_mock
        }

    # 1. LIVE BACKEND MODE - Strict Execution
    if not use_mock:
        status_info = check_backend_status()
        if not status_info["available"]:
            return {
                "answer": f"Live Backend Unavailable: {status_info['error_message']}. Please verify your teammates' backend implementation.",
                "sources": [],
                "success": False,
                "is_mock": False
            }

        real_rag = status_info["module"]
        try:
            result = real_rag.answer_question(clean_q)
            if not isinstance(result, dict):
                return {
                    "answer": "Live backend returned an invalid response format (expected dict).",
                    "sources": [],
                    "success": False,
                    "is_mock": False
                }
            result["is_mock"] = False
            return result
        except Exception as e:
            return {
                "answer": f"Live backend execution error while generating answer: {str(e)}",
                "sources": [],
                "success": False,
                "is_mock": False
            }

    # 2. MOCK MODE - Simulated developer response
    time.sleep(0.6)  # Simulated retrieval latency

    if not _MOCK_DOCUMENTS:
        return {
            "answer": "Cannot generate simulated answer: No documents exist in the knowledge base. Please upload a PDF documentation file first.",
            "sources": [],
            "success": False,
            "is_mock": True
        }

    doc_name = _MOCK_DOCUMENTS[0].get("filename", "documentation.pdf")
    q_lower = clean_q.lower()

    if any(k in q_lower for k in ["fastapi", "api", "route", "endpoint"]):
        answer = (
            "FastAPI routes are declared using decorator methods such as `@app.get()` and `@app.post()`.\n\n"
            "```python\n"
            "from fastapi import FastAPI\n\n"
            "app = FastAPI(title=\"DevDocs AI Service\")\n\n"
            "@app.get(\"/api/v1/health\")\n"
            "async def health_check():\n"
            "    return {\"status\": \"ok\", \"mode\": \"active\"}\n"
            "```\n\n"
            "FastAPI automatically provides interactive OpenAPI documentation at `/docs`."
        )
        sources = [
            {
                "document_name": doc_name,
                "page": 7,
                "section": "Routing & Path Operations",
                "content": "Path operation functions can be declared as async def to run directly on the asyncio event loop."
            },
            {
                "document_name": doc_name,
                "page": 12,
                "section": "OpenAPI Specification",
                "content": "Swagger UI interactive documentation is served automatically at the /docs path."
            }
        ]
    elif any(k in q_lower for k in ["python", "type", "syntax", "variable"]):
        answer = (
            "Modern Python utilizes type hinting and structural pattern matching:\n\n"
            "```python\n"
            "def process_query(query: str) -> dict[str, str]:\n"
            "    match query.split():\n"
            "        case [\"search\", term]:\n"
            "            return {\"action\": \"search\", \"term\": term}\n"
            "        case _:\n"
            "            return {\"action\": \"default\"}\n"
            "```"
        )
        sources = [
            {
                "document_name": doc_name,
                "page": 15,
                "section": "Type Hints & Pattern Matching",
                "content": "Type annotations allow static type checkers such as Mypy to catch type mismatches ahead of runtime."
            }
        ]
    elif any(k in q_lower for k in ["chunk", "embedding", "rag", "vector"]):
        answer = (
            "The RAG ingestion pipeline processes technical documentation in three distinct phases:\n\n"
            "1. **Text Extraction**: Extracts clean text while preserving markdown and headings.\n"
            "2. **Recursive Splitting**: Splits text into chunks of ~500 tokens with 50-token overlap.\n"
            "3. **Vector Embeddings**: Encodes chunks into high-dimensional vectors stored in a vector index."
        )
        sources = [
            {
                "document_name": doc_name,
                "page": 3,
                "section": "RAG Pipeline Architecture",
                "content": "Recursive character text splitting maintains semantic boundaries and context between adjacent chunks."
            }
        ]
    else:
        answer = (
            f"Based on the indexed document **{doc_name}**, here is the retrieved guidance:\n\n"
            f"> *\"{clean_q}\"*\n\n"
            "The documentation recommends modular architectures, comprehensive error handling, and decoupled service layers."
        )
        sources = [
            {
                "document_name": doc_name,
                "page": 19,
                "section": "Architecture & Best Practices",
                "content": f"Core conventions and design patterns relevant to '{clean_q}' are defined in the architectural specifications."
            }
        ]

    return {
        "answer": answer,
        "sources": sources,
        "success": True,
        "is_mock": True
    }
