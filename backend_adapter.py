"""
backend_adapter.py - Integration Layer between DevDocs AI Frontend and Backend

This module serves as the contract and bridge between the Streamlit frontend
and the RAG backend (developed by Team Members 2 and 3).

Frontend Integration Contracts:
1. ingest_document(file_path: str, use_mock: bool = True) -> dict
2. list_documents(owner: Optional[str] = None, limit: int = 100, use_mock: bool = False) -> list
3. answer_question(query: Optional[str] = None, filters: Optional[dict] = None, top_k: int = 5, question: Optional[str] = None, use_mock: bool = False) -> dict
4. ingest_pdf(file_bytes: bytes, document_name: str, owner: str = 'default_owner', document_id: Optional[str] = None) -> dict

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
        if "backend.rag_pipeline" in sys.modules:
            module = sys.modules["backend.rag_pipeline"]
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
_MOCK_DOCUMENTS: List[Dict[str, Any]] = []


def get_sample_demo_document() -> Dict[str, Any]:
    """Provides a sample simulated document for optional demo testing."""
    return {
        "document_id": "demo_doc_001",
        "filename": "fastapi_quickstart_guide.pdf (Demo)",
        "document_name": "fastapi_quickstart_guide.pdf (Demo)",
        "status": "simulated_indexed",
        "total_pages": 42,
        "total_chunks": 168,
        "chunk_count": 168,
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
                "document_name": filename,
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
                    "document_name": filename,
                    "status": "failed",
                    "message": "Live backend returned invalid format (expected dict).",
                    "is_mock": False
                }
            result["is_mock"] = False
            result.setdefault("filename", filename)
            result.setdefault("document_name", filename)
            return result
        except Exception as e:
            return {
                "success": False,
                "document_id": "",
                "filename": filename,
                "document_name": filename,
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
                "document_name": doc["filename"],
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
        "document_name": display_name,
        "file_path": file_path,
        "file_hash": file_hash,
        "status": "simulated_indexed",
        "total_pages": estimated_pages,
        "total_chunks": chunks,
        "chunk_count": chunks,
        "is_mock": True
    }
    _MOCK_DOCUMENTS.append(new_doc)

    return {
        "success": True,
        "document_id": doc_id,
        "filename": display_name,
        "document_name": display_name,
        "status": "simulated_indexed",
        "message": f"Document '{display_name}' processed in Mock Mode ({estimated_pages} pages, {chunks} chunks).",
        "is_mock": True
    }


def list_documents(
    owner: Optional[str] = None,
    limit: int = 100,
    use_mock: bool = False
) -> List[Dict[str, Any]]:
    """
    Retrieve all documents indexed in the knowledge base.

    Parameters:
        owner (Optional[str]): Filter documents by owner.
        limit (int): Maximum number of documents to retrieve.
        use_mock (bool): If True, returns mock store. If False, calls live backend.

    Returns:
        list of dict: [
            {
                "document_id": "doc_001",
                "filename": "python_guide.pdf",
                "document_name": "python_guide.pdf",
                "status": "indexed",
                "total_pages": 25,
                "total_chunks": 100,
                "chunk_count": 100,
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
            import inspect
            sig = inspect.signature(real_rag.list_documents)
            kwargs = {}
            if "owner" in sig.parameters:
                kwargs["owner"] = owner
            if "limit" in sig.parameters:
                kwargs["limit"] = limit
            docs = real_rag.list_documents(**kwargs)
            if not isinstance(docs, list):
                raise BackendExecutionError("Live backend list_documents() did not return a list.")
            for doc in docs:
                if isinstance(doc, dict):
                    doc["is_mock"] = False
                    # Bridge keys for both frontend and backend contracts
                    doc_name = doc.get("document_name") or doc.get("filename") or "unknown.pdf"
                    doc["filename"] = doc_name
                    doc["document_name"] = doc_name
                    c_count = doc.get("chunk_count") if doc.get("chunk_count") is not None else doc.get("total_chunks", 0)
                    doc["chunk_count"] = c_count
                    doc["total_chunks"] = c_count
                    if "total_pages" not in doc:
                        doc["total_pages"] = doc.get("page_count", "N/A")
                    if "status" not in doc:
                        doc["status"] = "indexed"
            return docs
        except BackendError:
            raise
        except Exception as e:
            raise BackendExecutionError(f"Live backend error in list_documents: {str(e)}") from e

    # 2. MOCK MODE - Return copy of in-memory simulated documents
    mock_docs = []
    for d in _MOCK_DOCUMENTS:
        doc_copy = dict(d)
        if owner and doc_copy.get("owner") and doc_copy.get("owner") != owner:
            continue
        mock_docs.append(doc_copy)
    if limit:
        mock_docs = mock_docs[:limit]
    return mock_docs


def answer_question(
    query: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
    question: Optional[str] = None,
    use_mock: bool = False
) -> Dict[str, Any]:
    """
    Query the RAG pipeline with a developer question and obtain an answer with citations.

    Parameters:
        query (Optional[str]): The natural language query from the developer.
        filters (Optional[Dict[str, Any]]): Metadata filters (e.g. {"document_id": "doc_1"}).
        top_k (int): Number of relevant chunks to retrieve.
        question (Optional[str]): Alias for query parameter.
        use_mock (bool): If True, returns simulated answer. If False, calls live backend.

    Returns:
        dict: {
            "answer": str,
            "citations": list,
            "sources": list,
            "status": "answered" | "not_found" | "conflict" | "error",
            "success": bool,
            "is_mock": bool
        }
    """
    raw_prompt = (query if query is not None else question) or ""
    clean_q = raw_prompt.strip()
    if not clean_q:
        return {
            "answer": "Please provide a valid question.",
            "citations": [],
            "sources": [],
            "status": "error",
            "success": False,
            "is_mock": use_mock
        }

    # 1. LIVE BACKEND MODE - Strict Execution
    if not use_mock:
        status_info = check_backend_status()
        if not status_info["available"]:
            return {
                "answer": f"Live Backend Unavailable: {status_info['error_message']}. Please verify your teammates' backend implementation.",
                "citations": [],
                "sources": [],
                "status": "error",
                "success": False,
                "is_mock": False
            }

        real_rag = status_info["module"]
        try:
            import inspect
            sig = inspect.signature(real_rag.answer_question)
            kwargs = {}
            if "filters" in sig.parameters:
                kwargs["filters"] = filters
            if "top_k" in sig.parameters:
                kwargs["top_k"] = top_k
            result = real_rag.answer_question(clean_q, **kwargs)
            if not isinstance(result, dict):
                return {
                    "answer": "Live backend returned an invalid response format (expected dict).",
                    "citations": [],
                    "sources": [],
                    "status": "error",
                    "success": False,
                    "is_mock": False
                }
            result["is_mock"] = False
            status_val = result.get("status", "answered")
            result["status"] = status_val
            result["success"] = (status_val in ["answered", "not_found", "conflict"]) and bool(result.get("answer"))

            # Normalize citations / sources so both interfaces are satisfied
            raw_cits = result.get("citations") or result.get("sources") or []
            normalized_cits = []
            for idx, c in enumerate(raw_cits):
                if isinstance(c, dict):
                    c_norm = dict(c)
                    c_norm["source_id"] = c_norm.get("source_id") or f"src_{idx + 1}"
                    c_norm["document_id"] = c_norm.get("document_id") or "doc_unknown"
                    c_norm["document_name"] = c_norm.get("document_name") or "Unknown Document"
                    page_num = c_norm.get("page_number") if c_norm.get("page_number") is not None else c_norm.get("page", 1)
                    c_norm["page_number"] = page_num
                    c_norm["page"] = page_num
                    c_norm["section"] = c_norm.get("section", "Excerpt")
                    txt = c_norm.get("text") or c_norm.get("content") or ""
                    c_norm["text"] = txt
                    c_norm["content"] = txt
                    normalized_cits.append(c_norm)

            result["citations"] = normalized_cits
            result["sources"] = normalized_cits
            return result
        except Exception as e:
            return {
                "answer": f"Live backend execution error while generating answer: {str(e)}",
                "citations": [],
                "sources": [],
                "status": "error",
                "success": False,
                "is_mock": False
            }

    # 2. MOCK MODE - Simulated developer response
    time.sleep(0.6)  # Simulated retrieval latency

    if not _MOCK_DOCUMENTS:
        return {
            "answer": "Cannot generate simulated answer: No documents exist in the knowledge base. Please upload a PDF documentation file first.",
            "citations": [],
            "sources": [],
            "status": "not_found",
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
                "page_number": 7,
                "section": "Routing & Path Operations",
                "content": "Path operation functions can be declared as async def to run directly on the asyncio event loop.",
                "text": "Path operation functions can be declared as async def to run directly on the asyncio event loop.",
                "source_id": "c1",
                "document_id": "mock_doc_001"
            },
            {
                "document_name": doc_name,
                "page": 12,
                "page_number": 12,
                "section": "OpenAPI Specification",
                "content": "Swagger UI interactive documentation is served automatically at the /docs path.",
                "text": "Swagger UI interactive documentation is served automatically at the /docs path.",
                "source_id": "c2",
                "document_id": "mock_doc_001"
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
                "page_number": 15,
                "section": "Type Hints & Pattern Matching",
                "content": "Type annotations allow static type checkers such as Mypy to catch type mismatches ahead of runtime.",
                "text": "Type annotations allow static type checkers such as Mypy to catch type mismatches ahead of runtime.",
                "source_id": "c1",
                "document_id": "mock_doc_001"
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
                "page_number": 3,
                "section": "RAG Pipeline Architecture",
                "content": "Recursive character text splitting maintains semantic boundaries and context between adjacent chunks.",
                "text": "Recursive character text splitting maintains semantic boundaries and context between adjacent chunks.",
                "source_id": "c1",
                "document_id": "mock_doc_001"
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
                "page_number": 19,
                "section": "Architecture & Best Practices",
                "content": f"Core conventions and design patterns relevant to '{clean_q}' are defined in the architectural specifications.",
                "text": f"Core conventions and design patterns relevant to '{clean_q}' are defined in the architectural specifications.",
                "source_id": "c1",
                "document_id": "mock_doc_001"
            }
        ]

    return {
        "answer": answer,
        "sources": sources,
        "citations": sources,
        "status": "answered",
        "success": True,
        "is_mock": True
    }


# -----------------------------------------------------------------------------
# 5. CONVENIENCE ALIASES & COMPATIBILITY BRIDGES
# -----------------------------------------------------------------------------
def ingest_pdf(
    file_bytes: Any,
    document_name: str,
    owner: str = "default_owner",
    document_id: Optional[str] = None
) -> Dict[str, Any]:
    """Compatibility bridge for raw bytes ingestion into live backend."""
    from backend.rag_pipeline import ingest_pdf as _ingest_pdf
    return _ingest_pdf(
        file_bytes=file_bytes,
        document_name=document_name,
        owner=owner,
        document_id=document_id
    )

ask_question = answer_question
query_documents = answer_question
upload_pdf = ingest_pdf
get_documents = list_documents
check_health = is_backend_available


# -----------------------------------------------------------------------------
# 6. RETRIEVAL EVALUATION & DATASET MANAGEMENT
# -----------------------------------------------------------------------------
def run_evaluation(k: int = 5, use_mock: bool = False) -> Dict[str, Any]:
    """
    Executes standard Information Retrieval benchmark metrics:
    Hit Rate@K, Precision@K, Recall@K (Demonstrating MongoDB Retrieval Evaluation Badge).
    """
    if not use_mock:
        try:
            from backend.evaluation import evaluate_retrieval
            metrics = evaluate_retrieval(k=k)
            metrics["is_mock"] = False
            return metrics
        except Exception as e:
            return {
                "error": str(e),
                "hit_rate": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "k": k,
                "total_queries": 0,
                "query_details": [],
                "is_mock": False
            }

    return {
        "hit_rate": 1.0,
        "precision": 0.80,
        "recall": 1.0,
        "k": k,
        "total_queries": 3,
        "query_details": [
            {"query": "How does Python handle exceptions?", "hit_rate": 1.0, "precision": 1.0, "recall": 1.0},
            {"query": "What is a Python decorator?", "hit_rate": 1.0, "precision": 0.8, "recall": 1.0},
            {"query": "How do generator functions and yield work?", "hit_rate": 1.0, "precision": 0.6, "recall": 1.0}
        ],
        "is_mock": True
    }


def load_curated_sample_documents(use_mock: bool = False) -> List[Dict[str, Any]]:
    """
    Loads the curated developer documentation dataset (FastAPI v1, Security v2, Database Spec)
    into either the mock storage or live MongoDB vector database.
    """
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    sample_files = [
        "FastAPI_Service_Guide_v1.pdf",
        "FastAPI_Security_v2.pdf",
        "Database_Cluster_Architecture.pdf"
    ]
    results = []
    for s_name in sample_files:
        f_path = os.path.join(data_dir, s_name)
        if not os.path.exists(f_path):
            try:
                from backend.sample_data_generator import generate_curated_dataset
                generate_curated_dataset()
            except Exception:
                pass
        if os.path.exists(f_path):
            res = ingest_document(f_path, use_mock=use_mock)
            results.append(res)
    return results
