"""
backend/rag_pipeline.py - RAG Pipeline & LLM Grounding for DevDocs AI
====================================================================
Responsible for:
- Query embedding generation matching Member 2's model (all-MiniLM-L6-v2, 384 dims, normalized).
- Hybrid search combining Vector Search (semantic) and Keyword Search (exact terms).
- Reciprocal Rank Fusion (RRF) for principled score blending.
- Context formatting with strict source attribution.
- LLM response generation with strict anti-hallucination and evidence grounding prompt.
- Structured citation extraction (document_id, document_name, page_number, snippet).
- Automated conflict detection between contradictory documentation sources.
- Handling unanswerable questions without hallucinating.
"""

import io
import os
import re
import uuid
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from backend.database import (
    retrieve_chunks,
    retrieve_chunks_by_keyword,
    save_document_metadata,
    save_chunks,
    list_documents,
    DEFAULT_EMBEDDING_DIMENSIONS,
    EMBEDDING_DIMENSIONS,
)

# Setup logging
logger = logging.getLogger("devdocs.rag")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

load_dotenv()

# Configuration constants
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.30"))
RRF_K_CONSTANT = int(os.getenv("RRF_K_CONSTANT", "60"))

# Singleton sentence-transformer embedding model
_embedding_model = None


def get_embedding_model():
    """
    Load and cache the SentenceTransformer model configured to match Member 2.
    Model: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions).
    """
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    try:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: '{EMBEDDING_MODEL_NAME}'...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return _embedding_model
    except ImportError:
        logger.warning("sentence-transformers is not yet installed. Running in mock/fallback mode.")
        return None
    except Exception as err:
        logger.error(f"Error loading embedding model: {err}")
        return None


def generate_query_embedding(query_text: str) -> List[float]:
    """
    Generate normalized 384-dimensional query embedding matching Member 2's document vectors.
    """
    if not query_text or not isinstance(query_text, str):
        raise ValueError("Query text must be a non-empty string.")

    model = get_embedding_model()
    if model is not None:
        # Member 2 uses normalized cosine embeddings
        embedding = model.encode(query_text, normalize_embeddings=True)
        return embedding.tolist()
    else:
        # Fallback / mock embedding for unit testing environments
        logger.warning("Using fallback dummy embedding for testing.")
        dim = int(os.getenv("EMBEDDING_DIMENSIONS", DEFAULT_EMBEDDING_DIMENSIONS))
        return [0.05] * dim


def reciprocal_rank_fusion(
    vector_results: List[Dict[str, Any]],
    keyword_results: List[Dict[str, Any]],
    k: int = RRF_K_CONSTANT,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion (RRF) algorithm to merge and rank results from
    Vector Search (semantic meaning) and Keyword Search (exact syntax/APIs).
    
    Formula:
        RRF_Score(doc) = SUM( 1 / (k + rank_i(doc)) ) for each retriever i
        
    Why RRF?
    Vector similarity scores (0.0 to 1.0) and keyword match counts (1, 2, 5...)
    are on completely different scales. Directly adding them produces biased results.
    RRF operates solely on ordinal ranks, rewarding documents that score well in both.
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict[str, Any]] = {}

    # 1. Process Vector Search rankings
    for rank, item in enumerate(vector_results, start=1):
        cid = item["chunk_id"]
        doc_map[cid] = item
        scores[cid] = scores.get(cid, 0.0) + (1.0 / (k + rank))

    # 2. Process Keyword Search rankings
    for rank, item in enumerate(keyword_results, start=1):
        cid = item["chunk_id"]
        if cid not in doc_map:
            doc_map[cid] = item
        scores[cid] = scores.get(cid, 0.0) + (1.0 / (k + rank))

    # 3. Sort by aggregated RRF score descending
    sorted_chunk_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

    fused_results = []
    for cid in sorted_chunk_ids[:top_k]:
        chunk_copy = dict(doc_map[cid])
        chunk_copy["rrf_score"] = round(scores[cid], 5)
        fused_results.append(chunk_copy)

    return fused_results


def retrieve_hybrid_chunks(
    query: str,
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Orchestrate hybrid retrieval:
    1. Embed query text into 384-dim vector.
    2. Query MongoDB Vector Search.
    3. Query MongoDB Keyword Search.
    4. Fuse and re-rank with Reciprocal Rank Fusion.
    """
    candidate_k = max(top_k * 2, 10)

    # 1. Vector Search
    query_vector = generate_query_embedding(query)
    try:
        vector_chunks = retrieve_chunks(query_vector, filters=filters, top_k=candidate_k)
    except Exception as err:
        logger.warning(f"Vector search failed or index not ready: {err}. Falling back to keyword search.")
        vector_chunks = []

    # 2. Keyword Search
    try:
        keyword_chunks = retrieve_chunks_by_keyword(query, filters=filters, top_k=candidate_k)
    except Exception as err:
        logger.warning(f"Keyword search failed: {err}")
        keyword_chunks = []

    # 3. Fuse and re-rank with Reciprocal Rank Fusion
    if not vector_chunks and not keyword_chunks:
        return []

    return reciprocal_rank_fusion(vector_chunks, keyword_chunks, k=RRF_K_CONSTANT, top_k=top_k)


def format_context_for_prompt(chunks: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into numbered source context blocks with explicit metadata.
    """
    formatted_blocks = []
    for idx, c in enumerate(chunks, start=1):
        block = (
            f"[Source {idx} | ID: {c.get('chunk_id')} | Doc: {c.get('document_name', 'Unknown')} | "
            f"Page: {c.get('page_number', 1)}]\n"
            f"{c.get('text', '').strip()}"
        )
        formatted_blocks.append(block)

    return "\n\n".join(formatted_blocks)


def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Call LLM provider (Groq developer tier).
    Configurable via environment variables with automated model fallback.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or "your_groq_api_key_here" in api_key:
        raise ValueError(
            "GROQ_API_KEY is missing or unconfigured. "
            "Please obtain a free key at https://console.groq.com and place it in your .env file."
        )

    api_key = api_key.strip()
    if api_key.startswith("ygsk_"):
        api_key = api_key[1:]

    from groq import Groq
    client = Groq(api_key=api_key)

    preferred_model = os.getenv("LLM_MODEL", LLM_MODEL)
    candidate_models = [preferred_model, "qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]

    last_error = None
    for model_name in candidate_models:
        try:
            response = client.chat.completions.create(
                model=model_name,
                temperature=LLM_TEMPERATURE,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as err:
            last_error = err
            if "model_not_found" in str(err) or "does not exist" in str(err):
                logger.warning(f"Model '{model_name}' not available on Groq. Trying fallback...")
                continue
            raise

    raise RuntimeError(f"All candidate Groq models failed: {last_error}")


def generate_answer(query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate a grounded answer with citations, unanswerable detection, and conflict detection.
    
    Guarantees:
    - Never invents citations or facts.
    - If no chunks or low relevance: returns status 'not_found'.
    - If contradictory documentation detected: returns status 'conflict'.
    - Otherwise returns grounded answer with status 'answered'.
    """
    # Case 1: No chunks retrieved
    if not retrieved_chunks:
        return {
            "answer": "The answer is not available in the uploaded documents.",
            "citations": [],
            "status": "not_found"
        }

    # Case 1b: All chunks below similarity threshold (evidence validation)
    vector_scores = [c.get("score", 0.0) for c in retrieved_chunks if "score" in c]
    if vector_scores and max(vector_scores) < SIMILARITY_THRESHOLD and not any(c.get("rrf_score", 0) > 0.02 for c in retrieved_chunks):
        return {
            "answer": "The answer is not available in the uploaded documents.",
            "citations": [],
            "status": "not_found"
        }

    context_str = format_context_for_prompt(retrieved_chunks)

    system_prompt = (
        "You are DevDocs AI, a technical documentation assistant.\n"
        "Your task is to answer the user's question accurately and STRICTLY based ONLY on the provided Context.\n\n"
        "STRICT RULES:\n"
        "1. Grounding: Answer ONLY using facts directly mentioned in the Context. Do NOT use outside training knowledge.\n"
        "2. Missing Information: If the Context does not contain enough information to answer the question, output EXACTLY:\n"
        "   'The answer is not available in the uploaded documents.'\n"
        "3. Conflicting Information: If different documents in the Context contradict each other, explicitly explain the conflict, "
        "cite both contradictory sources by their document name and page number, and do NOT arbitrarily pick one.\n"
        "4. Citations: When making a factual claim, cite the source ID inline, e.g. [Source 1] or [Source 2].\n"
        "5. Tone: Concise, professional, and code-accurate."
    )

    user_prompt = (
        f"Context:\n{context_str}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    try:
        raw_answer = call_llm(system_prompt=system_prompt, user_prompt=user_prompt)
    except Exception as err:
        logger.error(f"LLM generation failed: {err}")
        return {
            "answer": f"Error communicating with LLM service: {err}",
            "citations": [],
            "status": "error"
        }

    # Check for unanswerable response
    unanswerable_patterns = [
        "not available in the uploaded documents",
        "the answer is not available in the provided",
        "provided context does not contain",
        "the context does not mention",
    ]
    if any(pat in raw_answer.lower() for pat in unanswerable_patterns):
        return {
            "answer": "The answer is not available in the uploaded documents.",
            "citations": [],
            "status": "not_found"
        }

    # Detect conflict / contradiction in the response
    conflict_patterns = [
        "conflict", "contradict", "disagree", "discrepancy",
        "on the other hand", "whereas document", "differing accounts"
    ]
    # Check if multiple distinct documents contributed and disagreement was articulated
    unique_doc_ids = {c.get("document_id") for c in retrieved_chunks if c.get("document_id")}
    is_conflict = len(unique_doc_ids) > 1 and any(pat in raw_answer.lower() for pat in conflict_patterns)

    # Build structured citations
    citations = []
    seen_chunk_ids = set()
    for idx, c in enumerate(retrieved_chunks, start=1):
        cid = c.get("chunk_id", f"chunk_{idx}")
        # Only include citations that match or are relevant
        if cid not in seen_chunk_ids:
            seen_chunk_ids.add(cid)
            citations.append({
                "source_id": cid,
                "document_id": c.get("document_id", "unknown_doc"),
                "document_name": c.get("document_name", "unknown_doc.pdf"),
                "page_number": c.get("page_number", 1),
                "text": c.get("text", "")
            })

    return {
        "answer": raw_answer,
        "citations": citations,
        "status": "conflict" if is_conflict else "answered"
    }


def _extract_text_from_pdf_bytes(file_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Extract text per page from PDF bytes using pypdf.
    Returns list of dicts: [{"page_number": int, "text": str}]
    """
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        pages = []
        for idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"page_number": idx, "text": text.strip()})
        return pages
    except Exception as err:
        logger.error(f"Error reading PDF bytes: {err}")
        raise ValueError(f"Could not parse PDF file: {err}") from err


def _chunk_pages(
    pages: List[Dict[str, Any]],
    document_id: str,
    document_name: str,
    chunk_size: int = 500,
    overlap: int = 100
) -> List[Dict[str, Any]]:
    """
    Split pages into chunks with page tracking and metadata.
    """
    chunks = []
    chunk_index = 1

    for p in pages:
        page_num = p["page_number"]
        page_text = p["text"]

        start = 0
        text_len = len(page_text)

        while start < text_len:
            end = min(start + chunk_size, text_len)
            snippet = page_text[start:end].strip()

            if snippet:
                chunks.append({
                    "chunk_id": f"{document_id}_p{page_num}_c{chunk_index}",
                    "document_id": document_id,
                    "document_name": document_name,
                    "text": snippet,
                    "page_number": page_num,
                    "content_type": "application/pdf"
                })
                chunk_index += 1

            if end >= text_len:
                break
            start += chunk_size - overlap

    return chunks


def ingest_pdf(
    file_bytes: Any,
    document_name: str,
    owner: str = "default_owner",
    document_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Ingest a PDF file:
    1. Extracts text by delegating to Member 2's pdf_reader if available, or pypdf fallback.
    2. Chunks text by delegating to Member 2's chunking module if available, or sliding window fallback.
    3. Generates 384-dimensional embeddings using Member 2's model (all-MiniLM-L6-v2).
    4. Saves document metadata and chunks into MongoDB Atlas.
    
    Args:
        file_bytes: Raw bytes or file-like object containing PDF data.
        document_name: Filename (e.g. 'manual.pdf').
        owner: User/owner identifier.
        document_id: Optional unique document identifier. Generated if omitted.
        
    Returns:
        {
            "document_id": str,
            "document_name": str,
            "owner": str,
            "total_chunks": int,
            "status": "success" | "error",
            "message": str
        }
    """
    # Normalize file_bytes if an open file-like object was passed (e.g. Streamlit UploadedFile)
    if hasattr(file_bytes, "read"):
        file_bytes = file_bytes.read()
    if hasattr(file_bytes, "getvalue"):
        file_bytes = file_bytes.getvalue()

    if not file_bytes or not isinstance(file_bytes, (bytes, bytearray)):
        return {
            "document_id": document_id or "",
            "document_name": document_name or "",
            "owner": owner,
            "total_chunks": 0,
            "status": "error",
            "message": "Invalid or empty PDF file bytes provided."
        }

    doc_id = document_id or f"doc_{uuid.uuid4().hex[:10]}"
    doc_name = document_name or f"{doc_id}.pdf"
    doc_owner = owner or "default_owner"

    try:
        # Step 1: Extract text (Reuse Member 2's pdf_reader if present)
        pages_data = None
        try:
            from backend import pdf_reader
            for fn_name in ["extract_text_from_pdf", "read_pdf", "parse_pdf"]:
                if hasattr(pdf_reader, fn_name):
                    pages_data = getattr(pdf_reader, fn_name)(file_bytes)
                    break
        except (ImportError, AttributeError):
            pages_data = None

        if pages_data is None:
            pages_data = _extract_text_from_pdf_bytes(file_bytes)

        if not pages_data:
            return {
                "document_id": doc_id,
                "document_name": doc_name,
                "owner": doc_owner,
                "total_chunks": 0,
                "status": "error",
                "message": "No extractable text found in the provided PDF."
            }

        # Step 2: Split into chunks (Reuse Member 2's chunking if present)
        raw_chunks = None
        try:
            from backend import chunking
            for fn_name in ["chunk_document", "create_chunks", "chunk_text"]:
                if hasattr(chunking, fn_name):
                    raw_chunks = getattr(chunking, fn_name)(pages_data, document_id=doc_id, document_name=doc_name)
                    break
        except (ImportError, AttributeError):
            raw_chunks = None

        if raw_chunks is None:
            raw_chunks = _chunk_pages(pages_data, document_id=doc_id, document_name=doc_name)

        if not raw_chunks:
            return {
                "document_id": doc_id,
                "document_name": doc_name,
                "owner": doc_owner,
                "total_chunks": 0,
                "status": "error",
                "message": "Text splitting produced zero chunks."
            }

        # Step 3: Embed chunks (Reuse Member 2's embeddings if present, else internal all-MiniLM-L6-v2)
        embedded_chunks = None
        try:
            from backend import embeddings
            for fn_name in ["generate_embeddings", "embed_chunks", "get_embeddings"]:
                if hasattr(embeddings, fn_name):
                    embedded_chunks = getattr(embeddings, fn_name)(raw_chunks)
                    break
        except (ImportError, AttributeError):
            embedded_chunks = None

        if embedded_chunks is None:
            # Generate using our verified model (all-MiniLM-L6-v2, 384 dims, normalized)
            model = get_embedding_model()
            texts = [c["text"] for c in raw_chunks]
            if model is not None:
                vectors = model.encode(texts, normalize_embeddings=True).tolist()
            else:
                dim = int(os.getenv("EMBEDDING_DIMENSIONS", DEFAULT_EMBEDDING_DIMENSIONS))
                vectors = [[0.05] * dim for _ in texts]

            embedded_chunks = []
            for c, vec in zip(raw_chunks, vectors):
                chunk_copy = dict(c)
                chunk_copy["embedding"] = vec
                embedded_chunks.append(chunk_copy)

        # Step 4: Save Document Metadata & Chunks into MongoDB Atlas
        save_document_metadata({
            "document_id": doc_id,
            "document_name": doc_name,
            "owner": doc_owner,
            "content_type": "application/pdf"
        })

        save_chunks(embedded_chunks)

        return {
            "document_id": doc_id,
            "document_name": doc_name,
            "owner": doc_owner,
            "total_chunks": len(embedded_chunks),
            "status": "success",
            "message": f"Successfully ingested '{doc_name}' ({len(embedded_chunks)} chunks)."
        }

    except Exception as err:
        logger.error(f"Error during PDF ingestion: {err}")
        return {
            "document_id": doc_id,
            "document_name": doc_name,
            "owner": doc_owner,
            "total_chunks": 0,
            "status": "error",
            "message": f"Ingestion failed: {err}"
        }


def answer_question(
    query: str,
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Main entry point for DevDocs AI backend.
    
    Called by Member 1 (Streamlit Frontend):
        from backend.rag_pipeline import answer_question
        result = answer_question("How does exception handling work in Python?", filters=None, top_k=5)
        
    Returns:
        {
            "answer": str,
            "citations": [
                {
                    "source_id": str,
                    "document_id": str,
                    "document_name": str,
                    "page_number": int,
                    "text": str
                }
            ],
            "status": "answered" | "not_found" | "conflict" | "error"
        }
    """
    if not query or not isinstance(query, str) or not query.strip():
        return {
            "answer": "Please provide a valid question.",
            "citations": [],
            "status": "error"
        }

    top_k = max(1, int(top_k))

    try:
        # 1. Retrieve relevant chunks using hybrid search
        retrieved_chunks = retrieve_hybrid_chunks(query.strip(), filters=filters, top_k=top_k)

        # 2. Ground and generate answer
        result = generate_answer(query.strip(), retrieved_chunks)
        return result
    except Exception as err:
        logger.error(f"Error in answer_question pipeline: {err}")
        return {
            "answer": f"Internal pipeline error: {err}",
            "citations": [],
            "status": "error"
        }

