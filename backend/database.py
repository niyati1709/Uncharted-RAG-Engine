"""
backend/database.py - MongoDB Atlas Database & Vector Search Module for DevDocs AI
==================================================================================
Responsible for:
- MongoDB connection management (with connection pooling and timeout handling).
- Database schema validation and indexing for 'documents' and 'chunks' collections.
- Idempotent document metadata insertion and bulk chunk storage.
- Embedding dimension validation (coordinated with Member 2).
- Vector search retrieval using MongoDB Atlas $vectorSearch.
"""

import os
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne, IndexModel, ASCENDING
from pymongo.errors import PyMongoError, ConnectionFailure, ConfigurationError, OperationFailure

# Setup logging
logger = logging.getLogger("devdocs.database")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Load environment variables from .env
load_dotenv()

# Configuration constants with safe defaults
MONGODB_URI: Optional[str] = os.getenv("MONGODB_URI")
DATABASE_NAME: str = os.getenv("MONGODB_DATABASE", "devdocs_ai")
DOCUMENTS_COLLECTION: str = os.getenv("MONGODB_DOCUMENTS_COLLECTION", "documents")
CHUNKS_COLLECTION: str = os.getenv("MONGODB_CHUNKS_COLLECTION", "chunks")
VECTOR_INDEX_NAME: str = os.getenv("MONGODB_VECTOR_INDEX_NAME", "chunks_vector_index")

# Embedding configuration (Coordinated with Member 2: all-MiniLM-L6-v2)
DEFAULT_EMBEDDING_DIMENSIONS: int = 384
EMBEDDING_DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", DEFAULT_EMBEDDING_DIMENSIONS))

# Cached MongoDB client (Singleton pattern to avoid opening repeated connections)
_mongo_client: Optional[MongoClient] = None


def get_client() -> MongoClient:
    """
    Establish or return the existing cached MongoClient instance.
    Configured with a 5-second server selection timeout to prevent application hanging.
    """
    global _mongo_client
    if _mongo_client is not None:
        return _mongo_client

    uri = os.getenv("MONGODB_URI")
    if not uri or "<password>" in uri or "<username>" in uri:
        raise ValueError(
            "MONGODB_URI environment variable is missing or contains unreplaced placeholders. "
            "Please configure your valid Atlas connection string in .env"
        )

    try:
        _mongo_client = MongoClient(
            uri,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            socketTimeoutMS=10000,
            appname="DevDocsAI"
        )
        return _mongo_client
    except (ConfigurationError, PyMongoError) as err:
        logger.error(f"Failed to initialize MongoClient: {err}")
        raise


def get_database(db_name: Optional[str] = None):
    """
    Return the MongoDB database instance.
    """
    client = get_client()
    return client[db_name or DATABASE_NAME]


def close_database() -> None:
    """
    Cleanly close the MongoDB connection and reset the cached client.
    """
    global _mongo_client
    if _mongo_client is not None:
        try:
            _mongo_client.close()
            logger.info("MongoDB connection closed cleanly.")
        except Exception as err:
            logger.warning(f"Error closing MongoDB connection: {err}")
        finally:
            _mongo_client = None


def ping_connection() -> Tuple[bool, str]:
    """
    Verify network connectivity and authentication against the MongoDB Atlas cluster.
    Returns:
        Tuple of (is_connected: bool, message: str)
    """
    try:
        client = get_client()
        # Admin ping command forces a round-trip network check
        client.admin.command("ping")
        return True, f"Successfully connected to MongoDB cluster (Database: '{DATABASE_NAME}')."
    except ConnectionFailure as e:
        return False, f"Connection failed. Please check network access/IP whitelist on Atlas: {e}"
    except ConfigurationError as e:
        return False, f"Configuration error. Please check your connection string in .env: {e}"
    except Exception as e:
        return False, f"Unexpected error while pinging MongoDB: {e}"


def init_indexes() -> Dict[str, List[str]]:
    """
    Create necessary indexes on documents and chunks collections for fast querying and data integrity.
    - documents: unique index on document_id, index on upload_date, index on owner.
    - chunks: unique index on chunk_id, index on document_id, index on content_type.
    """
    db = get_database()
    docs_col = db[DOCUMENTS_COLLECTION]
    chunks_col = db[CHUNKS_COLLECTION]

    created = {"documents": [], "chunks": []}

    try:
        # Document collection indexes
        doc_indexes = [
            IndexModel([("document_id", ASCENDING)], unique=True, name="idx_document_id_unique"),
            IndexModel([("owner", ASCENDING)], name="idx_document_owner"),
            IndexModel([("upload_date", ASCENDING)], name="idx_document_upload_date"),
        ]
        doc_res = docs_col.create_indexes(doc_indexes)
        created["documents"] = doc_res
        logger.info(f"Created indexes for '{DOCUMENTS_COLLECTION}': {doc_res}")

        # Chunks collection indexes
        chunk_indexes = [
            IndexModel([("chunk_id", ASCENDING)], unique=True, name="idx_chunk_id_unique"),
            IndexModel([("document_id", ASCENDING)], name="idx_chunk_document_id"),
            IndexModel([("content_type", ASCENDING)], name="idx_chunk_content_type"),
        ]
        chunk_res = chunks_col.create_indexes(chunk_indexes)
        created["chunks"] = chunk_res
        logger.info(f"Created indexes for '{CHUNKS_COLLECTION}': {chunk_res}")

        return created
    except PyMongoError as err:
        logger.error(f"Error initializing database indexes: {err}")
        raise


def save_document_metadata(document: Dict[str, Any]) -> Dict[str, Any]:
    """
    Store or update document metadata in the 'documents' collection.
    
    Required fields in document:
    - document_id (str)
    - document_name (str)
    
    Optional fields:
    - upload_date (str/datetime)
    - owner (str)
    - content_type (str)
    """
    if not isinstance(document, dict):
        raise TypeError("Document metadata must be a dictionary.")

    doc_id = document.get("document_id")
    doc_name = document.get("document_name")

    if not doc_id or not isinstance(doc_id, str):
        raise ValueError("Missing or invalid 'document_id' in document metadata.")
    if not doc_name or not isinstance(doc_name, str):
        raise ValueError("Missing or invalid 'document_name' in document metadata.")

    # Populate defaults
    payload = {
        "document_id": doc_id,
        "document_name": doc_name,
        "upload_date": document.get("upload_date") or datetime.now(timezone.utc).isoformat(),
        "owner": document.get("owner", "default_owner"),
        "content_type": document.get("content_type", "application/pdf"),
    }
    # Preserve any additional custom metadata
    for k, v in document.items():
        if k not in payload:
            payload[k] = v

    db = get_database()
    docs_col = db[DOCUMENTS_COLLECTION]

    try:
        docs_col.replace_one(
            {"document_id": doc_id},
            payload,
            upsert=True
        )
        logger.info(f"Saved document metadata for '{doc_id}' ({doc_name}).")
        return payload
    except PyMongoError as err:
        logger.error(f"Failed to save document metadata for '{doc_id}': {err}")
        raise


def save_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate and save chunks produced by Member 2 into the 'chunks' collection.
    Uses bulk upsert so re-uploading an existing document updates its chunks idempotently.
    
    Validates:
    - chunks is a non-empty list of dicts.
    - Each chunk must contain 'chunk_id', 'document_id', 'text', 'embedding'.
    - 'embedding' must be a list of floats matching EMBEDDING_DIMENSIONS (384).
    """
    if not isinstance(chunks, list) or len(chunks) == 0:
        raise ValueError("Chunks must be a non-empty list of chunk dictionaries.")

    operations = []
    expected_dim = int(os.getenv("EMBEDDING_DIMENSIONS", DEFAULT_EMBEDDING_DIMENSIONS))

    for idx, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise TypeError(f"Chunk at index {idx} is not a dictionary.")

        chunk_id = chunk.get("chunk_id")
        doc_id = chunk.get("document_id")
        text = chunk.get("text")
        embedding = chunk.get("embedding")

        if not chunk_id or not isinstance(chunk_id, str):
            raise ValueError(f"Chunk at index {idx} has missing or invalid 'chunk_id'.")
        if not doc_id or not isinstance(doc_id, str):
            raise ValueError(f"Chunk at index {idx} has missing or invalid 'document_id'.")
        if text is None or not isinstance(text, str):
            raise ValueError(f"Chunk '{chunk_id}' has missing or non-string 'text'.")

        # Validate embedding
        if not isinstance(embedding, list) or len(embedding) == 0:
            raise ValueError(
                f"Chunk '{chunk_id}' is missing a valid embedding list. "
                "Embeddings must be generated before saving chunks."
            )
        if len(embedding) != expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch for chunk '{chunk_id}'. "
                f"Expected {expected_dim} dimensions (from {os.getenv('EMBEDDING_MODEL_NAME', 'all-MiniLM-L6-v2')}), "
                f"but received {len(embedding)} dimensions."
            )

        # Standardized chunk document
        normalized_chunk = {
            "chunk_id": chunk_id,
            "document_id": doc_id,
            "document_name": chunk.get("document_name", ""),
            "text": text,
            "page_number": chunk.get("page_number", 1),
            "embedding": [float(x) for x in embedding],
            "content_type": chunk.get("content_type", "application/pdf"),
        }

        # Preserve any additional metadata provided by Member 2
        for k, v in chunk.items():
            if k not in normalized_chunk:
                normalized_chunk[k] = v

        operations.append(
            UpdateOne(
                {"chunk_id": chunk_id},
                {"$set": normalized_chunk},
                upsert=True
            )
        )

    db = get_database()
    chunks_col = db[CHUNKS_COLLECTION]

    try:
        result = chunks_col.bulk_write(operations, ordered=False)
        total_affected = result.upserted_count + result.modified_count + result.matched_count
        logger.info(
            f"Successfully processed {len(chunks)} chunks: "
            f"{result.upserted_count} inserted, {result.modified_count} updated."
        )
        return {
            "status": "success",
            "total_chunks": len(chunks),
            "upserted_count": result.upserted_count,
            "modified_count": result.modified_count,
        }
    except PyMongoError as err:
        logger.error(f"Bulk write failed while saving chunks: {err}")
        raise


def retrieve_chunks(
    query_embedding: List[float],
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Semantic Vector Search using MongoDB Atlas $vectorSearch aggregation pipeline.
    
    Args:
        query_embedding: List of 384 floating-point values generated from the query text.
        filters: Optional dictionary for metadata filtering (e.g., {"document_id": "doc_001"}).
        top_k: Maximum number of nearest neighbors to return (default: 5).
        
    Returns:
        List of chunks with source metadata, text, and vector similarity score.
    """
    expected_dim = int(os.getenv("EMBEDDING_DIMENSIONS", DEFAULT_EMBEDDING_DIMENSIONS))
    if not isinstance(query_embedding, list) or len(query_embedding) != expected_dim:
        raise ValueError(
            f"Invalid query_embedding: must be a list of {expected_dim} float values, "
            f"got {len(query_embedding) if isinstance(query_embedding, list) else type(query_embedding)}."
        )

    top_k = max(1, int(top_k))
    vector_index = os.getenv("MONGODB_VECTOR_INDEX_NAME", VECTOR_INDEX_NAME)

    # Construct Atlas $vectorSearch definition
    vector_search_stage: Dict[str, Any] = {
        "index": vector_index,
        "path": "embedding",
        "queryVector": [float(x) for x in query_embedding],
        "numCandidates": max(top_k * 10, 50),
        "limit": top_k,
    }

    # Add filter if provided
    if filters and isinstance(filters, dict):
        # Format filters for Atlas Vector Search
        filter_clause = {}
        for k, v in filters.items():
            filter_clause[k] = {"$eq": v}
        vector_search_stage["filter"] = filter_clause

    pipeline = [
        {"$vectorSearch": vector_search_stage},
        {
            "$project": {
                "_id": 0,
                "chunk_id": 1,
                "document_id": 1,
                "document_name": 1,
                "text": 1,
                "page_number": 1,
                "content_type": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]

    db = get_database()
    chunks_col = db[CHUNKS_COLLECTION]

    try:
        cursor = chunks_col.aggregate(pipeline)
        results = list(cursor)
        return results
    except OperationFailure as err:
        logger.error(f"Vector search execution error: {err}")
        # Informative error message for hackathon troubleshooting
        if "Index not found" in str(err) or "vectorSearch" in str(err):
            raise RuntimeError(
                f"Atlas Vector Search failed. Ensure that the vector index '{vector_index}' "
                f"is created on collection '{CHUNKS_COLLECTION}' and has finished building on MongoDB Atlas. "
                f"Underlying error: {err}"
            ) from err
        raise
    except PyMongoError as err:
        logger.error(f"MongoDB error during vector retrieval: {err}")
        raise


def retrieve_chunks_by_keyword(
    query_text: str,
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Keyword Search for exact technical terms, function names, error codes, and identifiers.
    Searches against the 'text' field using regex / text matching.
    
    Args:
        query_text: The user's query string.
        filters: Optional metadata filters.
        top_k: Maximum number of keyword results to return.
        
    Returns:
        List of chunks with metadata and basic keyword occurrence score.
    """
    if not query_text or not isinstance(query_text, str):
        return []

    top_k = max(1, int(top_k))
    db = get_database()
    chunks_col = db[CHUNKS_COLLECTION]

    # Clean query terms into individual keywords (skip short filler words)
    keywords = [k.strip() for k in query_text.split() if len(k.strip()) > 2]
    if not keywords:
        keywords = [query_text.strip()]

    # Construct regex search matching any of the technical keywords (case-insensitive)
    regex_pattern = "|".join([f"\\b{k}\\b" for k in keywords])

    query_filter: Dict[str, Any] = {
        "text": {"$regex": regex_pattern, "$options": "i"}
    }

    # Apply additional metadata filters if passed
    if filters and isinstance(filters, dict):
        for k, v in filters.items():
            query_filter[k] = v

    try:
        cursor = chunks_col.find(
            query_filter,
            {
                "_id": 0,
                "chunk_id": 1,
                "document_id": 1,
                "document_name": 1,
                "text": 1,
                "page_number": 1,
                "content_type": 1,
            }
        )
        if hasattr(cursor, "limit"):
            cursor = cursor.limit(top_k * 2)

        results = []
        for doc in cursor:
            # Calculate match count as keyword relevance score
            text_lower = doc.get("text", "").lower()
            matches = sum(text_lower.count(k.lower()) for k in keywords)
            doc["score"] = float(matches)
            results.append(doc)

        # Sort descending by keyword occurrence count
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    except PyMongoError as err:
        logger.error(f"Keyword search failed: {err}")
        return []


def get_document(document_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a single document's metadata by its document_id."""
    db = get_database()
    return db[DOCUMENTS_COLLECTION].find_one({"document_id": document_id}, {"_id": 0})


def delete_document(document_id: str) -> Dict[str, int]:
    """Delete a document and all of its associated chunks."""
    db = get_database()
    doc_res = db[DOCUMENTS_COLLECTION].delete_one({"document_id": document_id})
    chunk_res = db[CHUNKS_COLLECTION].delete_many({"document_id": document_id})
    return {
        "deleted_documents": doc_res.deleted_count,
        "deleted_chunks": chunk_res.deleted_count
    }


def list_documents(owner: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """
    List uploaded documents from the 'documents' collection.
    
    Args:
        owner: Optional owner username to filter by.
        limit: Maximum number of documents to return (default: 100).
        
    Returns:
        List of document dictionaries sorted by upload_date descending.
        Each dictionary contains:
        - document_id (str)
        - document_name (str)
        - upload_date (str)
        - owner (str)
        - content_type (str)
        - chunk_count (int)
    """
    db = get_database()
    docs_col = db[DOCUMENTS_COLLECTION]
    chunks_col = db[CHUNKS_COLLECTION]

    query_filter: Dict[str, Any] = {}
    if owner and isinstance(owner, str) and owner.strip():
        query_filter["owner"] = owner.strip()

    limit = max(1, min(int(limit), 1000))

    try:
        cursor = docs_col.find(query_filter, {"_id": 0})
        if hasattr(cursor, "sort") and not isinstance(cursor, list):
            cursor = cursor.sort("upload_date", -1)
        if hasattr(cursor, "limit") and not isinstance(cursor, list):
            cursor = cursor.limit(limit)

        documents = list(cursor)
        if isinstance(cursor, list):
            documents.sort(key=lambda d: d.get("upload_date", ""), reverse=True)
            documents = documents[:limit]

        # Attach chunk count for each document
        for doc in documents:
            doc_id = doc.get("document_id")
            if doc_id:
                try:
                    count = chunks_col.count_documents({"document_id": doc_id})
                    doc["chunk_count"] = count
                except Exception:
                    doc["chunk_count"] = 0
            else:
                doc["chunk_count"] = 0

        return documents

    except PyMongoError as err:
        logger.error(f"Failed to list documents: {err}")
        return []
