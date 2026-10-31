"""
backend/sample_data_generator.py - Generates realistic developer documentation PDFs
for DevDocs AI Hackathon Demonstrations.
"""

import os
from typing import List

def create_pdf(pages_text: List[str], output_path: str):
    """
    Creates a valid standard PDF 1.4 file with clean formatting and text streams
    without requiring external heavyweight PDF generation libraries.
    """
    objects = []

    def add_obj(content: str) -> int:
        objects.append(content)
        return len(objects)

    # 1: Catalog
    add_obj("<< /Type /Catalog /Pages 2 0 R >>")
    # 2: Pages placeholder (will be replaced)
    add_obj("")
    # 3: Standard Helvetica Font
    add_obj("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    # 4: Bold Font
    add_obj("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")

    page_refs = []
    for page_content in pages_text:
        stream_cmds = ["BT", "50 740 Td", "14 TL"]
        lines = page_content.strip().split("\n")
        
        for line in lines:
            trimmed = line.strip()
            if not trimmed:
                stream_cmds.append("14 TL")
                continue
            
            # Select font based on heading or body
            if trimmed.startswith("# "):
                stream_cmds.append("/F2 16 Tf")
                text = trimmed[2:].replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
                stream_cmds.append(f"({text}) '")
                stream_cmds.append("/F1 10 Tf")
            elif trimmed.startswith("## "):
                stream_cmds.append("/F2 12 Tf")
                text = trimmed[3:].replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
                stream_cmds.append(f"({text}) '")
                stream_cmds.append("/F1 10 Tf")
            else:
                stream_cmds.append("/F1 10 Tf")
                text = trimmed.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
                stream_cmds.append(f"({text}) '")

        stream_cmds.append("ET")
        stream_body = "\n".join(stream_cmds)
        stream_bytes = stream_body.encode("latin-1", errors="replace")
        
        stream_obj = f"<< /Length {len(stream_bytes)} >>\nstream\n{stream_body}\nendstream"
        s_id = add_obj(stream_obj)

        p_obj = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {s_id} 0 R "
            f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> >>"
        )
        p_id = add_obj(p_obj)
        page_refs.append(f"{p_id} 0 R")

    # Update 2: Pages catalog
    kids_str = " ".join(page_refs)
    objects[1] = f"<< /Type /Pages /Kids [ {kids_str} ] /Count {len(page_refs)} >>"

    # Assemble complete PDF binary
    pdf_bytes = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects):
        offsets.append(len(pdf_bytes))
        pdf_bytes.extend(f"{i+1} 0 obj\n{obj}\nendobj\n".encode("latin-1"))

    xref_offset = len(pdf_bytes)
    pdf_bytes.extend(b"xref\n")
    pdf_bytes.extend(f"0 {len(objects)+1}\n".encode("latin-1"))
    pdf_bytes.extend(b"0000000000 65535 f \n")
    for off in offsets:
        pdf_bytes.extend(f"{off:010d} 00000 n \n".encode("latin-1"))

    pdf_bytes.extend(
        f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("latin-1")
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(pdf_bytes)
    return output_path


def generate_curated_dataset():
    """Generates the 3 curated developer documentation PDF files."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

    # Document 1: FastAPI Service Guide v1.2
    doc1_pages = [
        """# FastAPI Microservice Specification - Version 1.2
## Section 1: Server Configuration and Architecture
DevDocs AI microservices architecture relies on FastAPI running on top of Uvicorn ASGI server.
The core authentication and routing service binds by default to HTTP port 8000.
The primary service health check endpoint is exposed at GET /api/v1/health returning status ok.
Worker concurrency is governed by Uvicorn process workers configured via WEB_CONCURRENCY environment variable.

Authentication in Version 1.2 is handled through standard HTTP Bearer Tokens.
Clients must obtain a token via POST /api/v1/auth/login with valid client credentials in JSON format.
Tokens are signed with HMAC-SHA256 and carry a validity TTL of 3600 seconds (1 hour).
All internal service communications use unencrypted HTTP over the local Docker bridge network.""",
        """# FastAPI Microservice Specification - Version 1.2
## Section 2: Rate Limiting and Resilience Policy
In Version 1.2, rate limiting is applied at the API gateway layer per client IP address.
The standard throughput threshold is fixed at 100 requests per minute per remote IP address.
When a client exceeds this limit, the gateway immediately returns HTTP 429 Too Many Requests.
A Retry-After header with a fixed delay of 60 seconds is sent with every 429 response.

For persistent data storage, the service connects to MongoDB Atlas using standard connection strings.
Connection pool size is capped at 50 connections with a server selection timeout of 5000 milliseconds.
Application logs are emitted in JSON format to stdout for log collection by fluentd."""
    ]
    path1 = os.path.join(data_dir, "FastAPI_Service_Guide_v1.pdf")
    create_pdf(doc1_pages, path1)

    # Document 2: FastAPI Security v2.0 (Contains explicit architectural evolutions & conflicts with v1)
    doc2_pages = [
        """# FastAPI Cloud Security and Migration Spec - Version 2.0
## Section 1: Port Allocation and TLS Requirements
This specification mandates urgent breaking security upgrades from Version 1.2 to Version 2.0.
All microservices must terminate plain HTTP traffic. The service is relocated to secure HTTPS port 8443.
Notice: Port 8000 is officially deprecated and forbidden in production environments.
Strict Transport Security (HSTS) with TLS 1.3 is enforced; incoming connections on port 8000 are rejected.

The authentication protocol has transitioned to OAuth2 with PKCE (Proof Key for Code Exchange).
The legacy endpoint POST /api/v1/auth/login is completely decommissioned and will return 410 Gone.
Clients must now authenticate exclusively against POST /api/v2/auth/token using asymmetric RSA keys.
Access token lifetime has been tightened to 900 seconds (15 minutes) with mandatory refresh tokens.""",
        """# FastAPI Cloud Security and Migration Spec - Version 2.0
## Section 2: Zero-Trust Gateway and Redis Caching
Version 2.0 replaces IP-based throttling with token bucket rate limiting backed by distributed Redis clusters.
The standard throughput threshold is increased to 500 requests per minute for authenticated developer accounts.
Unauthenticated probes are limited to 20 requests per minute before experiencing an automatic 15-minute ban.

All database queries must route through mutual TLS (mTLS) with client certificate validation.
Sensitive request payloads containing developer access tokens must be masked in application logs.
Monitoring alerts are dispatched to PagerDuty if error rates exceed 1 percent over a 5-minute rolling window."""
    ]
    path2 = os.path.join(data_dir, "FastAPI_Security_v2.pdf")
    create_pdf(doc2_pages, path2)

    # Document 3: Database & Vector Index Architecture
    doc3_pages = [
        """# DevDocs AI - Database and Vector Index Architecture Spec
## Section 1: MongoDB Atlas Multi-Region Cluster
The DevDocs AI storage engine is hosted on MongoDB Atlas M10 dedicated cluster across three availability zones.
Documentation metadata is organized into collection 'documents' and text chunks are stored in 'chunks'.
Each document record tracks document_id, document_name, upload_date, owner, and content_type.
Each chunk record contains chunk_id, document_id, page_number, text, and vector embedding.

Vector search is executed natively in MongoDB Atlas using the specialized $vectorSearch aggregation operator.
The vector index is designated as 'chunks_vector_index' built on the 'embedding' field.
The embedding model is sentence-transformers/all-MiniLM-L6-v2 outputting exactly 384 dimensions.
The index metric type is cosine similarity with 150 candidates evaluated per vector search query.""",
        """# DevDocs AI - Database and Vector Index Architecture Spec
## Section 2: Hybrid Search and Benchmark Evaluation
To capture both broad semantic intent and exact code symbols, DevDocs AI implements Hybrid Search.
Semantic vector retrieval is fused with lexical BM25/regex matching using Reciprocal Rank Fusion (RRF).
The RRF constant k is calibrated to 60, allowing balanced combination of dense vectors and exact keywords.

Retrieval performance is continuously measured against ground-truth technical queries.
Evaluation metrics include Hit Rate@5, Precision@5, and Recall@5.
Target benchmark standards require a Hit Rate@5 greater than 0.85 across benchmark queries.
Cloud snapshot backups are automatically executed every 24 hours with a rolling 7-day retention policy."""
    ]
    path3 = os.path.join(data_dir, "Database_Cluster_Architecture.pdf")
    create_pdf(doc3_pages, path3)

    return [path1, path2, path3]

if __name__ == "__main__":
    generated = generate_curated_dataset()
    print("Generated curated dataset:")
    for g in generated:
        print(" -", g)

