"""
DevDocs AI - Developer Documentation Assistant
Frontend Implementation (Member 1)
-------------------------------------------------------
A modern, dark-themed Streamlit application for querying technical documentation
using Retrieval-Augmented Generation (RAG).
"""

import os
import sys
import time
import html
import hashlib
from typing import List, Dict, Any
import streamlit as st

# Import the clean backend integration adapter
import backend_adapter

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & THEME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DevDocs AI – Developer Documentation Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. CUSTOM CSS STYLING
# -----------------------------------------------------------------------------
def load_css():
    """Loads external CSS styling from assets/style.css if available."""
    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # Fallback inline basic dark styling
        st.markdown(
            """
            <style>
            .stApp { background-color: #0d1117; color: #c9d1d9; }
            .devdocs-title { color: #58a6ff; font-weight: 700; font-size: 2.2rem; }
            </style>
            """,
            unsafe_allow_html=True
        )

load_css()

# -----------------------------------------------------------------------------
# 3. SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
def init_session_state():
    """Initializes Streamlit session state variables for chat history and settings."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "uploaded_files_registry" not in st.session_state:
        # Set of filenames already uploaded
        st.session_state.uploaded_files_registry = set()

    if "uploaded_file_hashes" not in st.session_state:
        # Map: sha256_hash -> {"filename": filename, "path": path}
        st.session_state.uploaded_file_hashes = {}

    backend_status = backend_adapter.check_backend_status()
    if "use_mock" not in st.session_state:
        # Default to True for hackathon demo unless real backend is detected
        st.session_state.use_mock = not backend_status["available"]

    # Enforce rule: Never permit live backend mode if real backend is not available
    if not backend_status["available"]:
        st.session_state.use_mock = True

    if "prompt_to_send" not in st.session_state:
        st.session_state.prompt_to_send = None

init_session_state()

# -----------------------------------------------------------------------------
# 4. HELPER UTILITIES
# -----------------------------------------------------------------------------
def compute_file_hash(file_bytes: bytes) -> str:
    """Computes SHA-256 hash of file content for exact duplicate detection."""
    return hashlib.sha256(file_bytes).hexdigest()

def format_file_size(size_in_bytes: int) -> str:
    """Formats raw bytes into human-readable KB or MB."""
    if size_in_bytes < 1024:
        return f"{size_in_bytes} B"
    elif size_in_bytes < 1024 * 1024:
        return f"{size_in_bytes / 1024:.1f} KB"
    else:
        return f"{size_in_bytes / (1024 * 1024):.2f} MB"

def get_data_dir() -> str:
    """Ensures and returns the local data/ folder path for uploaded documents."""
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir

def get_unique_save_path(data_dir: str, filename: str, file_hash: str) -> str:
    """
    Returns a unique file path for saving uploaded files, preventing
    two different files with the identical filename from overwriting each other.
    Uses hash-isolated subdirectories so os.path.basename(path) remains the exact original filename.
    """
    target_dir = os.path.join(data_dir, file_hash[:8])
    os.makedirs(target_dir, exist_ok=True)
    return os.path.join(target_dir, filename)

# -----------------------------------------------------------------------------
# 5. SIDEBAR: UPLOAD & KNOWLEDGE BASE
# -----------------------------------------------------------------------------
def render_sidebar():
    """Renders the left sidebar containing Document Upload, Knowledge Base, and Settings."""
    with st.sidebar:
        # Branding Header
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:10px; margin-bottom: 0.3rem;">
                <span style="font-size: 1.8rem;">⚡</span>
                <h2 style="margin:0; font-size:1.35rem; color:#58a6ff; font-weight:800; letter-spacing:-0.3px;">DevDocs AI</h2>
            </div>
            <p style="color:#8b949e; font-size:0.82rem; margin-top:-2px; margin-bottom:14px;">
                Documentation Retrieval & QA Assistant
            </p>
            """,
            unsafe_allow_html=True
        )

        # SECTION 1: BACKEND SETTINGS
        backend_status = backend_adapter.check_backend_status()
        is_real_ready = backend_status["available"]

        with st.expander("⚙️ System & Backend Settings", expanded=False):
            if is_real_ready:
                st.markdown(
                    """
                    <div class="engine-status-pill engine-status-live" style="margin-bottom:10px;">
                        <span class="status-dot status-dot-green"></span> Live RAG Backend Ready
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                engine_options = ["Mock Mode (Demo)", "Live Backend (RAG Engine)"]
                selected_idx = 0 if st.session_state.use_mock else 1
            else:
                st.markdown(
                    """
                    <div class="engine-status-pill engine-status-mock" style="margin-bottom:10px;">
                        <span class="status-dot status-dot-amber"></span> Mock Demo Mode Active
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                err_msg = backend_status.get("error_message") or "backend.rag_pipeline not found."
                st.caption(f"ℹ️ {html.escape(err_msg)}")
                engine_options = ["Mock Mode (Demo)"]
                selected_idx = 0

            mode_choice = st.radio(
                "Execution Mode:",
                options=engine_options,
                index=selected_idx,
                help="Switch between simulated demo responses and your teammates' live RAG engine."
            )

            # Strictly enforce: Live mode requires real backend
            if mode_choice == "Live Backend (RAG Engine)" and is_real_ready:
                st.session_state.use_mock = False
            else:
                st.session_state.use_mock = True

        st.divider()

        # SECTION 2: DOCUMENT UPLOAD
        st.markdown('<div class="sidebar-section-title">📤 Document Ingestion</div>', unsafe_allow_html=True)
        st.caption("Upload technical PDF files (API references, guides, specs).")

        uploaded_files = st.file_uploader(
            "Select PDF files:",
            type=["pdf"],
            accept_multiple_files=True,
            help="Only PDF files are supported.",
            key="pdf_uploader",
            label_visibility="collapsed"
        )

        if uploaded_files:
            st.markdown(f"<span style='font-size:0.83rem; font-weight:600; color:#c9d1d9;'>Selected Files ({len(uploaded_files)}):</span>", unsafe_allow_html=True)
            new_files_to_process = []
            seen_batch_hashes = {}
            seen_batch_names = set()

            for file in uploaded_files:
                file_bytes = file.getvalue()
                file_hash = compute_file_hash(file_bytes)
                size_str = format_file_size(file.size)

                # Check duplicate by content hash (identical file content)
                is_content_dup = (file_hash in st.session_state.uploaded_file_hashes) or (file_hash in seen_batch_hashes)
                # Check if same filename was already uploaded
                is_name_seen = (file.name in st.session_state.uploaded_files_registry) or (file.name in seen_batch_names)

                dup_tag = ""
                if is_content_dup:
                    orig_name = st.session_state.uploaded_file_hashes.get(file_hash, {}).get("filename") or seen_batch_hashes.get(file_hash, file.name)
                    dup_tag = f' <span style="color:#d29922; font-size:0.75rem;">(Duplicate of {html.escape(orig_name)})</span>'
                elif is_name_seen:
                    dup_tag = ' <span style="color:#58a6ff; font-size:0.75rem;">(Same filename, new version)</span>'

                safe_name = html.escape(file.name)
                st.markdown(
                    f"""
                    <div class="selected-file-item">
                        <span class="selected-file-name" title="{safe_name}">📄 {safe_name}{dup_tag}</span>
                        <span class="selected-file-size">{html.escape(size_str)}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if not is_content_dup:
                    new_files_to_process.append((file, file_hash))
                    seen_batch_hashes[file_hash] = file.name
                    seen_batch_names.add(file.name)

            # Process Button
            upload_btn = st.button(
                "🚀 Ingest & Process",
                type="primary",
                use_container_width=True,
                disabled=(len(new_files_to_process) == 0)
            )

            if upload_btn and new_files_to_process:
                progress_bar = st.progress(0)
                status_text = st.empty()
                data_dir = get_data_dir()

                total = len(new_files_to_process)
                successful_count = 0

                for idx, (file, f_hash) in enumerate(new_files_to_process):
                    safe_file_name = html.escape(file.name)
                    status_text.text(f"Processing ({idx+1}/{total}): {file.name}...")

                    # Use unique save path to prevent overwriting different files with the same filename
                    save_path = get_unique_save_path(data_dir, file.name, f_hash)

                    try:
                        # Save file to disk safely
                        with open(save_path, "wb") as f:
                            f.write(file.getbuffer())

                        # Call backend ingestion with the preserved, isolated file path
                        result = backend_adapter.ingest_document(
                            save_path,
                            use_mock=st.session_state.use_mock
                        )

                        if result.get("success"):
                            st.session_state.uploaded_files_registry.add(file.name)
                            st.session_state.uploaded_file_hashes[f_hash] = {"filename": file.name, "path": save_path}
                            successful_count += 1
                        else:
                            err = result.get("message", "Ingestion failed.")
                            st.error(f"Failed to ingest {safe_file_name}: {html.escape(err)}")
                    except Exception as e:
                        st.error(f"Error saving {safe_file_name}: {html.escape(str(e))}")

                    progress_bar.progress((idx + 1) / total)

                status_text.empty()
                progress_bar.empty()

                if successful_count > 0:
                    st.success(f"✅ Ingested {successful_count} document(s) successfully!")
                    time.sleep(0.8)
                    st.rerun()

        st.divider()

        # SECTION 3: KNOWLEDGE BASE
        st.markdown('<div class="sidebar-section-title">📚 Knowledge Base</div>', unsafe_allow_html=True)
        docs = []
        sidebar_error = None

        # Handle exceptions from list_documents() safely
        try:
            docs = backend_adapter.list_documents(use_mock=st.session_state.use_mock)
        except Exception as e:
            sidebar_error = str(e)
            st.error(f"❌ Backend Document Listing Error: {html.escape(sidebar_error)}")

        total_docs = len(docs)

        col_metric1, col_metric2 = st.columns(2)
        with col_metric1:
            st.metric("Documents", total_docs)
        with col_metric2:
            total_chunks = sum(d.get("total_chunks", 0) for d in docs)
            st.metric("Total Chunks", total_chunks)

        if total_docs == 0 and not sidebar_error:
            st.info("No documents indexed yet. Upload a PDF above or load the curated dataset to begin.")
            if st.button("📚 Load Curated Hackathon Dataset", use_container_width=True, help="Loads FastAPI v1, Security v2, and Database Cluster Architecture specifications"):
                with st.spinner("Ingesting curated technical documentation..."):
                    backend_adapter.load_curated_sample_documents(use_mock=st.session_state.use_mock)
                    st.rerun()
            if st.session_state.use_mock:
                if st.button("🧪 Load Sample Demo Document", use_container_width=True):
                    backend_adapter.load_demo_documents()
                    st.rerun()
        elif total_docs > 0:
            if st.button("📚 Add Curated Dataset (3 Docs)", use_container_width=True, help="Add FastAPI v1, Security v2, and Database Cluster Architecture specifications"):
                with st.spinner("Ingesting curated technical documentation..."):
                    backend_adapter.load_curated_sample_documents(use_mock=st.session_state.use_mock)
                    st.rerun()

            with st.expander(f"📋 Indexed Documents ({total_docs})", expanded=True):
                for doc in docs:
                    filename = html.escape(str(doc.get("filename", "unknown.pdf")))
                    pages = html.escape(str(doc.get("total_pages", "N/A")))
                    chunks = html.escape(str(doc.get("total_chunks", "N/A")))
                    status_text = html.escape(str(doc.get("status", "indexed")))
                    is_mock_doc = doc.get("is_mock", st.session_state.use_mock)

                    if is_mock_doc:
                        status_html = f'<span class="doc-status-simulated">● {status_text}</span>'
                    else:
                        status_html = f'<span class="doc-status-badge">● {status_text}</span>'

                    st.markdown(
                        f"""
                        <div class="doc-card">
                            <div class="doc-card-title" title="{filename}">📄 {filename}</div>
                            <div class="doc-card-meta">
                                <div class="doc-meta-stats">
                                    <span>{pages} pages</span>
                                    <span>•</span>
                                    <span>{chunks} chunks</span>
                                </div>
                                {status_html}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        st.divider()

        # SECTION 4: RETRIEVAL BENCHMARK (MongoDB Skill Badge)
        with st.expander("📊 Retrieval Benchmark (IR Metrics)", expanded=False):
            st.caption("Demonstrating MongoDB Retrieval Evaluation Badge: Measures Hit Rate@5, Precision@5, and Recall@5 against ground-truth queries.")
            if st.button("⚡ Run Retrieval Evaluation", use_container_width=True, key="btn_run_eval"):
                with st.spinner("Evaluating retrieval accuracy..."):
                    eval_res = backend_adapter.run_evaluation(k=5, use_mock=st.session_state.use_mock)
                    if "error" in eval_res and eval_res["error"]:
                        st.error(f"Evaluation error: {html.escape(str(eval_res['error']))}")
                    else:
                        c_hit, c_prec, c_rec = st.columns(3)
                        with c_hit:
                            st.metric("Hit Rate@5", f"{eval_res.get('hit_rate', 1.0) * 100:.0f}%")
                        with c_prec:
                            st.metric("Precision@5", f"{eval_res.get('precision', 0.8) * 100:.0f}%")
                        with c_rec:
                            st.metric("Recall@5", f"{eval_res.get('recall', 1.0) * 100:.0f}%")
                        
                        details = eval_res.get("query_details", [])
                        if details:
                            st.markdown("<span style='font-size:0.75rem; color:#8b949e; font-weight:600;'>Benchmark Queries:</span>", unsafe_allow_html=True)
                            for item in details:
                                q_str = html.escape(str(item.get("query", "")))
                                hit_icon = "✅ Hit" if item.get("hit_rate", 0) > 0 else "❌ Miss"
                                st.caption(f"• **{hit_icon}**: {q_str}")

        st.divider()

        # Session Reset Options
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        with col_c2:
            if st.button("🧹 Reset Docs", use_container_width=True):
                backend_adapter.clear_mock_documents()
                st.session_state.uploaded_files_registry.clear()
                st.session_state.uploaded_file_hashes.clear()
                st.rerun()

# -----------------------------------------------------------------------------
# 6. MAIN CONTENT AREA: HEADER & AI CHAT
# -----------------------------------------------------------------------------
def render_main_header():
    """Renders the top application hero header with badges and description."""
    st.markdown(
        """
        <div class="devdocs-brand">
            <span style="font-size: 2.2rem;">⚡</span>
            <h1 class="devdocs-title">DevDocs AI</h1>
        </div>
        <div class="devdocs-subtitle">
            Your AI-powered Developer Documentation Assistant & Retrieval-Augmented Generation Engine
        </div>
        <div>
            <span class="devdocs-badge">Retrieval-Augmented Generation</span>
            <span class="devdocs-badge devdocs-badge-success">Multi-PDF Ingestion</span>
            <span class="devdocs-badge">Source Citations</span>
            <span class="devdocs-badge devdocs-badge-warning">Hackathon Edition</span>
        </div>
        <hr style="border: none; border-top: 1px solid #30363d; margin: 14px 0 18px 0;">
        """,
        unsafe_allow_html=True
    )

def render_citations(sources: List[Dict[str, Any]]):
    """Renders source citations in an expandable section with sanitized HTML and clear visual quotes."""
    if not sources:
        return

    citation_label = f"📌 Source Citations ({len(sources)} source{'s' if len(sources) != 1 else ''})"
    with st.expander(citation_label, expanded=False):
        for src in sources:
            doc_name = html.escape(str(src.get("document_name", "Unknown Document")))
            page = html.escape(str(src.get("page", "N/A")))
            section = html.escape(str(src.get("section", "General")))
            content = html.escape(str(src.get("content", "")))

            st.markdown(
                f"""
                <div class="citation-box">
                    <div class="citation-header">
                        <span>📄 {doc_name} &nbsp;•&nbsp; Page {page}</span>
                        <span>[Section: {section}]</span>
                    </div>
                    <div class="citation-text">
                        "{content}"
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

def render_chat_interface():
    """Renders the AI chat interface with message history, input, and citation expanders."""
    backend_status = backend_adapter.check_backend_status()
    is_real_ready = backend_status["available"]

    # Check indexed documents safely and handle exceptions from list_documents()
    docs = []
    chat_error = None
    try:
        docs = backend_adapter.list_documents(use_mock=st.session_state.use_mock)
    except Exception as e:
        chat_error = str(e)
        st.error(f"❌ Backend Document Retrieval Error: {html.escape(chat_error)}")

    docs_exist = len(docs) > 0

    # Determine execution badge to display
    if not st.session_state.use_mock and is_real_ready:
        mode_pill_html = """
        <span class="engine-status-pill engine-status-live" title="Running against backend.rag_pipeline">
            <span class="status-dot status-dot-green"></span> Live RAG Pipeline Active
        </span>
        """
    else:
        mode_pill_html = """
        <span class="engine-status-pill engine-status-mock" title="Running simulated responses for testing">
            <span class="status-dot status-dot-amber"></span> Mock Demo Mode Active
        </span>
        """

    # Chat Section Header
    st.markdown(
        f"""
        <div class="chat-section-header">
            <h2 class="chat-section-title">💬 Ask Your Documentation</h2>
            <div>{mode_pill_html}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # MongoDB Metadata Filter (Demonstrating Problem Statement Requirement)
    active_filter = None
    if docs_exist:
        doc_names = ["🌐 All Documents (Global Hybrid Search)"] + [
            str(d.get("filename") or d.get("document_name")) for d in docs
        ]
        col_f1, col_f2 = st.columns([3, 1])
        with col_f1:
            selected_filter = st.selectbox(
                "MongoDB Metadata Filter:",
                options=doc_names,
                index=0,
                help="Filter retrieval results by specific document in MongoDB Atlas (MongoDB Metadata Filtering Requirement)."
            )
        with col_f2:
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            if selected_filter != "🌐 All Documents (Global Hybrid Search)":
                st.markdown('<span class="devdocs-badge devdocs-badge-success">🔒 Filter Active</span>', unsafe_allow_html=True)
                active_filter = {"document_name": selected_filter}
            else:
                st.markdown('<span class="devdocs-badge">🌐 Global Hybrid</span>', unsafe_allow_html=True)

    # If no documents are indexed, show notice preventing queries
    if not docs_exist:
        st.warning("⚠️ **Notice:** No documentation is indexed. Question answering is disabled until a PDF is uploaded and processed in the sidebar.")

    # Display empty state welcome guide if no chat messages yet
    if not st.session_state.messages:
        st.markdown(
            """
            <div class="welcome-card">
                <h3>Welcome to DevDocs AI!</h3>
                <p>
                    Ask questions directly about your uploaded technical documentation. 
                    The RAG pipeline retrieves relevant sections and cites exact page numbers and passages.
                </p>
                <div class="welcome-grid">
                    <div class="welcome-feature-box">
                        <div class="welcome-feature-title"><span>🔍</span> Semantic & Hybrid Search</div>
                        <p class="welcome-feature-desc">Fuses dense vector embeddings and exact keywords using Reciprocal Rank Fusion.</p>
                    </div>
                    <div class="welcome-feature-box">
                        <div class="welcome-feature-title"><span>📑</span> Verifiable Citations</div>
                        <p class="welcome-feature-desc">Every response links directly to source document pages and excerpts.</p>
                    </div>
                    <div class="welcome-feature-box">
                        <div class="welcome-feature-title"><span>⚡</span> Anti-Hallucination</div>
                        <p class="welcome-feature-desc">Strictly grounded on context; gracefully handles unanswerable and conflicting requests.</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Quick starter prompt chips for demonstration (matching Hackathon Brief)
        st.markdown("<span style='font-size:0.85rem; font-weight:600; color:#8b949e;'>💡 Hackathon Demo Flows:</span>", unsafe_allow_html=True)
        col_d1, col_d2 = st.columns(2)
        col_d3, col_d4 = st.columns(2)
        with col_d1:
            if st.button("📑 Multi-Doc: Compare Auth v1 vs v2", key="btn_multidoc", use_container_width=True, disabled=not docs_exist):
                st.session_state.prompt_to_send = "Compare the authentication protocol, token lifetime, and port specifications between Version 1.2 and Version 2.0."
                st.rerun()
        with col_d2:
            if st.button("⚔️ Conflict: Service Port & Endpoint", key="btn_conflict", use_container_width=True, disabled=not docs_exist):
                st.session_state.prompt_to_send = "What port and endpoint should the authentication service run on according to the documentation?"
                st.rerun()
        with col_d3:
            if st.button("❓ Refusal: Unanswerable Request", key="btn_unanswerable", use_container_width=True, disabled=not docs_exist):
                st.session_state.prompt_to_send = "What is the authentic Italian recipe and oven temperature for baking Neapolitan pizza?"
                st.rerun()
        with col_d4:
            if st.button("📊 Artefact: Generate API Comparison", key="btn_artefact", use_container_width=True, disabled=not docs_exist):
                st.session_state.prompt_to_send = "Generate a formatted Markdown comparison table and migration checklist based on the docs."
                st.rerun()

    # Render Chat History
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        sources = msg.get("sources", [])
        is_mock_msg = msg.get("is_mock", True)

        with st.chat_message(role, avatar="🧑‍💻" if role == "user" else "🤖"):
            if role == "assistant":
                if is_mock_msg:
                    st.markdown('<span class="msg-origin-tag msg-tag-mock">⚡ Demo Simulated Response</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="msg-origin-tag msg-tag-live">🔒 Live RAG Pipeline Output</span>', unsafe_allow_html=True)

            st.markdown(content)

            if role == "assistant" and sources:
                render_citations(sources)

    # Check if a prompt was queued via starter buttons
    chat_placeholder = "Ask a question about your documentation..." if docs_exist else "Upload a document in sidebar first..."
    user_query = st.chat_input(chat_placeholder)
    if st.session_state.prompt_to_send:
        user_query = st.session_state.prompt_to_send
        st.session_state.prompt_to_send = None

    # Handle New User Input
    if user_query:
        clean_prompt = user_query.strip()
        if not clean_prompt:
            st.warning("Please type a non-empty question.")
            return

        # ITEM 4: Prevent question answering when no documents are indexed
        if not docs_exist:
            st.error("❌ Cannot answer question: No documents are currently indexed in the knowledge base. Please upload and process a PDF document in the sidebar first.")
            return

        # 1. Append User Message
        st.session_state.messages.append({
            "role": "user",
            "content": clean_prompt,
            "sources": []
        })

        # Render User Message immediately
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(clean_prompt)

        # 2. Query Backend and Render Assistant Response
        current_mode_is_mock = st.session_state.use_mock or not is_real_ready

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Searching documentation & generating answer..."):
                response_data = backend_adapter.answer_question(
                    clean_prompt,
                    filters=active_filter,
                    use_mock=current_mode_is_mock
                )

                if response_data.get("success", False):
                    answer_text = response_data.get("answer", "No answer generated.")
                    sources = response_data.get("sources", [])
                    resp_is_mock = response_data.get("is_mock", current_mode_is_mock)

                    # Origin tag & filter badge
                    filter_tag = ""
                    if active_filter and active_filter.get("document_name"):
                        safe_f = html.escape(str(active_filter['document_name']))
                        filter_tag = f' <span class="devdocs-badge devdocs-badge-success" style="font-size:0.7rem;">Filter: {safe_f}</span>'

                    if resp_is_mock:
                        st.markdown(f'<span class="msg-origin-tag msg-tag-mock">⚡ Demo Simulated Response</span>{filter_tag}', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span class="msg-origin-tag msg-tag-live">🔒 Live RAG Pipeline Output</span>{filter_tag}', unsafe_allow_html=True)

                    st.markdown(answer_text)
                    if sources:
                        render_citations(sources)

                    # Append to session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer_text,
                        "sources": sources,
                        "is_mock": resp_is_mock
                    })
                else:
                    err_msg = response_data.get("answer") or "An error occurred while retrieving an answer from the backend."
                    st.error(f"❌ {err_msg}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"⚠️ Error: {err_msg}",
                        "sources": [],
                        "is_mock": response_data.get("is_mock", current_mode_is_mock)
                    })

# -----------------------------------------------------------------------------
# 7. MAIN APPLICATION ENTRY POINT
# -----------------------------------------------------------------------------
def main():
    """Main function organizing the layout and executing UI components."""
    # Render Sidebar
    render_sidebar()

    # Render Main Area
    render_main_header()
    render_chat_interface()

if __name__ == "__main__":
    main()
