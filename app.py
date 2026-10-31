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
        # Map: sha256_hash -> {"filename": filename}
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

            for file in uploaded_files:
                file_bytes = file.getvalue()
                file_hash = compute_file_hash(file_bytes)
                size_str = format_file_size(file.size)

                # Duplicate detection via SHA-256 hash or filename
                is_hash_dup = file_hash in st.session_state.uploaded_file_hashes
                is_name_dup = file.name in st.session_state.uploaded_files_registry

                dup_tag = ""
                if is_hash_dup:
                    orig_name = st.session_state.uploaded_file_hashes[file_hash]["filename"]
                    dup_tag = f' <span style="color:#d29922; font-size:0.75rem;">(Duplicate of {html.escape(orig_name)})</span>'
                elif is_name_dup:
                    dup_tag = ' <span style="color:#d29922; font-size:0.75rem;">(Already uploaded)</span>'

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

                if not is_hash_dup and not is_name_dup:
                    new_files_to_process.append((file, file_hash))

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
                    save_path = os.path.join(data_dir, file.name)

                    try:
                        # Save file to disk
                        with open(save_path, "wb") as f:
                            f.write(file.getbuffer())

                        # Call backend ingestion function
                        result = backend_adapter.ingest_document(
                            save_path,
                            use_mock=st.session_state.use_mock
                        )

                        if result.get("success"):
                            st.session_state.uploaded_files_registry.add(file.name)
                            st.session_state.uploaded_file_hashes[f_hash] = {"filename": file.name}
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
        backend_error = None

        try:
            docs = backend_adapter.list_documents(use_mock=st.session_state.use_mock)
        except backend_adapter.BackendError as e:
            backend_error = str(e)
            st.error(f"❌ Backend Document Listing Error: {html.escape(backend_error)}")

        total_docs = len(docs)

        col_metric1, col_metric2 = st.columns(2)
        with col_metric1:
            st.metric("Documents", total_docs)
        with col_metric2:
            total_chunks = sum(d.get("total_chunks", 0) for d in docs)
            st.metric("Total Chunks", total_chunks)

        if total_docs == 0 and not backend_error:
            st.info("No documents indexed yet. Upload a PDF above to build your knowledge base.")
            # Optional demo data loader for mock mode
            if st.session_state.use_mock:
                if st.button("🧪 Load Sample Demo Document", use_container_width=True):
                    backend_adapter.load_demo_documents()
                    st.rerun()
        elif total_docs > 0:
            with st.expander(f"📋 Indexed Documents ({total_docs})", expanded=True):
                for doc in docs:
                    filename = html.escape(str(doc.get("filename", "unknown.pdf")))
                    pages = html.escape(str(doc.get("total_pages", "N/A")))
                    chunks = html.escape(str(doc.get("total_chunks", "N/A")))
                    is_mock_doc = doc.get("is_mock", st.session_state.use_mock)

                    if is_mock_doc:
                        status_html = '<span class="doc-status-simulated">● simulated (demo)</span>'
                    else:
                        status_html = '<span class="doc-status-badge">● indexed</span>'

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

        # Session Reset Option
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        with col_c2:
            if st.session_state.use_mock and st.button("🧹 Reset Docs", use_container_width=True):
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

    # Check indexed documents safely
    try:
        docs = backend_adapter.list_documents(use_mock=st.session_state.use_mock)
    except backend_adapter.BackendError:
        docs = []

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
                        <div class="welcome-feature-title"><span>🔍</span> Semantic Search</div>
                        <p class="welcome-feature-desc">Locates relevant documentation paragraphs even when phrasing differs.</p>
                    </div>
                    <div class="welcome-feature-box">
                        <div class="welcome-feature-title"><span>📑</span> Verifiable Citations</div>
                        <p class="welcome-feature-desc">Every response links directly to source document pages and excerpts.</p>
                    </div>
                    <div class="welcome-feature-box">
                        <div class="welcome-feature-title"><span>⚡</span> Code Intelligence</div>
                        <p class="welcome-feature-desc">Formatted Markdown answers with syntax-highlighted code snippets.</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Quick starter prompt chips for demonstration
        st.markdown("<span style='font-size:0.85rem; font-weight:600; color:#8b949e;'>💡 Quick Prompts:</span>", unsafe_allow_html=True)
        col_q1, col_q2, col_q3 = st.columns(3)
        with col_q1:
            if st.button("How do I create async API routes?", key="q1_btn", use_container_width=True):
                st.session_state.prompt_to_send = "How do I create async API routes in FastAPI?"
                st.rerun()
        with col_q2:
            if st.button("What are Python type annotations?", key="q2_btn", use_container_width=True):
                st.session_state.prompt_to_send = "What are the best practices for Python type annotations?"
                st.rerun()
        with col_q3:
            if st.button("Explain RAG chunking parameters", key="q3_btn", use_container_width=True):
                st.session_state.prompt_to_send = "Explain the RAG chunking strategy and parameters"
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
    user_query = st.chat_input("Ask a question about your uploaded documentation...")
    if st.session_state.prompt_to_send:
        user_query = st.session_state.prompt_to_send
        st.session_state.prompt_to_send = None

    # Handle New User Input
    if user_query:
        clean_prompt = user_query.strip()
        if not clean_prompt:
            st.warning("Please type a non-empty question.")
            return

        # TASK 5: Prevent question answering when no documents are indexed
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
                    use_mock=current_mode_is_mock
                )

                if response_data.get("success", False):
                    answer_text = response_data.get("answer", "No answer generated.")
                    sources = response_data.get("sources", [])
                    resp_is_mock = response_data.get("is_mock", current_mode_is_mock)

                    # Origin tag
                    if resp_is_mock:
                        st.markdown('<span class="msg-origin-tag msg-tag-mock">⚡ Demo Simulated Response</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="msg-origin-tag msg-tag-live">🔒 Live RAG Pipeline Output</span>', unsafe_allow_html=True)

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
