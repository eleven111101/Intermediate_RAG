import sys
from pathlib import Path
import yaml
import streamlit as st
import requests
from datetime import datetime

# ------------------------------------------------------------
# Ensure project root is importable
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rag_project.db_checker_files import SystemChecker
from rag_project.scripts.ingestion import run_ingestion

# ------------------------------------------------------------
# Load config
# ------------------------------------------------------------
with open(PROJECT_ROOT / "config.yaml") as f:
    config = yaml.safe_load(f)

RAW_DIR = PROJECT_ROOT / config["paths"]["data"]["raw_dir"]

# ------------------------------------------------------------
# Custom CSS for Beautiful Design
# ------------------------------------------------------------
@st.cache_resource
def load_custom_css():
    st.markdown("""
    <style>
    /* Global Styles */
    .stApp {
        background: #1a1a2e;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Main Title */
    .main-title {
        font-family: 'Courier New', 'Consolas', monospace;
        font-size: 3rem;
        font-weight: 700;
        color: #667eea;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    
    .subtitle {
        text-align: center;
        color: #a8b2d1;
        font-size: 1.1rem;
        margin-bottom: 3rem;
        font-weight: 300;
        letter-spacing: 1px;
    }
    
    /* Card Styles */
    .status-card {
        padding: 1.5rem 0;
        margin: 1rem 0;
    }
    
    /* Section Headers */
    .section-header {
        font-family: 'Courier New', 'Consolas', monospace;
        font-size: 1.5rem;
        font-weight: 700;
        color: #fff;
        margin: 2rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }
    
    .section-icon {
        font-size: 1.5rem;
    }
    
    /* Status Indicators */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.9rem;
        letter-spacing: 0.5px;
        margin: 0.5rem 0;
    }
    
    .status-active {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: #fff;
        box-shadow: 0 4px 15px rgba(56, 239, 125, 0.3);
    }
    
    .status-inactive {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: #fff;
        box-shadow: 0 4px 15px rgba(235, 51, 73, 0.3);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Input Fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        color: #fff;
        padding: 0.75rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.3);
    }
    
    /* File Uploader */
    .uploadedFile {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Success/Warning/Error Messages */
    .stSuccess, .stWarning, .stError, .stInfo {
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* JSON Display */
    .json-display {
        padding: 1rem;
        font-family: 'Courier New', 'Consolas', monospace;
        color: #a8b2d1;
        margin: 1rem 0;
    }
    
    /* Answer Box */
    .answer-box {
        border-left: 4px solid #667eea;
        padding: 1.5rem;
        margin: 1.5rem 0;
    }
    
    .answer-text {
        color: #e6f1ff;
        font-size: 1.1rem;
        line-height: 1.8;
    }
    
    /* Sources List */
    .sources-container {
        border-left: 3px solid #667eea;
        padding: 1rem 0 1rem 1.5rem;
        margin-top: 1.5rem;
    }
    
    .source-item {
        color: #a8b2d1;
        padding: 0.5rem 0;
        font-size: 0.95rem;
    }
    
    /* Stats Grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 2rem;
        margin: 1.5rem 0;
    }
    
    .stat-card {
        padding: 1rem;
        text-align: center;
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
        font-family: 'Courier New', 'Consolas', monospace;
    }
    
    .stat-label {
        color: #a8b2d1;
        font-size: 0.9rem;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: #667eea !important;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.5), transparent);
        margin: 2rem 0;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.2);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    </style>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------
# Streamlit setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="RAG Control Panel",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load custom CSS
load_custom_css()

# Header
st.markdown('<h1 class="main-title">🚀 RAG CONTROL PANEL</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Intelligent Document Retrieval & Query System</p>', unsafe_allow_html=True)

# Initialize system checker
checker = SystemChecker(PROJECT_ROOT)
db_status = checker.db_info()

# ------------------------------------------------------------
# Vector DB Status Section
# ------------------------------------------------------------
st.markdown('<div class="section-header"><span class="section-icon">📦</span>Vector Database Status</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    if db_status["exists"]:
        st.markdown('<span class="status-badge status-active">✓ OPERATIONAL</span>', unsafe_allow_html=True)
        
        # Display stats in a grid
        if "doc_count" in db_status or "chunk_count" in db_status:
            st.markdown('<div class="stats-grid">', unsafe_allow_html=True)
            
            if "doc_count" in db_status:
                st.markdown(f'''
                <div class="stat-card">
                    <div class="stat-value">{db_status["doc_count"]}</div>
                    <div class="stat-label">Documents</div>
                </div>
                ''', unsafe_allow_html=True)
            
            if "chunk_count" in db_status:
                st.markdown(f'''
                <div class="stat-card">
                    <div class="stat-value">{db_status["chunk_count"]}</div>
                    <div class="stat-label">Chunks</div>
                </div>
                ''', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Expandable detailed info
        with st.expander("📊 View Detailed Information"):
            st.json(db_status)
    else:
        st.markdown('<span class="status-badge status-inactive">✗ NOT INITIALIZED</span>', unsafe_allow_html=True)
        st.info("💡 **Getting Started:** Upload documents below and run the ingestion pipeline to create your vector database.")

with col2:
    st.markdown("**System Info**")
    st.markdown(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
    st.markdown(f"📁 Config loaded ✓")

st.markdown("<hr>", unsafe_allow_html=True)

# ------------------------------------------------------------
# Upload Documents Section
# ------------------------------------------------------------
st.markdown('<div class="section-header"><span class="section-icon">📄</span>Document Upload</div>', unsafe_allow_html=True)

files = st.file_uploader(
    "Drop your documents here or click to browse",
    accept_multiple_files=True,
    help="Supported formats: PDF, TXT, DOCX, and more"
)

if files:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    uploaded_files = []
    
    for f in files:
        file_path = RAW_DIR / f.name
        with open(file_path, "wb") as out:
            out.write(f.getbuffer())
        uploaded_files.append(f.name)
    
    st.success(f"✅ Successfully uploaded {len(uploaded_files)} file(s)")
    
    with st.expander("📋 View uploaded files"):
        for fname in uploaded_files:
            st.markdown(f"- 📄 {fname}")

st.markdown("<hr>", unsafe_allow_html=True)

# ------------------------------------------------------------
# Ingestion Pipeline Section
# ------------------------------------------------------------
st.markdown('<div class="section-header"><span class="section-icon">⚙️</span>Ingestion Pipeline</div>', unsafe_allow_html=True)

st.markdown("Process uploaded documents and build the vector database for semantic search.")

col1, col2 = st.columns([1, 3])

with col1:
    run_button = st.button("🚀 Run Ingestion", use_container_width=True)

with col2:
    if db_status["exists"]:
        st.info("💡 Running ingestion will update the existing vector database with new documents.")

if run_button:
    with st.spinner("🔄 Processing documents and building vector embeddings..."):
        try:
            result = run_ingestion(PROJECT_ROOT)
            
            chunks_ingested = result.get('chunks_ingested', 0)
            st.success(f"✅ Ingestion complete! Processed **{chunks_ingested}** chunks")
            
            # Display additional metrics if available
            if isinstance(result, dict):
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric("Chunks Ingested", chunks_ingested)
                
                if "processing_time" in result:
                    with metric_col2:
                        st.metric("Processing Time", f"{result['processing_time']:.2f}s")
            
            st.balloons()
            
            # Refresh the page to update DB status
            if hasattr(st, "rerun"):
                st.rerun()
            else:
                st.experimental_rerun()
                
        except Exception as e:
            st.error(f"❌ Ingestion failed: {str(e)}")

st.markdown("<hr>", unsafe_allow_html=True)

# ------------------------------------------------------------
# Query Interface Section
# ------------------------------------------------------------
st.markdown('<div class="section-header"><span class="section-icon">💬</span>Query Interface</div>', unsafe_allow_html=True)

if not db_status["exists"]:
    st.warning("⚠️ Query interface disabled. Please initialize the vector database first.")
else:
    st.markdown("Ask questions about your documents using natural language.")
    
    query = st.text_input(
        "Enter your question",
        placeholder="e.g., What are the main findings in the research papers?",
        label_visibility="collapsed"
    )
    
    col1, col2, col3 = st.columns([1, 1, 3])
    
    with col1:
        ask_button = st.button("🔍 Ask", use_container_width=True)
    
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.rerun()
    
    if ask_button and query:
        with st.spinner("🤔 Analyzing your question and retrieving relevant information..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/query",
                    json={"query": query},
                    timeout=300
                ).json()
                
                status = response.get("status")
                
                if status == "invalid_query":
                    st.warning(f"⚠️ {response.get('answer')}")
                
                elif status == "no_context":
                    st.warning(f"ℹ️ {response.get('answer')}")
                
                elif status == "success":
                    # Display answer in styled box
                    st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                    st.markdown("### 💡 Answer")
                    st.markdown(f'<div class="answer-text">{response.get("answer", "")}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Display sources if available
                    sources = response.get("sources", [])
                    if sources:
                        st.markdown('<div class="sources-container">', unsafe_allow_html=True)
                        st.markdown("### 📚 Sources")
                        for i, src in enumerate(sources, 1):
                            st.markdown(f'<div class="source-item">{i}. {src}</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                
                else:
                    st.error("❌ Unexpected response from backend")
                    with st.expander("🔍 View raw response"):
                        st.json(response)
                        
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to the backend API. Please ensure the server is running at http://127.0.0.1:8000")
            except requests.exceptions.Timeout:
                st.error("❌ Request timed out. The query is taking too long to process.")
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    '<p style="text-align: center; color: #a8b2d1; font-size: 0.9rem; margin-top: 2rem;">Built with ❤️ by PHOENIX CyberSecurity • Powered by Gravity_AI</p>',
    unsafe_allow_html=True
)