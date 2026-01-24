import streamlit as st
import requests
from typing import List
import time

# ----------------------------- 
# CONFIGURATION
# ----------------------------- 
API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="GRAVITY AI Base",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------- 
# CUSTOM THEME & STYLING
# ----------------------------- 
st.markdown(
    """
    <style>
    /* Main color palette: Purple, Black, Red accents */
    :root {
        --primary-purple: #8B5CF6;
        --dark-purple: #6D28D9;
        --accent-red: #EF4444;
        --dark-bg: #0F0F0F;
        --card-bg: #1A1A1A;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Global background */
    .stApp {
        background: linear-gradient(135deg, #0F0F0F 0%, #1A0B2E 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1A0B2E 0%, #0F0F0F 100%);
        border-right: 2px solid var(--primary-purple);
    }
    
    [data-testid="stSidebar"] .stRadio > label {
        color: #FFFFFF;
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    /* Radio buttons */
    [data-testid="stSidebar"] .stRadio > div {
        background: rgba(139, 92, 246, 0.1);
        border-radius: 12px;
        padding: 1rem;
        margin-top: 1rem;
    }
    
    [data-testid="stSidebar"] .stRadio label {
        color: #E0E0E0;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(139, 92, 246, 0.2);
        color: var(--primary-purple);
    }
    
    /* Title styling */
    h1 {
        color: #FFFFFF;
        font-weight: 800;
        background: linear-gradient(90deg, var(--primary-purple), var(--accent-red));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 2rem;
    }
    
    h2, h3 {
        color: #FFFFFF;
    }
    
    /* Card containers */
    .upload-container, .query-container {
        background: rgba(26, 26, 26, 0.8);
        border: 2px solid var(--primary-purple);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(139, 92, 246, 0.3);
        backdrop-filter: blur(10px);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(139, 92, 246, 0.05);
        border: 2px dashed var(--primary-purple);
        border-radius: 12px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: var(--accent-red);
        background: rgba(239, 68, 68, 0.05);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-purple), var(--dark-purple));
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(139, 92, 246, 0.4);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(139, 92, 246, 0.6);
        background: linear-gradient(135deg, var(--dark-purple), var(--accent-red));
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Text input */
    .stTextInput > div > div > input {
        background: rgba(26, 26, 26, 0.8);
        border: 2px solid var(--primary-purple);
        border-radius: 12px;
        color: white;
        padding: 1rem;
        font-size: 1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--accent-red);
        box-shadow: 0 0 0 2px rgba(239, 68, 68, 0.2);
    }
    
    /* Success/Warning/Error messages */
    .stSuccess, .stWarning, .stError {
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Answer box */
    .answer-box {
        background: rgba(26, 26, 26, 0.9);
        border-left: 4px solid var(--primary-purple);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #E0E0E0;
        line-height: 1.8;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--primary-purple), transparent);
        margin: 2rem 0;
    }
    
    /* Loading spinner customization */
    .stSpinner > div {
        border-top-color: var(--primary-purple) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------- 
# SIDEBAR NAVIGATION
# ----------------------------- 
with st.sidebar:
    st.markdown("# ⚡ GRAVITY AI")
    st.markdown("### Knowledge Base System")
    st.markdown("---")
    
    page = st.radio(
        "Navigate",
        ["📥 Load Documents", "💬 Ask Questions"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #808080; font-size: 0.8rem; margin-top: 2rem;'>
            Powered by AI<br>
            <span style='color: var(--primary-purple);'>GRAVITY Base v1.0</span>
        </div>
        """,
        unsafe_allow_html=True
    )

# ----------------------------- 
# HELPER FUNCTIONS
# ----------------------------- 
def make_api_request(endpoint: str, method: str = "POST", **kwargs):
    """Make API request with error handling"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        response = requests.request(method, url, timeout=30, **kwargs)
        return response
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to API server. Please ensure FastAPI is running.")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. Please try again.")
        return None
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        return None

# ====================================================== 
# PAGE 1: LOAD DOCUMENTS
# ====================================================== 
if page == "📥 Load Documents":
    st.title("📥 Load Documents")
    st.markdown("Upload your documents to build the knowledge base")
    
    st.markdown("<div class='upload-container'>", unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Choose PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="You can upload multiple files at once"
    )
    
    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) selected:**")
        for file in uploaded_files:
            file_size = len(file.getvalue()) / 1024  # KB
            st.markdown(f"- 📄 `{file.name}` ({file_size:.1f} KB)")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔮 Ingest Documents"):
            if not uploaded_files:
                st.warning("⚠️ Please upload at least one file before ingesting.")
            else:
                with st.spinner("Processing documents..."):
                    files = [
                        ("files", (f.name, f.getvalue(), f.type))
                        for f in uploaded_files
                    ]
                    
                    response = make_api_request("/ingest", files=files)
                    
                    if response and response.status_code == 200:
                        st.success("✅ Documents successfully ingested into knowledge base!")
                        time.sleep(0.5)
                        st.balloons()
                    elif response:
                        st.error(f"❌ Failed to ingest documents. Status: {response.status_code}")
                        if response.text:
                            with st.expander("View error details"):
                                st.code(response.text)

# ====================================================== 
# PAGE 2: ASK QUESTIONS
# ====================================================== 
elif page == "💬 Ask Questions":
    st.title("💬 Ask Questions")
    st.markdown("Query your knowledge base with natural language")
    
    st.markdown("<div class='query-container'>", unsafe_allow_html=True)
    
    query = st.text_input(
        "Your question",
        placeholder="e.g., What are the main topics covered in the documents?",
        label_visibility="collapsed"
    )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🧠 Get Answer"):
            if not query.strip():
                st.warning("⚠️ Please enter a question.")
            else:
                with st.spinner("Thinking..."):
                    response = make_api_request(
                        "/query",
                        json={"question": query}
                    )
                    
                    if response and response.status_code == 200:
                        answer = response.json().get("answer", "No answer provided.")
                        
                        st.markdown("---")
                        st.subheader("📌 Answer")
                        st.markdown(
                            f"<div class='answer-box'>{answer}</div>",
                            unsafe_allow_html=True
                        )
                        
                        # Optional: Show additional metadata if available
                        if "sources" in response.json():
                            with st.expander("📚 View sources"):
                                st.json(response.json()["sources"])
                    elif response:
                        st.error(f"❌ Failed to get answer. Status: {response.status_code}")
                        if response.text:
                            with st.expander("View error details"):
                                st.code(response.text)
    
    # Quick examples
    st.markdown("---")
    st.markdown("### 💡 Example Questions")
    examples = [
        "Summarize the key points from the documents",
        "What are the main recommendations?",
        "Explain the methodology used"
    ]
    
    cols = st.columns(len(examples))
    for idx, example in enumerate(examples):
        with cols[idx]:
            if st.button(example, key=f"example_{idx}", use_container_width=True):
                st.session_state.example_query = example
                st.rerun()
    
    # Handle example query selection
    if "example_query" in st.session_state:
        query = st.session_state.example_query
        del st.session_state.example_query