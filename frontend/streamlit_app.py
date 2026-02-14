import sys
from pathlib import Path
import yaml
import streamlit as st
import requests

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
# Streamlit setup
# ------------------------------------------------------------
st.set_page_config(page_title="RAG Control Panel", layout="wide")
st.title("RAG Control Panel")

checker = SystemChecker(PROJECT_ROOT)
db_status = checker.db_info()

# ------------------------------------------------------------
# Vector DB Status
# ------------------------------------------------------------
st.header("📦 Vector DB Status")

if db_status["exists"]:
    st.success("Vector DB loaded")
    st.json(db_status)
else:
    st.error("Vector DB not present")
    st.info("Upload documents and run ingestion to create the Vector DB.")

# ------------------------------------------------------------
# Upload Documents
# ------------------------------------------------------------
st.header("📄 Upload Documents")

files = st.file_uploader(
    "Upload files for ingestion",
    accept_multiple_files=True
)

if files:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for f in files:
        with open(RAW_DIR / f.name, "wb") as out:
            out.write(f.getbuffer())
    st.success("Files uploaded successfully")

# ------------------------------------------------------------
# Ingestion
# ------------------------------------------------------------
st.header("⚙️ Ingestion Pipeline")

if st.button("Run Ingestion"):
    with st.spinner("Running ingestion pipeline..."):
        result = run_ingestion(PROJECT_ROOT)

    st.success(
        f"Ingestion complete. Chunks ingested: {result.get('chunks_ingested', 0)}"
    )
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


# ------------------------------------------------------------
# Query
# ------------------------------------------------------------
st.header("💬 Query RAG")

if not db_status["exists"]:
    st.warning("Query disabled. No Vector DB found.")
else:
    query = st.text_input("Ask a question")

    if st.button("Ask") and query:
        with st.spinner("Thinking..."):
            response = requests.post(
                "http://127.0.0.1:8000/query",
                json={"query": query},
                timeout=300
            ).json()

        # ---------- Handle response properly ----------
        status = response.get("status")

        if status == "invalid_query":
            st.warning(response.get("answer"))

        elif status == "no_context":
            st.warning(response.get("answer"))

        elif status == "success":
            st.subheader("Answer")
            st.markdown(response.get("answer", ""))

            sources = response.get("sources", [])
            if sources:
                st.subheader("Sources")
                for src in sources:
                    st.markdown(f"- {src}")

        else:
            st.error("Unexpected response from backend")
            st.json(response)
