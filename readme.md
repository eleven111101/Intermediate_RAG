# 🚀 Intermediate RAG System

> A production-oriented Retrieval-Augmented Generation (RAG) pipeline built with open-source tools. This system prioritizes correctness, debuggability, and scalability over quick demos.

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()

---

## 📖 Overview

This project implements a complete RAG system that:

- ✅ **Ingests and processes** documents (TXT, PDF)
- ✅ **Generates semantic embeddings** using sentence transformers
- ✅ **Performs vector similarity search** with ChromaDB
- ✅ **Generates grounded answers** using local LLMs via Ollama
- ✅ **Maintains metadata and logs** in a traditional database
- ✅ **Automatic pipeline orchestration** with intelligent system checks

---

## 🔧 Current Capabilities

| Feature | Status | Notes |
|---------|--------|-------|
| Automatic ingestion pipeline | ✅ Complete | Single entry-point execution |
| Recursive + semantic chunking | ✅ Complete | Dual-strategy approach |
| Sentence-Transformer embeddings | ✅ Complete | 384-dimensional vectors |
| Persistent ChromaDB vector store | ✅ Complete | HNSW + cosine similarity |
| Query-time vector retrieval | ✅ Complete | Top-K similarity search |
| Optional reranking layer | 🚧 In Progress | Placeholder implemented |
| LLM answer generation | 📋 Planned | Next major milestone |

**Legend**: ✅ Complete | 🚧 In Progress | 📋 Planned

---

## 🏗️ Architecture

### System Flow Diagram

```
┌──────────────────┐
│   User Query     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Query Embedding  │ ← Sentence Transformers (384d)
└────────┬─────────┘
         │
         ▼
┌─────────────────────────┐
│ ChromaDB Vector Search  │ ← Cosine similarity search
└────────┬────────────────┘
         │
         ▼
┌────────────────────────┐
│ Top-K Relevant Chunks  │ ← Retrieve 3-5 most relevant
└────────┬───────────────┘
         │
         ▼
┌───────────────────────┐
│  Context Formatting   │ ← Build prompt with context
└────────┬──────────────┘
         │
         ▼
┌──────────────────────┐
│ Local LLM (Ollama)   │ ← Generate grounded answer
└────────┬─────────────┘
         │
         ▼
┌─────────────────────────────┐
│ Grounded Answer + Sources   │
└─────────────────────────────┘
```

### High-Level Architecture

```
main.py
  └── checker_files.py
        ├── checks raw data
        ├── checks vector DB
        └── decides: INGEST or QUERY

INGESTION FLOW
  raw data
    → loaders
    → recursive chunking
    → semantic chunking
    → embeddings
    → ChromaDB (persistent)

QUERY FLOW
  user query
    → embeddings
    → retriever (vector search)
    → reranker (optional)
    → context output
    → LLM generation (planned)
```

---

## 📁 Project Structure

```
INTERMEDIATE_RAG/
│
├── main.py                       # Single entry point - automatic mode detection
├── config.yaml                   # System configuration
├── requirements.txt              # Pinned dependencies
├── pyproject.toml               # Project metadata
├── .gitignore
├── LICENSE
└── README.md
│
├── rag_project/
│   ├── checker_files.py          # System validator & decision engine
│   │
│   ├── scripts/
│   │   ├── ingestion.py          # Ingestion pipeline orchestrator
│   │   └── query_rag.py          # Query pipeline orchestrator
│   │
│   └── rag/                      # Core RAG pipeline logic
│       │
│       ├── __init__.py
│       │
│       ├── loaders/              # Document ingestion
│       │   ├── __init__.py
│       │   └── loader.py         # PDF, TXT, web loaders
│       │
│       ├── chunking/             # Text splitting strategies
│       │   ├── __init__.py
│       │   ├── recursive.py      # Recursive character splitting
│       │   └── semantic.py       # Semantic-based chunking
│       │
│       ├── embeddings/           # Vector generation
│       │   ├── __init__.py
│       │   └── hf_embeddings.py  # HuggingFace sentence transformers
│       │
│       ├── chromaDB/             # Vector database management
│       │   ├── __init__.py
│       │   └── chroma_store.py   # ChromaDB integration
│       │
│       ├── retriever/            # Search and ranking
│       │   ├── __init__.py
│       │   ├── retriever.py      # Similarity search
│       │   └── reranker.py       # Result reranking (placeholder)
│       │
│       ├── prompts/              # LLM prompt templates
│       │   ├── __init__.py
│       │   └── templates.py      # System/user prompts
│       │
│       └── llm/                  # LLM integration
│           ├── __init__.py
│           └── ollama_llm.py     # Local Ollama client (planned)
│
├── db/                           # Metadata & logging
│   ├── __init__.py
│   └── models.py                 # SQLite/PostgreSQL schemas
│
├── data/
│   ├── raw/                      # Source documents (gitignored)
│   ├── processed/                # Cleaned text chunks
│   └── chroma/                   # Vector database storage (gitignored)
│
├── vector_store/
│   └── chroma/                   # Persistent ChromaDB storage
│
├── notebooks/                    # Jupyter experiments
│   ├── 01_data_exploration.ipynb
│   ├── 02_chunking_strategies.ipynb
│   └── 03_retrieval_evaluation.ipynb
│
└── tests/                        # Unit & integration tests
    ├── test_chunking.py
    ├── test_retrieval.py
    └── test_pipeline.py
```

---

## 🛠️ Technology Stack

| Component          | Technology                                    | Purpose                           |
|--------------------|-----------------------------------------------|-----------------------------------|
| **Language**       | Python 3.12+                                  | Core programming language         |
| **Framework**      | LangChain                                     | RAG orchestration                 |
| **Vector Store**   | ChromaDB                                      | Embedding storage & search        |
| **Embeddings**     | Sentence Transformers (all-MiniLM-L6-v2)      | Text → 384d vectors               |
| **LLM**            | Ollama (llama2, mistral, etc.)                | Local inference (planned)         |
| **Database**       | SQLite / PostgreSQL                           | Metadata & logs                   |
| **Package Mgr**    | uv / pip                                      | Dependency management             |

---

## 🧠 Understanding the Components

### What is RAG?

Retrieval-Augmented Generation (RAG) combines:

- **Information Retrieval** (vector search)
- **Language Models** (LLMs)

Instead of relying only on model memory, the LLM is given relevant retrieved context at query time.

```
User Query
  → Embed query
  → Retrieve relevant chunks
  → Provide context to LLM
  → Generate grounded answer
```

### Embedding Model: all-MiniLM-L6-v2

We use `sentence-transformers/all-MiniLM-L6-v2` which produces **384-dimensional embeddings**:

- ✅ **384 dimensions** capture semantic meaning
- ✅ Each dimension represents a fragment of context
- ✅ Similar meanings → vectors close in space
- ✅ No single dimension is interpretable
- ✅ **Meaning emerges from all 384 dimensions combined**

#### Why This Model?

| Criteria      | Rating | Notes                                  |
|---------------|--------|----------------------------------------|
| Speed         | ⭐⭐⭐⭐⭐ | Fast inference (~5ms per sentence)     |
| Size          | ⭐⭐⭐⭐⭐ | Compact vectors (384d vs 768d/1024d)   |
| Quality       | ⭐⭐⭐⭐   | Good semantic understanding            |
| Memory        | ⭐⭐⭐⭐⭐ | Low RAM usage                          |

---

## 📌 Core RAG Components

### 1. **rag/**
Root module containing the full Retrieval-Augmented Generation pipeline.

### 2. **rag/loaders/**
Loads raw data from files or sources and converts it into clean text.
- `loader.py` — Handles ingestion of PDFs, text, web pages, or datasets.

### 3. **rag/chunking/**
Splits large documents into smaller, meaningful text chunks.
- `recursive.py` — Recursively splits text by structure while preserving context.
- `semantic.py` — Splits text based on semantic meaning rather than fixed size.

**Chunking Strategy:**
```
Recursive chunking → preserves structure
Semantic chunking → improves meaning coherence
Both applied sequentially (intermediate RAG level)
```

### 4. **rag/embeddings/**
Converts text chunks into numerical vector embeddings.
- `hf_embeddings.py` — Generates embeddings using HuggingFace models.

### 5. **rag/chromaDB/**
Stores and retrieves embeddings using a vector database.
- `chroma_store.py` — Manages embedding storage and similarity search via ChromaDB.
- Uses HNSW + cosine similarity for efficient retrieval

### 6. **rag/retriever/**
Fetches the most relevant chunks for a given user query.
- `retriever.py` — Performs vector similarity search.
- `reranker.py` — Reorders retrieved chunks for higher relevance and accuracy (placeholder).

### 7. **rag/prompts/**
Contains prompt templates that guide how the LLM uses retrieved context.
- `templates.py` — System and user prompt templates.

### 8. **rag/llm/**
Handles interaction with the language model for final answer generation.
- `ollama_llm.py` — Sends context and queries to a local Ollama-hosted LLM (planned).

### 9. **checker_files.py**
System validator and decision engine that:
- Checks if raw data exists
- Checks if vector DB exists
- Automatically decides: INGEST or QUERY mode
- No manual flags or mode switching required

---

## 🎯 Design Philosophy

| Principle                  | Description                                                          |
|----------------------------|----------------------------------------------------------------------|
| **Retrieval Quality First** | The quality of retrieved context matters more than LLM sophistication |
| **Honest Uncertainty**      | System rejects queries when relevant knowledge is missing            |
| **No Hallucinations**       | Answers must be grounded in retrieved documents                      |
| **Progressive Complexity**  | Simple → Correct → Scalable                                          |
| **Clean Separation**        | Modular design for easy testing and deployment                       |
| **Automatic Orchestration** | Intelligent system checks eliminate manual configuration             |

> **Key Principle**: RAG is **80% retrieval and data quality**, **20% generation**.

---

## 🚀 Getting Started

### Prerequisites

- ✅ Python 3.12+
- ✅ [Ollama](https://ollama.ai/) installed and running (for LLM generation - planned)
- ✅ `uv` package manager (optional but recommended)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd INTERMEDIATE_RAG

# Install dependencies
pip install -r requirements.txt

# Or using uv (faster)
uv pip install -r requirements.txt

# Pull an LLM model (if using Ollama - for future use)
ollama pull llama2
```

### Quick Start

```bash
# 1. Place your documents in data/raw/
# Example: data/raw/company_docs.pdf, data/raw/report.txt

# 2. Run the system (automatic mode detection)
python main.py

# System will automatically:
# - Detect if ingestion is needed
# - Process documents and build vector index
# - OR allow you to query if index exists
```

### How the System Runs

**Single Command:**
```bash
python main.py
```

**What Happens Automatically:**

1. **System Check**
   - Raw data exists?
   - ChromaDB exists?

2. **Decision**
   - ❌ No DB → run ingestion
   - ✅ DB exists → run query

**No manual flags. No mode switching.**

---

## 📥 Ingestion Pipeline

### What It Does

1. Load raw documents (.txt, .pdf)
2. Recursive chunking (structure-based)
3. Temporary embeddings
4. Semantic chunking (meaning-based)
5. Final embeddings
6. Store vectors in ChromaDB (persistent)

### Sample Output

```
==============================
 RAG INGESTION PIPELINE STARTED
==============================

[STEP 1] Loading raw documents
→ Loaded documents: 1

[STEP 2] Recursive chunking
→ Recursive chunks: 67

[STEP 3] Generating temporary embeddings
→ Temporary embeddings generated

[STEP 4] Semantic chunking
→ Semantic chunks: 128

[STEP 5] Generating final embeddings
→ Final embeddings: 128

[STEP 6] Storing in ChromaDB
→ Stored 128 chunks in ChromaDB

==============================
 INGESTION PIPELINE COMPLETED
==============================
```

---

## 🔍 Query Pipeline

### What It Does

1. Load persistent ChromaDB
2. Display DB metadata
3. Accept user query
4. Embed query
5. Retrieve top-K chunks
6. (Optional) rerank
7. Print retrieved context
8. (Planned) Generate answer with LLM

### Sample Output

```
==============================
 RAG QUERY SERVICE
==============================
Vector DB Path  : vector_store/chroma
Collection Name: documents
Top-K          : 5
==============================

Ask a question: What is The Ashwa Riders?

=== RETRIEVED CONTEXT ===

[1] THE ASHWA RIDERS
OFF-ROAD ATV BUSINESS OVERVIEW...
----

[2] Marketing Strategy
The Ashwa Riders focuses on...
----
```

---

## 💻 Usage Examples

### Basic Query

```python
from rag import RAGPipeline

# Initialize the pipeline
pipeline = RAGPipeline()

# Query the system
result = pipeline.query("What is the main topic discussed in the documents?")

print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
print(f"Confidence: {result['confidence']}")
```

### Advanced Configuration

```python
from rag import RAGPipeline

pipeline = RAGPipeline(
    chunk_size=512,
    chunk_overlap=50,
    top_k=5,
    model_name="llama2",
    temperature=0.7
)

result = pipeline.query(
    query="Explain the product roadmap",
    temperature=0.7,
    max_tokens=500,
    return_sources=True
)
```

### Batch Processing

```python
from rag import RAGPipeline

pipeline = RAGPipeline()

queries = [
    "What are the key features?",
    "Who are the competitors?",
    "What is the pricing model?"
]

results = pipeline.batch_query(queries)

for query, result in zip(queries, results):
    print(f"Q: {query}")
    print(f"A: {result['answer']}\n")
```

---

## ⚙️ Configuration

Edit `config.yaml` to customize system behavior:

```yaml
# Vector Store
chroma:
  persist_directory: "./vector_store/chroma"
  collection_name: "documents"
  distance_metric: "cosine"  # cosine, l2, ip

# Embeddings
embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cpu"  # cpu, cuda, mps
  batch_size: 32

# Chunking
chunking:
  # Recursive chunking
  chunk_size: 512
  chunk_overlap: 50
  strategy: "recursive"  # recursive, semantic
  
  # Semantic chunking
  similarity_threshold: 0.5
  min_chunk_size: 100

# Retrieval
retrieval:
  top_k: 5
  score_threshold: 0.7
  rerank: false
  rerank_top_n: 10

# LLM (Planned)
llm:
  provider: "ollama"
  model: "llama2"  # llama2, mistral, codellama
  base_url: "http://localhost:11434"
  temperature: 0.7
  max_tokens: 500
  top_p: 0.9

# Logging
logging:
  level: "INFO"
  file: "logs/rag.log"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

---

## 📊 Vector Database Comparison

Use this table to choose a Vector DB based on your actual needs, not hype.

| Vector DB        | Speed | Cost | Scale | Simplicity | Metadata | Cloud/Local | Stage              | Best Use Case                          |
|------------------|-------|------|-------|------------|----------|-------------|--------------------|----------------------------------------|
| **FAISS**        | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ❌ | Local | Learning/Research | Maximum speed, custom systems, research |
| **ChromaDB**     | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | Local | Learning/Prototyping | RAG pipelines, local apps, fast iteration |
| **Qdrant**       | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | Both | Learning→Production | Strong filtering, self-hosted or cloud |
| **Weaviate**     | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ | Both | Production | Hybrid search, schema-based retrieval |
| **Milvus**       | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ✅ | Both | Production (Large) | Billions of vectors, distributed systems |
| **Pinecone**     | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | Cloud | Production | Managed service, zero ops |
| **Elasticsearch**| ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ✅ | Both | Production | Keyword + vector hybrid search |
| **OpenSearch**   | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ✅ | Both | Production | Open-source ES alternative |

### 🧠 How to Read This Table

| Dimension    | Explanation                                                                 |
|--------------|-----------------------------------------------------------------------------|
| **Speed**    | Raw similarity search performance (FAISS & Milvus are fastest)              |
| **Cost**     | ⭐⭐⭐⭐⭐ = free/local, ⭐⭐ = paid/managed infrastructure                          |
| **Scale**    | How many vectors you can handle (ChromaDB → millions, Milvus → billions)    |
| **Simplicity**| How fast you can get started with minimal infrastructure                   |
| **Metadata** | Ability to store & filter by document info (critical for real RAG systems)  |

### 💡 Our Choice: ChromaDB

We chose **ChromaDB** for this project because:

- ✅ **Zero ops** - No server setup required
- ✅ **Local first** - Perfect for learning & prototyping
- ✅ **Metadata support** - Filter by document properties
- ✅ **Fast enough** - Handles millions of vectors
- ✅ **Easy to upgrade** - Can migrate to Qdrant/Weaviate later
- ✅ **Persistent storage** - No re-ingestion needed

---

## 📊 Development Status & Roadmap

### Current Status

| Feature                     | Status           | Priority | Notes |
|-----------------------------|------------------|----------|-------|
| Document ingestion          | ✅ Done          | High     | PDF, TXT supported |
| Recursive text chunking     | ✅ Done          | High     | Structure-based |
| Semantic text chunking      | ✅ Done          | High     | Meaning-based |
| Embedding generation        | ✅ Done          | High     | all-MiniLM-L6-v2 |
| Vector storage (ChromaDB)   | ✅ Done          | High     | Persistent HNSW |
| Similarity search           | ✅ Done          | High     | Top-K retrieval |
| Automatic orchestration     | ✅ Done          | High     | Smart mode detection |
| Reranking                   | ✅ Done    | Medium   | Placeholder ready |
| LLM integration             | ✅ Done        | High     | Ollama integration |
| Prompt templates            | ✅ Done        | High     | Context formatting |
| Evaluation framework        | 🚧 In Progress        | Medium   | Metrics & testing |
| Query optimization          | 🚧 In Progress      | Medium   | Hybrid search |
| Web UI (Streamlit)          | 🚧 In Progress       | Low      | User interface |
| API (FastAPI)               | 🚧 In Progress        | Low      | REST endpoints |

**Legend**: ✅ Done | 🚧 In Progress | 📋 Planned

### Roadmap

#### Phase 1: Core Pipeline ✅ (COMPLETE)
- [x] Document ingestion (PDF, TXT)
- [x] Recursive chunking
- [x] Semantic chunking
- [x] Vector embeddings (all-MiniLM-L6-v2)
- [x] ChromaDB integration
- [x] Similarity search
- [x] Automatic orchestration

#### Phase 2: LLM Integration 🚧 (NEXT)
- [x] Ollama integration
- [x] Prompt engineering
- [ ] Answer generation
- [ ] Citation tracking
- [ ] Context window management

#### Phase 3: Enhancement 📋
- [ ] Cross-encoder reranking
- [ ] Hybrid search (vector + keyword)
- [ ] Query expansion
- [ ] Multi-query retrieval
- [ ] Metadata filtering

#### Phase 4: Production 📋
- [ ] Evaluation metrics (precision, recall, F1)
- [ ] Performance monitoring
- [ ] API deployment (FastAPI)
- [ ] Streamlit web UI
- [ ] Docker containerization
- [ ] CI/CD pipeline

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=rag tests/

# Run specific test module
pytest tests/test_retrieval.py -v

# Run with verbose output
pytest -vv

# Generate HTML coverage report
pytest --cov=rag --cov-report=html tests/

# Run only unit tests
pytest tests/ -m unit

# Run only integration tests
pytest tests/ -m integration
```

---

## 🧩 Why This Architecture?

1. **Clean separation of concerns**
   - Each component has a single responsibility
   - Easy to test and debug

2. **Production-aligned structure**
   - Follows industry best practices
   - Scales from prototype to production

3. **Easy to convert into microservices**
   - Each module can become a FastAPI service
   - Ready for containerization

4. **Easy DB or model swaps**
   - Abstracted interfaces
   - Plug-and-play components

5. **Debuggable and testable**
   - Clear data flow
   - Comprehensive logging

6. **Automatic orchestration**
   - No manual mode switching
   - Intelligent system checks

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run linting
ruff check .

# Format code
black .

# Type checking
mypy rag/

# Run tests
pytest
```

### Code Style

- Follow PEP 8
- Use type hints
- Write docstrings (Google style)
- Keep functions focused and small
- Add tests for new features

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **LangChain** - RAG framework and orchestration
- **ChromaDB** - Vector storage and similarity search
- **Sentence Transformers** - High-quality embeddings
- **Ollama** - Local LLM inference
- **HuggingFace** - Open-source models and community

---

## 📧 Contact

For questions, feedback, or collaboration:

- 📧 **Email**: jyotiradityaparihar@gmail.com
- 💬 **GitHub Issues**: [Open an issue](https://github.com/your-repo/issues)
- 💼 **LinkedIn**: [Jyotiraditya Singh](https://www.linkedin.com/in/jyotiraditya-singh-959488248/)

---

## 📚 Additional Resources

### Learning RAG
- [RAG Best Practices](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [LangChain RAG Guide](https://python.langchain.com/docs/use_cases/question_answering/)
- [Building Production RAG Systems](https://www.anyscale.com/blog/a-comprehensive-guide-for-building-rag-based-llm-applications-part-1)

### Documentation
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [Ollama Models](https://ollama.ai/library)
- [LangChain Docs](https://python.langchain.com/)

### Papers
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)

---

## ✨ Status Summary

**System is stable and working up to vector retrieval.**

- ✅ Ingestion pipeline: **Complete**
- ✅ Vector storage: **Complete**
- ✅ Query retrieval: **Complete**
- 🚧 LLM integration: **In Progress**
- 📋 Production features: **Planned**

**Ready for LLM integration and production deployment.**

---

<div align="center">

**Built with ❤️ by GRAVITY-AI for production-grade RAG systems**

⭐ **Star this repo** if you find it helpful !

</div>