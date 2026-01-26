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

---

## 🏗️ Architecture

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

---

## 📁 Project Structure

```
rag_project/
│
├── rag/                          # Core RAG pipeline logic
│   ├── __init__.py
│   ├── loaders/                  # Document ingestion
│   │   ├── __init__.py
│   │   └── loader.py             # PDF, TXT, web loaders
│   │
│   ├── chunking/                 # Text splitting strategies
│   │   ├── __init__.py
│   │   ├── recursive.py          # Recursive character splitting
│   │   └── semantic.py           # Semantic-based chunking
│   │
│   ├── embeddings/               # Vector generation
│   │   ├── __init__.py
│   │   └── hf_embeddings.py      # HuggingFace sentence transformers
│   │
│   ├── vectorstore/              # Vector database management
│   │   ├── __init__.py
│   │   └── chroma_store.py       # ChromaDB integration
│   │
│   ├── retriever/                # Search and ranking
│   │   ├── __init__.py
│   │   ├── retriever.py          # Similarity search
│   │   └── reranker.py           # Result reranking
│   │
│   ├── prompts/                  # LLM prompt templates
│   │   ├── __init__.py
│   │   └── templates.py          # System/user prompts
│   │
│   └── llm/                      # LLM integration
│       ├── __init__.py
│       └── ollama_llm.py         # Local Ollama client
│
├── db/                           # Metadata & logging
│   ├── __init__.py
│   └── models.py                 # SQLite/PostgreSQL schemas
│
├── data/
│   ├── raw/                      # Source documents (gitignored)
│   ├── processed/                # Cleaned text chunks
│   └── chroma/                   # Vector database storage
│
├── notebooks/                    # Jupyter experiments
│   ├── 01_data_exploration.ipynb
│   ├── 02_chunking_strategies.ipynb
│   └── 03_retrieval_evaluation.ipynb
│
├── scripts/                      # CLI tools
│   ├── ingest.py                 # Batch document ingestion
│   ├── build_index.py            # Build vector index
│   └── query.py                  # Interactive query tool
│
├── tests/                        # Unit & integration tests
│   ├── test_chunking.py
│   ├── test_retrieval.py
│   └── test_pipeline.py
│
├── config.yaml                   # Configuration file
├── pyproject.toml                # Project metadata
├── requirements.txt              # Pinned dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🛠️ Technology Stack

| Component          | Technology                                    | Purpose                           |
|--------------------|-----------------------------------------------|-----------------------------------|
| **Language**       | Python 3.12+                                  | Core programming language         |
| **Framework**      | LangChain                                     | RAG orchestration                 |
| **Vector Store**   | ChromaDB                                      | Embedding storage & search        |
| **Embeddings**     | Sentence Transformers (all-MiniLM-L6-v2)      | Text → 384d vectors               |
| **LLM**            | Ollama (llama2, mistral, etc.)                | Local inference                   |
| **Database**       | SQLite / PostgreSQL                           | Metadata & logs                   |
| **Package Mgr**    | uv / pip                                      | Dependency management             |

---

## 🧠 Embedding Model: all-MiniLM-L6-v2

We use `sentence-transformers/all-MiniLM-L6-v2` which produces **384-dimensional embeddings**:

- ✅ **384 dimensions** capture semantic meaning
- ✅ Each dimension represents a fragment of context
- ✅ Similar meanings → vectors close in space
- ✅ No single dimension is interpretable
- ✅ **Meaning emerges from all 384 dimensions combined**

### Why This Model?

| Criteria      | Rating | Notes                                  |
|---------------|--------|----------------------------------------|
| Speed         | ⭐⭐⭐⭐⭐ | Fast inference (~5ms per sentence)     |
| Size          | ⭐⭐⭐⭐⭐ | Compact vectors (384d vs 768d/1024d)   |
| Quality       | ⭐⭐⭐⭐   | Good semantic understanding            |
| Memory        | ⭐⭐⭐⭐⭐ | Low RAM usage                          |

---

## 📌 Core RAG Components (One-Line Explanations)

### 1. **rag/**
Root module containing the full Retrieval-Augmented Generation pipeline.

### 2. **rag/loaders/**
Loads raw data from files or sources and converts it into clean text.
- `loader.py` — Handles ingestion of PDFs, text, web pages, or datasets.

### 3. **rag/chunking/**
Splits large documents into smaller, meaningful text chunks.
- `recursive.py` — Recursively splits text by structure while preserving context.
- `semantic.py` — Splits text based on semantic meaning rather than fixed size.

### 4. **rag/embeddings/**
Converts text chunks into numerical vector embeddings.
- `hf_embeddings.py` — Generates embeddings using HuggingFace models.

### 5. **rag/vectorstore/**
Stores and retrieves embeddings using a vector database.
- `chroma_store.py` — Manages embedding storage and similarity search via ChromaDB.

### 6. **rag/retriever/**
Fetches the most relevant chunks for a given user query.
- `retriever.py` — Performs vector similarity search.
- `reranker.py` — Reorders retrieved chunks for higher relevance and accuracy.

### 7. **rag/prompts/**
Contains prompt templates that guide how the LLM uses retrieved context.
- `templates.py` — System and user prompt templates.

### 8. **rag/llm/**
Handles interaction with the language model for final answer generation.
- `ollama_llm.py` — Sends context and queries to a local Ollama-hosted LLM.

### 9. **__init__.py (all folders)**
Marks directories as Python modules and enables clean imports.

---

## 🎯 Design Philosophy

| Principle                  | Description                                                          |
|----------------------------|----------------------------------------------------------------------|
| **Retrieval Quality First** | The quality of retrieved context matters more than LLM sophistication |
| **Honest Uncertainty**      | System rejects queries when relevant knowledge is missing            |
| **No Hallucinations**       | Answers must be grounded in retrieved documents                      |
| **Progressive Complexity**  | Simple → Correct → Scalable                                          |

> **Key Principle**: RAG is **80% retrieval and data quality**, **20% generation**.

---

## 🚀 Getting Started

### Prerequisites

- ✅ Python 3.12+
- ✅ [Ollama](https://ollama.ai/) installed and running
- ✅ `uv` package manager (optional but recommended)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd rag_project

# Install dependencies
pip install -r requirements.txt

# Or using uv (faster)
uv pip install -r requirements.txt

# Pull an LLM model (if using Ollama)
ollama pull llama2
```

### Quick Start

```bash
# 1. Place your documents in data/raw/
# Example: data/raw/company_docs.pdf

# 2. Process documents
python scripts/ingest.py

# 3. Build vector index
python scripts/build_index.py

# 4. Run queries
python scripts/query.py "What is the main topic discussed in the documents?"
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

## 📊 Development Status

| Feature                     | Status           | Priority |
|-----------------------------|------------------|----------|
| Document ingestion          | ✅ Done          | High     |
| Text chunking               | ✅ Done          | High     |
| Embedding generation        | ✅ Done          | High     |
| Vector storage (ChromaDB)   | ✅ Done          | High     |
| Similarity search           | 🚧 In Progress   | High     |
| LLM integration             | 🚧 In Progress   | High     |
| Reranking                   | 🚧 In Progress   | Medium   |
| Evaluation framework        | 📋 Planned       | Medium   |
| Query optimization          | 📋 Planned       | Low      |
| Web UI (Streamlit)          | 📋 Planned       | Low      |

**Legend**: ✅ Done | 🚧 In Progress | 📋 Planned

---

## 🗺️ Roadmap

### Phase 1: Core Pipeline ✅
- [x] Document ingestion (PDF, TXT)
- [x] Vector embeddings (all-MiniLM-L6-v2)
- [x] ChromaDB integration
- [ ] Similarity search (90% complete)

### Phase 2: LLM Integration 🚧
- [ ] Ollama integration
- [ ] Prompt engineering
- [ ] Answer generation
- [ ] Citation tracking

### Phase 3: Enhancement 📋
- [ ] Reranking layer (cross-encoder)
- [ ] Hybrid search (vector + keyword)
- [ ] Query expansion
- [ ] Multi-query retrieval

### Phase 4: Production 📋
- [ ] Evaluation metrics (precision, recall)
- [ ] Performance monitoring
- [ ] API deployment (FastAPI)
- [ ] Streamlit web UI
- [ ] Docker containerization

---

## ⚙️ Configuration

Edit `config.yaml` or set environment variables:

```yaml
# Vector Store
chroma:
  persist_directory: "./data/chroma"
  collection_name: "documents"
  distance_metric: "cosine"  # cosine, l2, ip

# Embeddings
embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cpu"  # cpu, cuda, mps
  batch_size: 32

# Chunking
chunking:
  chunk_size: 512
  chunk_overlap: 50
  strategy: "recursive"  # recursive, semantic

# Retrieval
retrieval:
  top_k: 5
  score_threshold: 0.7
  rerank: false

# LLM
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
```

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
```

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **LangChain** - RAG framework and orchestration
- **ChromaDB** - Vector storage and similarity search
- **Sentence Transformers** - High-quality embeddings
- **Ollama** - Local LLM inference
- **HuggingFace** - Open-source models

---

## 📧 Contact

For questions, feedback, or collaboration:

- 📧 Email: your- jyotiradityaparihar@gmail.com
- 💬 GitHub Issues: [Open an issue](https://github.com/your-repo/issues)
- [in] Linkdin : https://www.linkedin.com/in/jyotiraditya-singh-959488248/

---

## 📚 Additional Resources

- [RAG Best Practices](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Ollama Models](https://ollama.ai/library)
- [LangChain RAG Guide](https://python.langchain.com/docs/use_cases/question_answering/)

---

<div align="center">

**Built with ❤️ for production-grade RAG systems**

⭐ **Star this repo** if you find it helpful!

</div>