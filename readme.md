# 🚀 Intermediate RAG System

> A **production-oriented Retrieval-Augmented Generation (RAG) system** built with open-source tools.  
> Designed for **correctness, observability, and real-world usage**, not demos.

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()

---

## 📖 Overview

This project implements a **complete, end-to-end RAG system** with:

- 📄 **Document upload and ingestion** - Support for PDF and TXT files
- ✂️ **Recursive + semantic chunking** - Structure and meaning-aware text splitting
- 🧠 **Embedding generation** - Sentence transformers for semantic understanding
- 📦 **Persistent vector storage** - ChromaDB for efficient similarity search
- 🔍 **Retrieval + reranking** - Top-K cosine search with optional reranking
- 🤖 **Local LLM answer generation** - Privacy-first inference with Ollama
- 🌐 **API layer** - FastAPI backend for programmatic access
- 🖥️ **UI control panel** - Streamlit interface for easy interaction
- 📊 **Full backend logging & timing** - Production-grade observability

This is **not a tutorial project** — it mirrors how RAG systems are built in production environments.

---

## 🔧 Current Capabilities

| Feature | Status | Notes |
|---------|--------|-------|
| Document upload (UI) | ✅ Complete | Streamlit interface |
| Ingestion pipeline | ✅ Complete | Logged & timed |
| Recursive chunking | ✅ Complete | Structure-aware splitting |
| Semantic chunking | ✅ Complete | Meaning-aware splitting |
| Sentence-Transformer embeddings | ✅ Complete | 384-dimensional vectors |
| ChromaDB persistent storage | ✅ Complete | Local vector database |
| Vector similarity retrieval | ✅ Complete | Top-K cosine search |
| Optional reranking | ✅ Implemented | Lightweight reranker |
| Local LLM inference | ✅ Complete | Ollama integration |
| FastAPI backend | ✅ Complete | RESTful JSON API |
| Streamlit control panel | ✅ Complete | Upload, ingest, query |
| System health checks | ✅ Complete | Database-aware monitoring |
| Full logging & timing | ✅ Complete | Ingestion, query, system logs |

---

## 🏗️ Architecture

### High-Level Flow

```
User (UI / API)
       ↓
    FastAPI
       ↓
  Query Pipeline
       ↓
Vector Retrieval (ChromaDB)
       ↓
  Context Assembly
       ↓
  Local LLM (Ollama)
       ↓
Grounded Answer (JSON)
```

### Ingestion Flow

```
Uploaded Documents
    → Loaders (PDF/TXT)
    → Recursive Chunking
    → Semantic Chunking
    → Embeddings Generation
    → ChromaDB (Persistent Storage)
```

### Query Flow

```
User Query
    → Query Embedding
    → Vector Search (Top-K)
    → Optional Reranking
    → Context Formatting
    → LLM Generation
    → Grounded Answer
```

---

## 📁 Project Structure

```
INTERMEDIATE_RAG/
│
├── frontend/
│   └── streamlit_app.py          # UI control panel
│
├── rag_project/
│   ├── api/
│   │   └── app.py                # FastAPI service
│   │
│   ├── checker_files.py          # Database & system checks
│   │
│   ├── scripts/
│   │   ├── ingestion.py          # Document ingestion pipeline
│   │   └── query_rag.py          # Query processing pipeline
│   │
│   └── rag/
│       ├── loaders/              # PDF / TXT document loaders
│       ├── chunking/             # Recursive + semantic chunking
│       ├── embeddings/           # Sentence transformer integration
│       ├── chromaDB/             # Vector store management
│       ├── retriever_files/      # Retriever + reranker logic
│       ├── prompts/              # LLM prompt templates
│       └── llm/                  # Ollama LLM integration
│
├── utils/
│   ├── logger.py                 # Unified logging utility
│   └── timer.py                  # Execution timing decorator
│
├── logs/                         # Runtime logs (gitignored)
│   ├── query.log
│   ├── ingestion.log
│   ├── system.log
│   └── app.log
│
├── data/
│   └── raw/                      # Uploaded documents storage
│
├── vector_store/
│   └── chroma/                   # Persistent ChromaDB storage
│
├── config.yaml                   # Central configuration file
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── .gitignore                    # Git ignore rules
```

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Language** | Python 3.12+ | Core development language |
| **Embeddings** | Sentence Transformers<br>(MiniLM-L6-v2) | Text to vector conversion |
| **Vector DB** | ChromaDB | Similarity search & storage |
| **LLM** | Ollama (local) | Answer generation |
| **API** | FastAPI | Backend REST interface |
| **UI** | Streamlit | Interactive control panel |
| **Logging** | Python logging | System observability |

---

## 🧠 Embedding Model Details

**Model:** `sentence-transformers/all-MiniLM-L6-v2`

A lightweight, fast, and effective embedding model optimized for semantic similarity tasks.

**Specifications:**
- **Dimensions:** 384
- **Speed:** ⭐⭐⭐⭐⭐ (Very fast inference)
- **Accuracy:** ⭐⭐⭐⭐ (Strong semantic understanding)
- **Memory:** ⭐⭐⭐⭐⭐ (Low memory footprint)

**Why this model?**
- Fast inference suitable for real-time applications
- Low memory usage enables deployment on modest hardware
- Strong semantic similarity performance for RAG tasks
- Well-supported by the Sentence Transformers library

---

## 📦 Vector Database: ChromaDB

**Why ChromaDB?**

- ✅ **Local-first** - No external server required
- ✅ **Zero setup** - Works out of the box
- ✅ **Metadata support** - Rich filtering capabilities
- ✅ **Persistent storage** - Data survives restarts
- ✅ **Migration path** - Easy to upgrade to Qdrant/Weaviate if needed

ChromaDB provides the perfect balance between simplicity and functionality for intermediate RAG systems.

---

## 🌐 FastAPI Backend

The API provides a clean, RESTful interface for RAG queries.

**Main Endpoint:** `POST /query`

**Request Format:**
```json
{
  "query": "What is Retrieval-Augmented Generation?"
}
```

**Response Format:**
```json
{
  "answer": "Retrieval-Augmented Generation (RAG) is...",
  "sources": [
    {
      "content": "...",
      "metadata": {...}
    }
  ],
  "retrieval_time": 0.22,
  "generation_time": 31.13
}
```

**Features:**
- JSON-only responses for easy integration
- Defensive error handling (`no_context`, `invalid_query`)
- Comprehensive logging with execution timing
- Health check endpoints for monitoring

---

## 🖥️ Streamlit Control Panel

An intuitive web interface for managing your RAG system.

**Features:**
- 📄 **Upload documents** - Drag and drop PDF/TXT files
- ⚙️ **Run ingestion** - Process documents manually
- 📦 **Database status** - Monitor vector store health
- 💬 **Query interface** - Interactive RAG queries
- 🛑 **Safe operations** - Graceful handling when DB is missing

**Design Philosophy:**
The UI is a thin client that delegates all logic to backend services, ensuring clean separation of concerns.

---

## 📊 Logging & Observability

All backend operations are logged with timestamps and execution metrics for production-grade observability.

| Log File | Purpose |
|----------|---------|
| `query.log` | Query processing, retrieval, and LLM timing |
| `ingestion.log` | Document chunking and embedding generation |
| `system.log` | Database health checks and system status |
| `app.log` | API lifecycle and request handling |

**Example Log Output:**
```
[2025-02-09 14:23:45] [INFO] RAG-QUERY - Query received: "What is RAG?"
[2025-02-09 14:23:45] [INFO] RAG-QUERY - Retrieval took 0.22s
[2025-02-09 14:23:45] [INFO] RAG-QUERY - Retrieved 5 chunks
[2025-02-09 14:24:16] [INFO] RAG-QUERY - LLM generation took 31.13s
[2025-02-09 14:24:16] [INFO] RAG-QUERY - Response generated successfully
```

---

## 🚀 Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.12+** - [Download](https://www.python.org/downloads/)
- **Ollama** - [Installation Guide](https://ollama.ai/)
- **pip** or **uv** - Python package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd INTERMEDIATE_RAG
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Pull an Ollama model:**
   ```bash
   ollama pull mistral
   ```
   
   *Alternatively, you can use other models like `llama2`, `phi`, etc.*

4. **Configure the system:**
   
   Edit `config.yaml` to customize:
   - Embedding model
   - Chunk sizes
   - LLM model name
   - Vector store settings

### Running the System

**1. Start the FastAPI Backend:**
```bash
uvicorn rag_project.api.app:app --reload
```
The API will be available at `http://localhost:8000`

**2. Launch the Streamlit UI:**
```bash
streamlit run frontend/streamlit_app.py
```
The UI will open automatically in your browser at `http://localhost:8501`

### First Steps

1. **Upload documents** via the Streamlit UI
2. **Run ingestion** to process and embed your documents
3. **Query your RAG system** and get grounded answers

---

## 🎯 Design Philosophy

This project follows production-first principles:

1. **Retrieval quality > generation** - 80% of RAG success comes from retrieval
2. **No hallucinations** - All answers grounded in retrieved context
3. **Defensive execution** - Graceful error handling at every layer
4. **Observable pipelines** - Comprehensive logging and timing
5. **Modular & testable** - Clean separation of concerns
6. **Production-first mindset** - Built to scale, not just demo

> "RAG is 80% retrieval, 20% generation. Get the retrieval right first."

---

## 🔍 Key Components Explained

### Chunking Strategy

**Recursive Chunking:**
- Splits documents based on structural elements (paragraphs, sentences)
- Preserves document hierarchy
- Maintains context boundaries

**Semantic Chunking:**
- Groups text based on meaning similarity
- Creates coherent, topically-focused chunks
- Reduces context fragmentation

### Retrieval Process

1. **Query Embedding** - Convert user query to 384-d vector
2. **Similarity Search** - Find top-K most similar chunks using cosine similarity
3. **Optional Reranking** - Re-score results for better relevance
4. **Context Assembly** - Combine retrieved chunks into coherent context

### LLM Integration

- **Local inference** via Ollama for privacy and control
- **Prompt engineering** to prevent hallucinations
- **Context grounding** to ensure factual responses
- **Streaming support** for better user experience

---

## 📈 Performance Considerations

**Typical Query Performance:**
- Retrieval: 0.1 - 0.5 seconds
- LLM Generation: 5 - 60 seconds (depending on model and hardware)
- Total: 5 - 60 seconds end-to-end

**Optimization Tips:**
- Use smaller Ollama models (e.g., `mistral`, `phi`) for faster inference
- Adjust `top_k` parameter to balance quality and speed
- Consider GPU acceleration for embedding generation
- Implement caching for frequently asked queries

---

## 🧪 Testing & Validation

**System Health Checks:**
```bash
# Check if vector store is accessible
python rag_project/checker_files.py
```

**API Health:**
```bash
curl http://localhost:8000/health
```

**Query Testing:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?"}'
```

---

## 🛣️ Roadmap

**Planned Enhancements:**
- [ ] Support for more document types (DOCX, CSV, JSON)
- [ ] Advanced reranking models (cross-encoders)
- [ ] Query expansion and reformulation
- [ ] Multi-query retrieval
- [ ] Hybrid search (vector + keyword)
- [ ] User feedback loop for retrieval quality
- [ ] Docker containerization
- [ ] Deployment guides (AWS, GCP, Azure)

---

## 🐛 Troubleshooting

**Common Issues:**

**Issue:** `ModuleNotFoundError: No module named 'chromadb'`
- **Solution:** Run `pip install -r requirements.txt`

**Issue:** `Connection refused to Ollama`
- **Solution:** Ensure Ollama is running: `ollama serve`

**Issue:** `No documents in vector store`
- **Solution:** Upload and ingest documents via the Streamlit UI

**Issue:** Slow query responses
- **Solution:** Use a smaller Ollama model or enable GPU acceleration

For more help, check the logs in the `logs/` directory.

---

## 📊 System Status

| Component | Status |
|-----------|--------|
| Ingestion Pipeline | ✅ Complete |
| Vector Database | ✅ Complete |
| Retrieval System | ✅ Complete |
| LLM Integration | ✅ Complete |
| FastAPI Backend | ✅ Complete |
| Streamlit UI | ✅ Complete |
| Logging & Monitoring | ✅ Complete |

**Overall Status:** ✅ System is stable and production-ready

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Code Standards:**
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include type hints
- Write tests for new features
- Update documentation as needed

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

This project builds upon excellent open-source tools and libraries:

- **[LangChain](https://github.com/langchain-ai/langchain)** - Document loaders and text splitting
- **[ChromaDB](https://github.com/chroma-core/chroma)** - Vector database
- **[Sentence Transformers](https://github.com/UKPLab/sentence-transformers)** - Embedding models
- **[Ollama](https://ollama.ai/)** - Local LLM inference
- **[HuggingFace](https://huggingface.co/)** - Model hosting and distribution
- **[FastAPI](https://fastapi.tiangolo.com/)** - Modern API framework
- **[Streamlit](https://streamlit.io/)** - UI framework

Special thanks to the open-source community for making production-grade RAG systems accessible to everyone.

---

## 📧 Contact & Support

- **Issues:** [GitHub Issues](https://github.com/your-username/intermediate-rag/issues)
- **Discussions:** [GitHub Discussions](https://github.com/your-username/intermediate-rag/discussions)
- **Email:** your-email@example.com

---

<div align="center">

**Built with ❤️ by GRAVITY-AI**

⭐ **Star this repo if it helped you!** ⭐

[Documentation](https://github.com/your-username/intermediate-rag/wiki) • [Report Bug](https://github.com/your-username/intermediate-rag/issues) • [Request Feature](https://github.com/your-username/intermediate-rag/issues)

</div>