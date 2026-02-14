# Intermediate RAG System

A production-oriented Retrieval-Augmented Generation (RAG) system built with open-source tools, designed for correctness, observability, and real-world deployment.

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-production--ready-success.svg)]()

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Performance Analysis](#performance-analysis)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project implements a complete, end-to-end RAG system that combines document retrieval with large language model generation to provide accurate, context-grounded answers. Unlike tutorial projects, this system is built with production principles in mind, featuring strict database lifecycle control, async processing, comprehensive logging, and configurable behavior.

### What Makes This Production-Ready

- **Strict Database Lifecycle Control**: No silent database creation, explicit separation between ingestion and query modes
- **Async FastAPI Backend**: Concurrent request handling with semaphore-based throttling
- **Comprehensive Observability**: Full logging and timing metrics for all operations
- **Configuration-Driven**: Central YAML configuration for all system parameters
- **Defensive Error Handling**: Graceful failures at every layer
- **Performance Monitoring**: Detailed timing breakdowns identifying bottlenecks

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                          │
│                    (Streamlit / API Client)                     │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Ingestion  │  │    Query     │  │    Health    │         │
│  │   Endpoint   │  │   Endpoint   │  │    Check     │         │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘         │
└─────────┼──────────────────┼──────────────────────────────────┘
          │                  │
          ▼                  ▼
┌─────────────────┐  ┌──────────────────────────────────────────┐
│   Ingestion     │  │        Query Pipeline                    │
│   Pipeline      │  │  ┌────────────┐  ┌─────────────┐        │
│  ┌──────────┐   │  │  │  Retriever │  │  Reranker   │        │
│  │ Loaders  │   │  │  │  (Top-K)   │  │ (Optional)  │        │
│  └────┬─────┘   │  │  └─────┬──────┘  └──────┬──────┘        │
│  ┌────▼─────┐   │  │        │                 │               │
│  │ Chunking │   │  │        └────────┬────────┘               │
│  └────┬─────┘   │  │                 ▼                        │
│  ┌────▼─────┐   │  │        ┌────────────────┐               │
│  │Embeddings│   │  │        │ Context Builder│               │
│  └────┬─────┘   │  │        └────────┬───────┘               │
└───────┼─────────┘  └─────────────────┼───────────────────────┘
        │                               │
        ▼                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ChromaDB Vector Store                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Collection: ashwa_rag                                    │  │
│  │  - Document chunks with embeddings                        │  │
│  │  - Metadata (source, chunk_id, etc.)                      │  │
│  │  - HNSW index for fast similarity search                  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────┐
                    │   Ollama LLM     │
                    │  (Local Model)   │
                    └──────────────────┘
```

### Ingestion Flow

```
Document Upload
      │
      ▼
┌─────────────┐
│   Loaders   │  PDF/TXT document loading
│ (LangChain) │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│   Recursive     │  Structure-aware text splitting
│   Chunking      │  - Chunk size: 1000 chars
└──────┬──────────┘  - Overlap: 200 chars
       │
       ▼
┌─────────────────┐
│   Semantic      │  Meaning-aware regrouping
│   Chunking      │  - Groups similar chunks
└──────┬──────────┘  - Preserves context
       │
       ▼
┌─────────────────┐
│   Embedding     │  sentence-transformers/all-MiniLM-L6-v2
│   Generation    │  - 384-dimensional vectors
└──────┬──────────┘  - Batch size: 32
       │
       ▼
┌─────────────────┐
│   ChromaDB      │  Persistent vector storage
│   Storage       │  - Cosine similarity
└─────────────────┘  - HNSW indexing
```

### Query Flow

```
User Query
    │
    ▼
┌──────────────────┐
│ Query Validation │  Min length check, sanitization
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│     Database     │  Verify DB exists (if required)
│  Existence Check │  Prevent auto-creation
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│      Query       │  Convert query to 384-d vector
│    Embedding     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Vector Search   │  Cosine similarity search
│    (Top-K)       │  - Fetch K candidates: 10
└────────┬─────────┘  - Return top K: 3
         │
         ▼
┌──────────────────┐
│    Reranker      │  Optional cross-encoder reranking
│   (Optional)     │  - Model: ms-marco-MiniLM-L-6-v2
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│     Context      │  Assemble retrieved chunks
│   Construction   │  - Truncate if needed (4000 chars)
└────────┬─────────┘  - Format with metadata
         │
         ▼
┌──────────────────┐
│   LLM Prompt     │  Structured prompt with context
│   Construction   │  - System message
└────────┬─────────┘  - User query + context
         │
         ▼
┌──────────────────┐
│  Ollama LLM      │  Local inference
│   Generation     │  - Temperature: 0.2
└────────┬─────────┘  - Max tokens: 256
         │
         ▼
┌──────────────────┐
│    Response      │  JSON response with:
│   Formatting     │  - Answer text
└──────────────────┘  - Source chunks
                      - Timing metrics
```

### Concurrency Control

```
┌───────────────────────────────────────────────────────┐
│              FastAPI Async Handler                    │
└────────────────────┬──────────────────────────────────┘
                     │
                     ▼
              ┌──────────────┐
              │  Semaphore   │  Max concurrent: 3
              │  (Optional)  │
              └──────┬───────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   Request 1  │ │   Request 2  │ │   Request 3  │
│  (Processing)│ │  (Processing)│ │  (Processing)│
└──────────────┘ └──────────────┘ └──────────────┘
        │            │            │
        └────────────┼────────────┘
                     │
                     ▼
              ┌──────────────┐
              │  Request 4   │
              │   (Waiting)  │
              └──────────────┘
```

---

## Key Features

### Core Capabilities

- **Document Ingestion**: Support for PDF and TXT files with automatic preprocessing
- **Hybrid Chunking**: Combines recursive and semantic chunking for optimal context preservation
- **Vector Embeddings**: Uses sentence-transformers for high-quality semantic representations
- **Persistent Storage**: ChromaDB with disk persistence and HNSW indexing
- **Advanced Retrieval**: Top-K similarity search with optional cross-encoder reranking
- **Local LLM**: Privacy-first answer generation using Ollama
- **RESTful API**: FastAPI backend with async processing and rate limiting
- **Interactive UI**: Streamlit-based control panel for document management and queries
- **Comprehensive Logging**: Production-grade observability with timing metrics

### Production Features

- **Strict Database Control**: Prevents accidental database creation in query mode
- **Lifecycle Management**: Clear separation between ingestion and query operations
- **Async Processing**: Non-blocking request handling with configurable concurrency limits
- **Request Timeout Protection**: Prevents hung requests from blocking the system
- **Auto-cleanup Mode**: Optional database clearing on startup for testing
- **Health Monitoring**: System checker validates database state and component availability
- **Context Truncation**: Automatic context limiting to prevent token overflow
- **Defensive Error Handling**: Graceful degradation with informative error messages

---

## Technology Stack

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Runtime** | Python | 3.12+ | Core language |
| **Embeddings** | Sentence Transformers | Latest | Text-to-vector conversion |
| **Vector DB** | ChromaDB | Latest | Similarity search and storage |
| **LLM** | Ollama | Latest | Local answer generation |
| **API Framework** | FastAPI | Latest | REST API backend |
| **UI Framework** | Streamlit | Latest | Web interface |
| **Document Processing** | LangChain | Latest | Loaders and chunking |
| **Configuration** | PyYAML | Latest | YAML config parsing |
| **Logging** | Python logging | Built-in | System observability |

### Model Specifications

#### Embedding Model: sentence-transformers/all-MiniLM-L6-v2

- **Dimensions**: 384
- **Max Sequence Length**: 256 tokens
- **Speed**: Excellent (suitable for real-time applications)
- **Memory**: Low footprint (~80MB)
- **Accuracy**: Strong performance on semantic similarity tasks
- **Use Case**: General-purpose semantic search

#### Reranking Model: cross-encoder/ms-marco-MiniLM-L-6-v2

- **Type**: Cross-encoder for passage reranking
- **Training**: MS MARCO dataset
- **Purpose**: Refine initial retrieval results
- **Activation**: Optional (configurable)

---

## Project Structure

```
INTERMEDIATE_RAG/
│
├── frontend/
│   └── streamlit_app.py              # Streamlit UI application
│
├── rag_project/
│   ├── api/
│   │   └── app.py                    # FastAPI application
│   │
│   ├── checker_files.py              # System health checker
│   │
│   ├── scripts/
│   │   ├── ingestion.py              # Document ingestion pipeline
│   │   └── query_rag.py              # Query processing pipeline
│   │
│   └── rag/
│       ├── loaders/
│       │   ├── pdf_loader.py         # PDF document loader
│       │   └── txt_loader.py         # TXT document loader
│       │
│       ├── chunking/
│       │   ├── recursive_chunker.py  # Structure-based chunking
│       │   └── semantic_chunker.py   # Meaning-based chunking
│       │
│       ├── embeddings/
│       │   └── embedder.py           # Embedding generation
│       │
│       ├── chromaDB/
│       │   └── chroma_store.py       # Vector store management
│       │
│       ├── retriever_files/
│       │   ├── retriever.py          # Document retrieval
│       │   └── reranker.py           # Result reranking
│       │
│       ├── prompts/
│       │   └── templates.py          # LLM prompt templates
│       │
│       └── llm/
│           └── ollama_client.py      # Ollama integration
│
├── utils/
│   ├── logger.py                     # Centralized logging
│   └── timer.py                      # Execution timing decorator
│
├── logs/                             # Runtime logs (gitignored)
│   ├── query.log                     # Query processing logs
│   ├── ingestion.log                 # Ingestion pipeline logs
│   ├── system.log                    # System health logs
│   └── app.log                       # API lifecycle logs
│
├── data/
│   └── raw/                          # Uploaded documents
│
├── vector_store/
│   └── chroma/                       # Persistent ChromaDB data
│
├── config.yaml                       # Central configuration
├── requirements.txt                  # Python dependencies
├── main.py                           # System launcher
├── README.md                         # This file
└── .gitignore                        # Git ignore rules
```

---

## Configuration

The system is controlled through `config.yaml`, which provides centralized configuration for all components.

### Configuration Structure

```yaml
project:
  name: "intermediate-rag"
  stage: "learning"           # learning | production
  entrypoint: "main.py"

paths:
  data:
    raw_dir: "rag_project/data/raw"
  vector_store:
    chroma_dir: "rag_project/vector_store/chroma"

chunking:
  recursive:
    chunk_size: 1000          # Characters per chunk
    overlap: 200              # Character overlap between chunks
  semantic:
    enabled: true             # Enable semantic regrouping
    method: "langchain"       # Implementation strategy

embeddings:
  provider: "huggingface"
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  normalize: true             # Normalize vectors for cosine similarity
  batch_size: 32              # Embedding batch size

chroma:
  collection_name: "ashwa_rag"
  persist: true               # Enable disk persistence
  distance_metric: "cosine"   # Similarity metric
  index_type: "hnsw"          # Approximate nearest neighbor index

retrieval:
  top_k: 3                    # Final chunks returned
  fetch_k: 10                 # Initial candidate pool
  use_reranker: true          # Enable reranking

reranker:
  enabled: false              # Activate cross-encoder reranking
  model_name: "cross-encoder/ms-marco-MiniLM-L-6-v2"

llm:
  provider: "ollama"
  model: "phi"                # Ollama model name
  temperature: 0.2            # Randomness (0.0 = deterministic)
  max_tokens: 256             # Maximum output length
  top_p: 0.9                  # Nucleus sampling
  repeat_penalty: 1.1         # Repetition penalty
  stream: false               # Enable streaming responses
  context_truncate: true      # Truncate long contexts
  max_context_chars: 4000     # Context size limit

database:
  auto_clear_on_start: false  # Wipe DB on startup (testing mode)
  require_existing_db: true   # Prevent auto-creation in query mode

api:
  host: "0.0.0.0"
  port: 8000
  max_concurrent_requests: 3  # LLM concurrency limit
  enable_semaphore: true      # Enable throttling
  workers: 2                  # Uvicorn worker processes
  min_query_length: 5         # Minimum query length
  request_timeout: 300        # Request timeout (seconds)

logging:
  level: "INFO"               # DEBUG | INFO | WARNING | ERROR
  log_to_file: false          # Enable file logging
  log_file: "logs/rag.log"    # Log file path
```

### Key Configuration Parameters

#### Database Control

- **auto_clear_on_start**: When `true`, deletes the entire vector store on system startup. Useful for testing and development. Should be `false` in production.

- **require_existing_db**: When `true`, query operations fail if the database doesn't exist, preventing accidental database creation. Enforces ingestion-first workflow.

#### Concurrency Management

- **max_concurrent_requests**: Limits simultaneous LLM generation requests to prevent resource exhaustion. Recommended: 3-5 for consumer hardware.

- **enable_semaphore**: Activates request throttling. When disabled, all requests are processed concurrently (not recommended for production).

#### Context Management

- **context_truncate**: Automatically truncates assembled context to `max_context_chars` to prevent token overflow.

- **max_context_chars**: Maximum characters allowed in the context sent to the LLM. Adjust based on your model's context window.

---

## Installation

### Prerequisites

- **Python 3.12 or higher**: [Download Python](https://www.python.org/downloads/)
- **Ollama**: [Installation Guide](https://ollama.ai/)
- **pip**: Python package manager (included with Python)
- **Git**: Version control system

### Step-by-Step Installation

#### 1. Clone the Repository

```bash
git clone <repository-url>
cd INTERMEDIATE_RAG
```

#### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

#### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- FastAPI and Uvicorn (API framework)
- Streamlit (UI framework)
- LangChain (document processing)
- ChromaDB (vector database)
- Sentence Transformers (embeddings)
- PyYAML (configuration)
- Additional utilities

#### 4. Install and Configure Ollama

Download and install Ollama from [ollama.ai](https://ollama.ai/), then pull a model:

```bash
# Recommended lightweight model
ollama pull phi

# Alternative models
ollama pull mistral
ollama pull llama2
```

Update `config.yaml` to match your chosen model:

```yaml
llm:
  model: "phi"  # or "mistral", "llama2", etc.
```

#### 5. Verify Installation

```bash
# Check Python version
python --version  # Should be 3.12+

# Check Ollama
ollama list  # Should show installed models

# Run system checker
python rag_project/checker_files.py
```

---

## Usage

### Quick Start

The system provides a unified launcher (`main.py`) that orchestrates all components:

```bash
python main.py
```

This will:
1. Run system health checks
2. Optionally clear the database (if configured)
3. Launch the FastAPI backend
4. Launch the Streamlit UI
5. Open the interface in your browser

### Manual Component Launch

#### Start FastAPI Backend

```bash
uvicorn rag_project.api.app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

API documentation: `http://localhost:8000/docs`

#### Start Streamlit UI

```bash
streamlit run frontend/streamlit_app.py
```

The UI will open automatically at `http://localhost:8501`

### Workflow

#### 1. Document Ingestion

**Via Streamlit UI:**

1. Navigate to the "Document Upload" section
2. Upload PDF or TXT files
3. Click "Run Ingestion" to process documents
4. Monitor ingestion progress in the logs

**Via API:**

```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{"document_path": "path/to/document.pdf"}'
```

#### 2. Query the System

**Via Streamlit UI:**

1. Navigate to the "Query" section
2. Enter your question
3. View the generated answer and source documents
4. Check timing metrics

**Via API:**

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is retrieval-augmented generation?"}'
```

#### 3. System Health Check

```bash
# Via API
curl http://localhost:8000/health

# Via CLI
python rag_project/checker_files.py
```

---

## API Reference

### Base URL

```
http://localhost:8000
```

### Endpoints

#### POST /query

Process a RAG query and return a grounded answer.

**Request Body:**

```json
{
  "query": "What is RAG?"
}
```

**Response:**

```json
{
  "answer": "Retrieval-Augmented Generation (RAG) is...",
  "sources": [
    {
      "content": "RAG combines retrieval and generation...",
      "metadata": {
        "source": "document.pdf",
        "chunk_id": "chunk_1"
      }
    }
  ],
  "retrieval_time": 0.22,
  "generation_time": 31.13
}
```

**Status Codes:**

- `200`: Success
- `400`: Invalid query (too short, empty)
- `500`: Server error (DB missing, LLM failure)

#### GET /health

Check system health and database status.

**Response:**

```json
{
  "status": "healthy",
  "database": "available",
  "timestamp": "2025-02-15T10:30:00Z"
}
```

#### GET /docs

Interactive API documentation (Swagger UI).

---

## Performance Analysis

### Timing Breakdown

Based on production logs, typical query performance:

| Operation | Time Range | Percentage |
|-----------|-----------|------------|
| Query Validation | 0.001s - 0.005s | <0.1% |
| Vector Retrieval | 0.03s - 0.13s | <1% |
| Reranking (optional) | 0.05s - 0.15s | <1% |
| Context Assembly | 0.01s - 0.05s | <0.1% |
| LLM Generation | 15s - 45s | >98% |

**Key Insights:**

- The LLM is responsible for 98%+ of query latency
- Vector retrieval is extremely fast (under 200ms)
- ChromaDB is not a bottleneck
- Optimization should focus on LLM inference speed

### Optimization Strategies

#### Reduce LLM Latency

1. **Use smaller models**: `phi` < `mistral` < `llama2`
2. **Enable GPU acceleration**: Configure Ollama for CUDA
3. **Reduce max_tokens**: Lower `max_tokens` in config
4. **Enable streaming**: Set `stream: true` for perceived speed

#### Improve Retrieval Quality

1. **Tune chunk size**: Experiment with `chunk_size` and `overlap`
2. **Enable reranking**: Set `reranker.enabled: true`
3. **Adjust top_k**: Balance between quality and context size
4. **Use semantic chunking**: Ensure `semantic.enabled: true`

#### Scale Concurrency

1. **Increase workers**: Raise `api.workers` for more parallelism
2. **Adjust semaphore**: Increase `max_concurrent_requests` for higher hardware
3. **Enable caching**: Implement query result caching (future feature)

---

## Troubleshooting

### Common Issues

#### Database Not Found

**Symptom:** `DatabaseError: Vector store does not exist`

**Solution:**
1. Ensure documents have been ingested
2. Check `database.auto_clear_on_start` is `false`
3. Verify `vector_store/chroma/` directory exists

#### Ollama Connection Failed

**Symptom:** `ConnectionError: Failed to connect to Ollama`

**Solution:**
```bash
# Start Ollama service
ollama serve

# Verify model is installed
ollama list

# Test model
ollama run phi "Hello"
```

#### Slow Query Performance

**Symptom:** Queries take 30+ seconds

**Solution:**
1. Use a smaller model: `phi` or `mistral`
2. Reduce `max_tokens` in config
3. Enable GPU acceleration for Ollama
4. Check system resources (CPU/RAM)

#### Empty Retrieval Results

**Symptom:** No documents retrieved for query

**Solution:**
1. Verify database contains documents
2. Check embedding model matches between ingestion and query
3. Adjust `top_k` or `fetch_k` values
4. Review query phrasing

#### Module Import Errors

**Symptom:** `ModuleNotFoundError: No module named 'X'`

**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Verify virtual environment is activated
which python  # Should point to venv
```

### Debug Mode

Enable debug logging for detailed diagnostics:

```yaml
logging:
  level: "DEBUG"
  log_to_file: true
```

Check logs in the `logs/` directory:
- `query.log`: Query processing details
- `ingestion.log`: Document processing details
- `system.log`: Health check results
- `app.log`: API lifecycle events

---

## Roadmap

### Planned Enhancements

**Phase 1: Extended Document Support**
- DOCX file support
- CSV/Excel file support
- JSON document parsing
- HTML web page ingestion

**Phase 2: Advanced Retrieval**
- Hybrid search (vector + keyword)
- Multi-query retrieval
- Query expansion and reformulation
- Parent document retrieval

**Phase 3: Quality Improvements**
- User feedback loop for retrieval quality
- Active learning for embedding fine-tuning
- Advanced reranking models
- Answer quality scoring

**Phase 4: Production Readiness**
- Docker containerization
- Kubernetes deployment manifests
- Cloud deployment guides (AWS, GCP, Azure)
- Monitoring and alerting integration
- Multi-tenancy support

**Phase 5: Advanced Features**
- Conversational memory
- Multi-turn dialogue support
- Citation tracking and verification
- Graph-based document relationships

---

## Contributing

Contributions are welcome. Please follow these guidelines:

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Make your changes
5. Run tests: `pytest tests/`
6. Commit changes: `git commit -m 'Add feature X'`
7. Push to branch: `git push origin feature/your-feature`
8. Open a Pull Request

### Code Standards

- Follow PEP 8 style guidelines
- Add type hints to all function signatures
- Include docstrings for all public methods
- Write unit tests for new features
- Update documentation as needed
- Keep commits atomic and well-described

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=rag_project

# Run specific test file
pytest tests/test_retrieval.py
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

This project builds upon excellent open-source tools:

- **LangChain**: Document loaders and text processing
- **ChromaDB**: Vector database infrastructure
- **Sentence Transformers**: High-quality embedding models
- **Ollama**: Local LLM inference platform
- **HuggingFace**: Model hosting and ecosystem
- **FastAPI**: Modern Python web framework
- **Streamlit**: Rapid UI development framework

---

## Contact

- **Issues**: [GitHub Issues](https://github.com/your-username/intermediate-rag/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/intermediate-rag/discussions)
- **Documentation**: [Wiki](https://github.com/your-username/intermediate-rag/wiki)

---

**Built by GRAVITY-AI**

For questions, feature requests, or bug reports, please use the GitHub issue tracker.

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
- **Email:** jyotiradityaparihar@gmail.com

---

<div align="center">

**Built with ❤️ by GRAVITY-AI**

⭐ **Star this repo if it helped you!** ⭐

[Documentation](https://github.com/your-username/intermediate-rag/wiki) • [Report Bug](https://github.com/your-username/intermediate-rag/issues) • [Request Feature](https://github.com/your-username/intermediate-rag/issues)

</div>