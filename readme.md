# Intermediate RAG System

A production-oriented Retrieval-Augmented Generation (RAG) pipeline built with open-source tools. This system prioritizes correctness, debuggability, and scalability over quick demos.

## Overview

This project implements a complete RAG system that:
- Ingests and processes documents (TXT, PDF)
- Generates semantic embeddings using sentence transformers
- Performs vector similarity search with ChromaDB
- Generates grounded answers using local LLMs via Ollama
- Maintains metadata and operational logs in a traditional database

## Architecture

```
┌─────────────┐
│ User Query  │
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│ Query Embedding  │
└──────┬───────────┘
       │
       ▼
┌─────────────────────────┐
│ ChromaDB Vector Search  │
└──────┬──────────────────┘
       │
       ▼
┌──────────────────────┐
│ Top-K Relevant Chunks│
└──────┬───────────────┘
       │
       ▼
┌────────────────────┐
│ Context Formatting │
└──────┬─────────────┘
       │
       ▼
┌──────────────────┐
│ Local LLM (Ollama)│
└──────┬───────────┘
       │
       ▼
┌─────────────────────────┐
│ Grounded Answer + Sources│
└─────────────────────────┘
```

## Project Structure

```
rag_project/
├── rag/                    # Core RAG pipeline logic
├── db/                     # Database layer (metadata, logs)
├── data/
│   ├── raw/               # Source documents (gitignored)
│   └── processed/         # Cleaned text for indexing
├── notebooks/             # Jupyter notebooks for experiments
├── scripts/               # CLI tools and batch processing
├── pyproject.toml         # Project metadata and dependencies
├── requirements.txt       # Pinned dependencies
├── .gitignore
└── README.md
```

## Technology Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.12 |
| **Framework** | LangChain |
| **Vector Store** | ChromaDB |
| **Embeddings** | Sentence Transformers (all-MiniLM-L6-v2) |
| **LLM** | Ollama (local inference) |
| **Database** | SQLite / PostgreSQL |
| **Package Manager** | uv |

## Embedding Model: all-MiniLM-L6-v2

We use `sentence-transformers/all-MiniLM-L6-v2` which produces **384-dimensional embeddings**:

- Each text chunk is converted to a 384-dimensional vector
- Each dimension captures a fragment of semantic meaning
- Similar meanings produce vectors that are close in vector space
- No single dimension is interpretable—**meaning emerges from all 384 dimensions combined**

This model balances:
- ✅ Speed (fast inference)
- ✅ Size (compact vectors)
- ✅ Quality (good semantic understanding)

## Design Philosophy

1. **Retrieval Quality First**: The quality of retrieved context matters more than LLM sophistication
2. **Honest Uncertainty**: System rejects queries when relevant knowledge is missing
3. **No Hallucinations**: Answers must be grounded in retrieved documents
4. **Progressive Complexity**: Simple → Correct → Scalable

> **Key Principle**: RAG is 80% retrieval and data quality, 20% generation.

## Getting Started

### Prerequisites

- Python 3.12+
- [Ollama](https://ollama.ai/) installed and running
- `uv` package manager (optional but recommended)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd rag_project

# Install dependencies
pip install -r requirements.txt

# Or using uv
uv pip install -r requirements.txt
```

### Quick Start

```bash
# 1. Place your documents in data/raw/

# 2. Process documents
python scripts/ingest.py

# 3. Build vector index
python scripts/build_index.py

# 4. Run queries
python scripts/query.py "Your question here"
```

## Usage Examples

### Basic Query

```python
from rag import RAGPipeline

# Initialize the pipeline
pipeline = RAGPipeline()

# Query the system
result = pipeline.query("What is the main topic discussed in the documents?")

print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
```

### Advanced Configuration

```python
from rag import RAGPipeline

pipeline = RAGPipeline(
    chunk_size=512,
    chunk_overlap=50,
    top_k=5,
    model_name="llama2"
)

result = pipeline.query(
    query="Your question",
    temperature=0.7,
    max_tokens=500
)
```

## Development Status

| Feature | Status |
|---------|--------|
| Document ingestion | 🚧 In Progress |
| Text chunking | 📋 Planned |
| Embedding generation | 📋 Planned |
| Vector storage (ChromaDB) | 📋 Planned |
| Similarity search | 📋 Planned |
| LLM integration | 📋 Planned |
| Reranking | 📋 Planned |
| Evaluation framework | 📋 Planned |
| Query optimization | 📋 Planned |

## Roadmap

### Phase 1: Core Pipeline ✅
- [ ] Document ingestion
- [ ] Vector embeddings
- [ ] Similarity search

### Phase 2: LLM Integration 🚧
- [ ] Ollama integration
- [ ] Prompt engineering
- [ ] Answer generation

### Phase 3: Enhancement 📋
- [ ] Reranking layer
- [ ] Hybrid search (vector + keyword)
- [ ] Query expansion

### Phase 4: Production 📋
- [ ] Evaluation metrics
- [ ] Performance monitoring
- [ ] API deployment

## Configuration

Edit `config.yaml` or set environment variables:

```yaml
# Vector Store
chroma:
  persist_directory: "./data/chroma"
  collection_name: "documents"

# Embeddings
embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cpu"

# Chunking
chunking:
  chunk_size: 512
  chunk_overlap: 50

# LLM
llm:
  provider: "ollama"
  model: "llama2"
  temperature: 0.7
  max_tokens: 500
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=rag tests/

# Run specific test module
pytest tests/test_retrieval.py
```

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- LangChain for the RAG framework
- ChromaDB for vector storage
- Sentence Transformers for embeddings
- Ollama for local LLM inference

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Built with ❤️ for production-grade RAG systems**