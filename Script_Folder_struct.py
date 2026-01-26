import os

BASE_DIR = "rag_project"

# Folder structure
folders = [
    "rag/loaders",
    "rag/chunking",
    "rag/embeddings",
    "rag/vectorstore",
    "rag/retriever",
    "rag/llm",
    "rag/prompts",

    "db",
    "data/raw",
    "data/processed",

    "scripts",

    "utils",

]

# Python files to create (relative to BASE_DIR)
python_files = [
    # Core package
    "rag/__init__.py",

    "rag/loaders/__init__.py",
    "rag/loaders/text_loader.py",
    "rag/loaders/pdf_loader.py",

    "rag/chunking/__init__.py",
    "rag/chunking/recursive.py",
    "rag/chunking/semantic.py",

    "rag/embeddings/__init__.py",
    "rag/embeddings/hf_embeddings.py",

    "rag/vectorstore/__init__.py",
    "rag/vectorstore/chroma_store.py",

    "rag/retriever/__init__.py",
    "rag/retriever/retriever.py",
    "rag/retriever/reranker.py",

    "rag/llm/__init__.py",
    "rag/llm/ollama_llm.py",

    # Scripts
    "scripts/ingestion.py",
    "scripts/query_rag.py",


    # Utilities
    "utils/logger.py",
]

def create_folders(base_dir, folders):
    for folder in folders:
        path = os.path.join(base_dir, folder)
        os.makedirs(path, exist_ok=True)
        print(f"📁 Created folder: {path}")

def create_python_files(base_dir, files):
    for file in files:
        path = os.path.join(base_dir, file)
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write("")  # empty starter file
            print(f"🐍 Created file: {path}")
        else:
            print(f"⚠️  File already exists, skipped: {path}")

if __name__ == "__main__":
    create_folders(BASE_DIR, folders)
    create_python_files(BASE_DIR, python_files)
    print("\n✅ RAG project scaffold created successfully.")
