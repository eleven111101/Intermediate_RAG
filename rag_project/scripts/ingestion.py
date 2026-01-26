from pathlib import Path
import uuid

from rag_project.pipeline_runner import run_pipeline
from rag_project.rag.chromaDB.chroma_store import ChromaStore


def ingest():
    print("\n=== INGESTION STARTED ===\n")

    project_root = Path(__file__).resolve().parents[1]
    persist_dir = project_root / "rag_project/vector_store/chroma"

    # -----------------------------
    # Run pipeline (till embeddings)
    # -----------------------------
    semantic_chunks, embeddings = run_pipeline()

    # -----------------------------
    # Prepare data for ChromaDB
    # -----------------------------
    ids = [str(uuid.uuid4()) for _ in semantic_chunks]
    documents = [doc.page_content for doc in semantic_chunks]
    metadatas = [doc.metadata for doc in semantic_chunks]

    # -----------------------------
    # Store in ChromaDB
    # -----------------------------
    store = ChromaStore(
        persist_dir=persist_dir,
        collection_name="ashwa_rag",
    )

    store.add_documents(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    store.persist()

    print(f"✔ Stored {len(ids)} chunks in ChromaDB")
    print("\n=== INGESTION COMPLETED ===\n")


if __name__ == "__main__":
    ingest()
