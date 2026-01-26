from pathlib import Path
import uuid
import yaml

from rag_project.rag.loaders.data_loader import DataLoader
from rag_project.rag.chunking.recursive_chunking import RecursiveChunker
from rag_project.rag.chunking.semantic_chunking import SemanticChunkerWrapper
from rag_project.rag.embeddings.embedding_service import EmbeddingService
from rag_project.rag.vector_store.chroma_store import ChromaStore


def load_config():
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_pipeline():
    print("\n==============================")
    print(" RAG INGESTION PIPELINE STARTED")
    print("==============================\n")

    project_root = Path(__file__).resolve().parents[1]
    config = load_config()

    # --------------------------------------------------
    # PATHS
    # --------------------------------------------------
    raw_dir = project_root / config["paths"]["data"]["raw_dir"]
    chroma_dir = project_root / config["paths"]["vector_store"]["chroma_dir"]

    # --------------------------------------------------
    # STEP 1: LOAD DATA
    # --------------------------------------------------
    print("[STEP 1] Loading raw documents")
    loader = DataLoader(raw_dir)
    documents = loader.load()
    print(f"→ Loaded documents: {len(documents)}\n")

    if not documents:
        print("❌ No documents found. Pipeline stopped.")
        return

    # --------------------------------------------------
    # STEP 2: RECURSIVE CHUNKING
    # --------------------------------------------------
    print("[STEP 2] Recursive chunking")
    rcfg = config["chunking"]["recursive"]

    recursive_chunker = RecursiveChunker(
        chunk_size=rcfg["chunk_size"],
        overlap=rcfg["overlap"],
    )
    recursive_chunks = recursive_chunker.chunk(documents)
    print(f"→ Recursive chunks: {len(recursive_chunks)}\n")

    # --------------------------------------------------
    # STEP 3: EMBEDDINGS (TEMP)
    # --------------------------------------------------
    print("[STEP 3] Generating temporary embeddings")
    embedder = EmbeddingService()
    _ = embedder.embed_documents(recursive_chunks)
    print("→ Temporary embeddings generated\n")

    # --------------------------------------------------
    # STEP 4: SEMANTIC CHUNKING
    # --------------------------------------------------
    print("[STEP 4] Semantic chunking")
    semantic_chunks = recursive_chunks

    if config["chunking"]["semantic"]["enabled"]:
        semantic_chunker = SemanticChunkerWrapper(embedder.model)
        semantic_chunks = semantic_chunker.chunk(recursive_chunks)

    print(f"→ Semantic chunks: {len(semantic_chunks)}\n")

    # --------------------------------------------------
    # STEP 5: FINAL EMBEDDINGS
    # --------------------------------------------------
    print("[STEP 5] Generating final embeddings")
    final_embeddings = embedder.embed_documents(semantic_chunks)
    print(f"→ Final embeddings: {len(final_embeddings)}\n")

    # --------------------------------------------------
    # STEP 6: STORE IN CHROMADB
    # --------------------------------------------------
    print("[STEP 6] Storing in ChromaDB")

    store = ChromaStore(
        persist_dir=chroma_dir,
        collection_name=config["chroma"]["collection_name"],
    )

    ids = [str(uuid.uuid4()) for _ in semantic_chunks]
    documents_text = [doc.page_content for doc in semantic_chunks]
    metadatas = [doc.metadata for doc in semantic_chunks]

    store.add(
        ids=ids,
        documents=documents_text,
        embeddings=final_embeddings,
        metadatas=metadatas,
    )

    print("→ ChromaDB auto-persisted data to disk\n")


    print(f"→ Stored {len(ids)} chunks in ChromaDB\n")

    print("==============================")
    print(" INGESTION PIPELINE COMPLETED")
    print("==============================\n")

    print("SUMMARY")
    print(f"• Raw documents    : {len(documents)}")
    print(f"• Recursive chunks : {len(recursive_chunks)}")
    print(f"• Semantic chunks  : {len(semantic_chunks)}")
    print(f"• Stored vectors   : {len(final_embeddings)}\n")


if __name__ == "__main__":
    run_pipeline()
