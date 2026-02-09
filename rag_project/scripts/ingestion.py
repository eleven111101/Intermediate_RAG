import yaml
from pathlib import Path

from utils.logger import setup_logger
from utils.timer import timed_block

from rag_project.rag.loaders.data_loader import DataLoader
from rag_project.rag.chunking.recursive_chunking import RecursiveChunker
from rag_project.rag.chunking.semantic_chunking import SemanticChunkerWrapper
from rag_project.rag.embeddings.embedding_service import EmbeddingService
from rag_project.rag.chromaDB.chroma_store import ChromaStore

logger = setup_logger("INGESTION", "ingestion.log")


def run_ingestion(project_root: Path) -> dict:
    logger.info("Starting ingestion pipeline")

    with open(project_root / "config.yaml") as f:
        config = yaml.safe_load(f)

    raw_dir = project_root / config["paths"]["data"]["raw_dir"]
    chroma_dir = project_root / config["paths"]["vector_store"]["chroma_dir"]
    chroma_dir.mkdir(parents=True, exist_ok=True)

    with timed_block("Loading documents", logger):
        documents = DataLoader(raw_dir).load()

    rcfg = config["chunking"]["recursive"]

    with timed_block("Recursive chunking", logger):
        recursive_chunks = RecursiveChunker(
            rcfg["chunk_size"], rcfg["overlap"]
        ).chunk(documents)

    embedder = EmbeddingService()

    semantic_chunks = recursive_chunks
    if config["chunking"]["semantic"]["enabled"]:
        with timed_block("Semantic chunking", logger):
            semantic_chunks = SemanticChunkerWrapper(
                embedder.model
            ).chunk(recursive_chunks)

    with timed_block("Embedding documents", logger):
        embeddings = embedder.embed_documents(semantic_chunks)

    store = ChromaStore(
        persist_dir=chroma_dir,
        collection_name=config["chroma"]["collection_name"],
    )

    with timed_block("Persisting to ChromaDB", logger):
        store.add(
            ids=[str(i) for i in range(len(semantic_chunks))],
            documents=[d.page_content for d in semantic_chunks],
            embeddings=embeddings,
            metadatas=[d.metadata for d in semantic_chunks],
        )

    logger.info(f"Ingestion completed | chunks={len(semantic_chunks)}")

    return {
        "status": "success",
        "chunks_ingested": len(semantic_chunks)
    }
