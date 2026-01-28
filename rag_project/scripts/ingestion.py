import yaml
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from rag_project.rag.loaders.data_loader import DataLoader
from rag_project.rag.chunking.recursive_chunking import RecursiveChunker
from rag_project.rag.chunking.semantic_chunking import SemanticChunkerWrapper
from rag_project.rag.embeddings.embedding_service import EmbeddingService
from rag_project.rag.chromaDB.chroma_store import ChromaStore


def main():
    root = Path(__file__).resolve().parents[2]

    with open(root / "config.yaml") as f:
        config = yaml.safe_load(f)

    raw_dir = root / config["paths"]["data"]["raw_dir"]
    chroma_dir = root / config["paths"]["vector_store"]["chroma_dir"]
    chroma_dir.mkdir(parents=True, exist_ok=True)

    documents = DataLoader(raw_dir).load()

    rcfg = config["chunking"]["recursive"]
    recursive_chunks = RecursiveChunker(
        rcfg["chunk_size"], rcfg["overlap"]
    ).chunk(documents)

    embedder = EmbeddingService()
    _ = embedder.embed_documents(recursive_chunks)

    semantic_chunks = recursive_chunks
    if config["chunking"]["semantic"]["enabled"]:
        semantic_chunks = SemanticChunkerWrapper(
            embedder.model
        ).chunk(recursive_chunks)

    embeddings = embedder.embed_documents(semantic_chunks)

    store = ChromaStore(
        persist_dir=chroma_dir,
        collection_name=config["chroma"]["collection_name"],
    )

    store.add(
        ids=[str(i) for i in range(len(semantic_chunks))],
        documents=[d.page_content for d in semantic_chunks],
        embeddings=embeddings,
        metadatas=[d.metadata for d in semantic_chunks],
    )

    print(f"\n✅ Ingested {len(semantic_chunks)} chunks into ChromaDB")


if __name__ == "__main__":
    main()
