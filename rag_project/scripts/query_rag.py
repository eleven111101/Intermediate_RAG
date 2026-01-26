
from pathlib import Path

from rag_project.rag.vector_store.chroma_store import ChromaStore
from rag_project.rag.embeddings.embedding_service import EmbeddingService


def query_rag():
    project_root = Path(__file__).resolve().parents[1]
    persist_dir = project_root / "rag_project/vector_store/chroma"

    embedder = EmbeddingService()

    store = ChromaStore(
        persist_dir=persist_dir,
        collection_name="ashwa_rag",
    )

    print("\n=== RAG QUERY MODE ===\n")

    while True:
        question = input("Ask a question (or 'exit'): ")
        if question.lower() == "exit":
            break

        query_embedding = embedder.model.embed_query(question)

        results = store.query(
            query_embedding=query_embedding,
            n_results=3,
        )

        print("\n--- Retrieved Context ---\n")
        for i, doc in enumerate(results["documents"][0]):
            print(f"[{i+1}] {doc[:300]}...\n")

        print("========================\n")


if __name__ == "__main__":
    query_rag()
