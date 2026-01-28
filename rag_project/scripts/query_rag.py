import yaml
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))


from rag_project.rag.retriever_files.retriever import DocumentRetriever
from rag_project.rag.retriever_files.reranker import Reranker



def main():
    root = Path(__file__).resolve().parents[2]

    with open(root / "config.yaml") as f:
        config = yaml.safe_load(f)

    chroma_dir = root / config["paths"]["vector_store"]["chroma_dir"]

    print("\n==============================")
    print(" RAG QUERY SERVICE")
    print("==============================")
    print(f"Vector DB Path  : {chroma_dir}")
    print(f"Collection Name : {config['chroma']['collection_name']}")
    print(f"Top-K           : {config['retrieval']['top_k']}")
    print("==============================\n")

    retriever = DocumentRetriever(
        chroma_dir=chroma_dir,
        collection_name=config["chroma"]["collection_name"],
        top_k=config["retrieval"]["top_k"],
    )

    query = input("Ask a question: ")

    docs = retriever.retrieve(query)

    if config.get("reranking", {}).get("enabled", False):
        docs = Reranker().rerank(query, docs)

    print("\n=== RETRIEVED CONTEXT ===\n")
    for i, d in enumerate(docs, 1):
        print(f"[{i}] {d[:500]}")
        print("----")


if __name__ == "__main__":
    main()
