import sys
from pathlib import Path
import yaml

# Ensure project root is visible
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from rag_project.rag.retriever_files.retriever import DocumentRetriever
from rag_project.rag.retriever_files.reranker import Reranker
from rag_project.rag.llm.ollama_llm import OllamaLLM


def main():
    with open(PROJECT_ROOT / "config.yaml", "r") as f:
        config = yaml.safe_load(f)

    chroma_dir = PROJECT_ROOT / config["paths"]["vector_store"]["chroma_dir"]

    print("\n==============================")
    print(" RAG QUERY + OLLAMA (phi)")
    print("==============================")
    print(f"Vector DB Path  : {chroma_dir}")
    print(f"Collection     : {config['chroma']['collection_name']}")
    print(f"Top-K          : {config['retrieval']['top_k']}")
    print("==============================\n")

    # Retriever
    retriever = DocumentRetriever(
        chroma_dir=chroma_dir,
        collection_name=config["chroma"]["collection_name"],
        top_k=config["retrieval"]["top_k"],
    )

    # Optional reranker
    reranker = Reranker()

    # LLM (loaded ONCE)
    llm = OllamaLLM(model_name=config["llm"]["model"])

    while True:
        question = input("\nAsk a question (type 'exit' to quit): ").strip()
        if question.lower() == "exit":
            break

        print("\n[STEP 1] Retrieving context...")
        docs = retriever.retrieve(question)

        if not docs:
            print("No relevant context found.")
            continue

        # Rerank (safe even if pass-through)
        docs = reranker.rerank(question, docs)

        context = "\n\n".join(docs)

        print("\n[STEP 2] Context sent to LLM (preview):")
        print("-" * 50)
        print(context[:1000])
        print("-" * 50)

        print("\n[STEP 3] Generating answer...")
        answer = llm.generate(question, context)

        print("\n==============================")
        print(" FINAL ANSWER")
        print("==============================\n")
        print(answer)
        print("\n==============================\n")


if __name__ == "__main__":
    main()
