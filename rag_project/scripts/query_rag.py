import sys
from pathlib import Path

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from utils.logger import setup_logger
from utils.timer import timed_block


# ------------------------------------------------------------
# Ensure project root is visible
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from rag_project.rag.retriever_files.retriever import DocumentRetriever
from rag_project.rag.retriever_files.reranker import Reranker
from rag_project.rag.llm.ollama_llm import OllamaLLM

# ------------------------------------------------------------
# Logger (INITIALIZE ONCE, AT MODULE LOAD)
# ------------------------------------------------------------
logger = setup_logger(
    name="RAG-QUERY",
    log_file="logs/query.log"
)


def main():
    logger.info("Starting RAG query service")

    # ------------------------------------------------------------
    # Load configuration
    # ------------------------------------------------------------
    with open(PROJECT_ROOT / "config.yaml", "r") as f:
        config = yaml.safe_load(f)

    chroma_dir = PROJECT_ROOT / config["paths"]["vector_store"]["chroma_dir"]

    logger.info(f"Vector DB Path: {chroma_dir}")
    logger.info(f"Collection: {config['chroma']['collection_name']}")
    logger.info(f"Top-K: {config['retrieval']['top_k']}")

    print("\n==============================")
    print(" RAG QUERY + OLLAMA (phi)")
    print("==============================")
    print(f"Vector DB Path  : {chroma_dir}")
    print(f"Collection     : {config['chroma']['collection_name']}")
    print(f"Top-K          : {config['retrieval']['top_k']}")
    print("==============================\n")

    # ------------------------------------------------------------
    # Initialize components
    # ------------------------------------------------------------
    with timed_block("Retriever initialization", logger):
        retriever = DocumentRetriever(
            chroma_dir=chroma_dir,
            collection_name=config["chroma"]["collection_name"],
            top_k=config["retrieval"]["top_k"],
        )

    reranker = Reranker()

    with timed_block("LLM initialization", logger):
        llm = OllamaLLM(model_name=config["llm"]["model"])

    # ------------------------------------------------------------
    # Query loop
    # ------------------------------------------------------------
    while True:
        question = input("\nAsk a question (type 'exit' to quit): ").strip()

        if question.lower() == "exit":
            logger.info("User exited query loop")
            break

        logger.info(f"User question: {question}")

        # ---------------- STEP 1: RETRIEVAL ----------------
        print("\n[STEP 1] Retrieving context...")
        with timed_block("Retrieval", logger):
            docs = retriever.retrieve(question)

        if not docs:
            logger.warning("No relevant context found")
            print("No relevant context found.")
            continue

        logger.info(f"Retrieved {len(docs)} chunks")

        # ---------------- STEP 2: RERANKING ----------------
        with timed_block("Reranking", logger):
            docs = reranker.rerank(question, docs)

        context = "\n\n".join(docs)

        # ---------------- STEP 3: CONTEXT PREVIEW ----------------
        print("\n[STEP 2] Context sent to LLM (preview):")
        print("-" * 50)
        print(context[:1000])
        print("-" * 50)

        # ---------------- STEP 4: LLM GENERATION ----------------
        print("\n[STEP 3] Generating answer...")
        with timed_block("LLM generation", logger):
            answer = llm.generate(question, context)

        # ---------------- OUTPUT ----------------
        print("\n==============================")
        print(" FINAL ANSWER")
        print("==============================\n")
        print(answer)
        print("\n==============================\n")

        logger.info("Answer generated successfully")


if __name__ == "__main__":
    main()
