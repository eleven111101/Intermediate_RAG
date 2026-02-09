import yaml
from pathlib import Path

from utils.logger import setup_logger
from utils.timer import timed_block

from rag_project.rag.retriever_files.retriever import DocumentRetriever
from rag_project.rag.retriever_files.reranker import Reranker
from rag_project.rag.llm.ollama_llm import OllamaLLM

PROJECT_ROOT = Path(__file__).resolve().parents[2]
logger = setup_logger("RAG-QUERY", "query.log")

with open(PROJECT_ROOT / "config.yaml") as f:
    config = yaml.safe_load(f)

retriever = DocumentRetriever(
    chroma_dir=PROJECT_ROOT / config["paths"]["vector_store"]["chroma_dir"],
    collection_name=config["chroma"]["collection_name"],
    top_k=config["retrieval"]["top_k"],
)

reranker = Reranker()
llm = OllamaLLM(model_name=config["llm"]["model"])


def run_rag_query(question: str) -> dict:
    logger.info(f"Received query: {question}")

    if len(question.strip()) < 5:
        logger.warning("Query too short")
        return {"status": "invalid_query", "answer": "Query too short."}

    with timed_block("Retrieval", logger):
        docs = retriever.retrieve(question)

    if not docs:
        logger.warning("No relevant context found")
        return {
            "status": "no_context",
            "answer": "No relevant data found in the knowledge base."
        }

    with timed_block("Reranking", logger):
        docs = reranker.rerank(question, docs)

    context = "\n\n".join(docs)

    with timed_block("LLM generation", logger):
        answer = llm.generate(question, context)

    logger.info("Query completed successfully")

    return {
        "status": "success",
        "query": question,
        "answer": answer,
        "sources": []
    }
