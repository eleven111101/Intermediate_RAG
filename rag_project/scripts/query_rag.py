# rag_project/scripts/query_rag.py

from typing import Optional
from utils.logger import setup_logger
from utils.timer import timed_block

logger = setup_logger("RAG-QUERY", "query.log")


class RAGService:
    """
    RAG Service holds initialized components.
    Initialized ONCE at API startup.
    """

    def __init__(self, config, retriever, llm, reranker=None):
        self.config = config
        self.retriever = retriever
        self.llm = llm
        self.reranker = reranker

    def run(self, question: str) -> dict:
        logger.info(f"Received query: {question}")

        # Validate query length
        min_length = self.config["api"].get("min_query_length", 5)
        if len(question.strip()) < min_length:
            return {
                "status": "invalid_query",
                "answer": "Query too short."
            }

        # ---------------- Retrieval ----------------
        with timed_block("Retrieval", logger):
            documents = self.retriever.retrieve(question)

        if not documents:
            return {
                "status": "no_context",
                "answer": "No relevant data found."
            }

        # ---------------- Optional Reranking ----------------
        if self.reranker:
            with timed_block("Reranking", logger):
                documents = self.reranker.rerank(question, documents)

        # ---------------- Context Truncation ----------------
        context = "\n\n".join(documents)

        if self.config["llm"].get("context_truncate", False):
            max_chars = self.config["llm"].get("max_context_chars", 4000)
            context = context[:max_chars]

        # ---------------- LLM Generation ----------------
        with timed_block("LLM generation", logger):
            answer = self.llm.generate(question, context)

        return {
            "status": "success",
            "query": question,
            "answer": answer,
            "sources": documents if self.config["query"]["show_context"] else []
        }
