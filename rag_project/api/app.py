# rag_project/api/app.py

import asyncio
import yaml
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from utils.logger import setup_logger
from rag_project.rag.retriever_files.retriever import DocumentRetriever
from rag_project.rag.retriever_files.reranker import Reranker
from rag_project.rag.llm.ollama_llm import OllamaLLM
from rag_project.scripts.query_rag import RAGService
from rag_project.db_checker_files import SystemChecker


# ------------------------------------------------------------
# Load Config
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

with open(PROJECT_ROOT / "config.yaml") as f:
    config = yaml.safe_load(f)

api_config = config["api"]

logger = setup_logger("API", "app.log")

app = FastAPI(title="RAG API")

# Global service container
rag_service: RAGService = None

# Concurrency
MAX_CONCURRENT_REQUESTS = api_config.get("max_concurrent_requests", 3)
ENABLE_SEMAPHORE = api_config.get("enable_semaphore", True)

semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS) if ENABLE_SEMAPHORE else None


class QueryRequest(BaseModel):
    query: str


# ------------------------------------------------------------
# Startup Initialization
# ------------------------------------------------------------
@app.on_event("startup")
async def startup():

    global rag_service

    logger.info("Initializing RAG components...")

    # Optional DB existence check
    require_db = config.get("database", {}).get("require_existing_db", False)
    checker = SystemChecker(PROJECT_ROOT)

    if require_db and not checker.db_exists():
        logger.warning("DB required but not found. Queries will fail until ingestion.")

    # Initialize retriever
    retriever = DocumentRetriever(
        chroma_dir=PROJECT_ROOT / config["paths"]["vector_store"]["chroma_dir"],
        collection_name=config["chroma"]["collection_name"],
        top_k=config["retrieval"]["top_k"],
        fetch_k=config["retrieval"].get("fetch_k"),
    )

    # Optional reranker
    reranker = None
    if config["reranker"]["enabled"]:
        reranker = Reranker(model_name=config["reranker"]["model_name"])


    # Initialize LLM ONCE
    llm = OllamaLLM()

    rag_service = RAGService(config, retriever, llm, reranker)

    logger.info("RAG components initialized successfully.")
    logger.info(f"Max concurrent requests: {MAX_CONCURRENT_REQUESTS}")
    logger.info(f"Semaphore enabled: {ENABLE_SEMAPHORE}")


# ------------------------------------------------------------
# Query Endpoint
# ------------------------------------------------------------
@app.post("/query")
async def query_rag(req: QueryRequest):

    if not rag_service:
        raise HTTPException(status_code=500, detail="RAG service not initialized")

    try:
        if ENABLE_SEMAPHORE:
            async with semaphore:
                result = await asyncio.wait_for(
                    asyncio.to_thread(rag_service.run, req.query),
                    timeout=api_config.get("request_timeout", 300)
                )
        else:
            result = await asyncio.wait_for(
                asyncio.to_thread(rag_service.run, req.query),
                timeout=api_config.get("request_timeout", 300)
            )

        return result

    except asyncio.TimeoutError:
        logger.error("Request timed out")
        raise HTTPException(status_code=504, detail="LLM request timed out")

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
