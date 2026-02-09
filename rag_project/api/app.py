from fastapi import FastAPI
from pydantic import BaseModel

from utils.logger import setup_logger
from rag_project.scripts.query_rag import run_rag_query

logger = setup_logger("API", "app.log")

app = FastAPI(title="RAG API")


class QueryRequest(BaseModel):
    query: str


@app.on_event("startup")
def startup():
    logger.info("FastAPI application started")


@app.post("/query")
def query_rag(req: QueryRequest):
    logger.info("POST /query called")
    return run_rag_query(req.query)
