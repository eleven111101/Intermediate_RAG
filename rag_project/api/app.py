from fastapi import FastAPI
from pydantic import BaseModel

from utils.paths import CHROMA_DIR, CONFIG
from utils.logger import setup_logger

from rag_project.rag.retriever_files.retriever import DocumentRetriever
from rag_project.rag.retriever_files.reranker import Reranker
from rag_project.rag.llm.ollama_llm import OllamaLLM

logger = setup_logger("RAG-API")

app = FastAPI(title="RAG Service")

# -------------------------
# Load ONCE (speed boost)
# -------------------------
retriever = DocumentRetriever(
    chroma_dir=CHROMA_DIR,
    collection_name=CONFIG["chroma"]["collection_name"],
    top_k=CONFIG["retrieval"]["top_k"],
)

reranker = Reranker()

llm = OllamaLLM(
    model_name=CONFIG["llm"]["model"]
)

# -------------------------
# Request / Response models
# -------------------------
class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    context: list[str]


# -------------------------
# Health check
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# -------------------------
# Query endpoint
# -------------------------
@app.post("/query", response_model=QueryResponse)
def query_rag(req: QueryRequest):
    logger.info(f"Question: {req.question}")

    docs = retriever.retrieve(req.question)
    docs = reranker.rerank(req.question, docs)

    context_texts = docs
    context = "\n\n".join(context_texts)

    answer = llm.generate(req.question, context)

    return {
        "answer": answer,
        "context": context_texts,
    }
