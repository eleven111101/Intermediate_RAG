from pathlib import Path
from typing import List

from rag_project.rag.embeddings.embedding_service import EmbeddingService
from rag_project.rag.chromaDB.chroma_store import ChromaStore


class DocumentRetriever:
    """
    Responsible only for vector retrieval.
    """

    def __init__(
        self,
        chroma_dir: Path,
        collection_name: str,
        top_k: int = 5,
        fetch_k: int | None = None,
        require_existing_db: bool = False,
    ):
        self.top_k = top_k
        self.fetch_k = fetch_k

        self.embedder = EmbeddingService()

        self.store = ChromaStore(
            persist_dir=chroma_dir,
            collection_name=collection_name,
            require_existing=require_existing_db,
        )

    def retrieve(self, query: str) -> List[str]:
        query_embedding = self.embedder.model.embed_query(query)

        results = self.store.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.fetch_k or self.top_k,
        )

        documents = results.get("documents", [[]])[0]

        return documents[: self.top_k]
