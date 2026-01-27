from pathlib import Path
from typing import List

from rag_project.rag.embeddings.embedding_service import EmbeddingService
from rag_project.rag.chromaDB.chroma_store import ChromaStore


class DocumentRetriever:
    """
    Library class responsible only for vector retrieval.
    """

    def __init__(
        self,
        chroma_dir: Path,
        collection_name: str,
        top_k: int = 5,
    ):
        self.top_k = top_k
        self.embedder = EmbeddingService()
        self.store = ChromaStore(
            persist_dir=chroma_dir,
            collection_name=collection_name,
        )

    def retrieve(self, query: str) -> List[str]:
        query_embedding = self.embedder.model.embed_query(query)

        results = self.store.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k,
        )

        return results.get("documents", [[]])[0]
