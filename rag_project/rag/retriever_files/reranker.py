from typing import List


class Reranker:
    """
    Reranker interface.
    By default, returns documents unchanged.
    Can be extended with ML-based reranking later.
    """

    def rerank(self, query: str, documents: List[str], top_n: int | None = None) -> List[str]:
        """
        Rerank retrieved documents based on relevance to query.

        Args:
            query (str): User query
            documents (List[str]): Retrieved documents
            top_n (int | None): Optional cutoff after reranking

        Returns:
            List[str]: Reranked documents
        """
        if not documents:
            return []

        if top_n:
            return documents[:top_n]

        return documents
