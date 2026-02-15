from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, documents, top_n=None):
        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        reranked_docs = [doc for doc, _ in ranked]

        if top_n:
            return reranked_docs[:top_n]

        return reranked_docs
