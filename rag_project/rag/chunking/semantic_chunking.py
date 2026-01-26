from langchain_experimental.text_splitter import SemanticChunker


class SemanticChunkerWrapper:
    def __init__(self, embedding_model):
        self.splitter = SemanticChunker(embedding_model)

    def chunk(self, documents):
        return self.splitter.split_documents(documents)
