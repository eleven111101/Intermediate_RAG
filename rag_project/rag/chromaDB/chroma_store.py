from pathlib import Path
import chromadb


class ChromaStore:
    def __init__(self, persist_dir: Path, collection_name: str):
        # 🔑 MUST use PersistentClient for disk storage
        self.client = chromadb.PersistentClient(
            path=str(persist_dir)
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def add(self, ids, documents, embeddings, metadatas):
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
