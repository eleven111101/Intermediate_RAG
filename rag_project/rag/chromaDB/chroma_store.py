from pathlib import Path
import chromadb
from chromadb.config import Settings


class ChromaStore:
    def __init__(
        self,
        persist_dir: Path,
        collection_name: str,
        require_existing: bool = False,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        # ----------------------------------------
        # Strict DB enforcement (READ MODE)
        # ----------------------------------------
        if require_existing:
            if not persist_dir.exists() or not any(persist_dir.iterdir()):
                raise RuntimeError(
                    "Vector DB does not exist. Run ingestion first."
                )

        self.client = chromadb.Client(
            Settings(
                persist_directory=str(persist_dir),
                is_persistent=True,
            )
        )

        # ----------------------------------------
        # Collection Handling
        # ----------------------------------------
        if require_existing:
            self.collection = self.client.get_collection(collection_name)
        else:
            self.collection = self.client.get_or_create_collection(
                collection_name
            )

    # ----------------------------------------
    # ADD METHOD (FOR INGESTION)
    # ----------------------------------------
    def add(self, ids, documents, embeddings, metadatas=None):
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )


    # ----------------------------------------
    # OPTIONAL PERSIST (if needed)
    # ----------------------------------------
    def persist(self):
        # For newer Chroma versions this may not be required
        pass
