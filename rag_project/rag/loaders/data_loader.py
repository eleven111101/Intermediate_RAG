from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyPDFLoader


class DataLoader:
    def __init__(self, raw_data_dir: Path):
        self.raw_data_dir = raw_data_dir

        if not self.raw_data_dir.exists():
            raise FileNotFoundError(f"Raw data directory not found: {raw_data_dir}")

    def load(self):
        documents = []

        for file_path in self.raw_data_dir.rglob("*"):
            if not file_path.is_file():
                continue

            if file_path.suffix.lower() == ".txt":
                loader = TextLoader(str(file_path), encoding="utf-8")

            elif file_path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file_path))

            else:
                continue

            docs = loader.load()
            documents.extend(docs)

        return documents
