from pathlib import Path
import yaml


class SystemChecker:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.config = self._load_config()

        self.raw_dir = project_root / self.config["paths"]["data"]["raw_dir"]
        self.chroma_dir = project_root / self.config["paths"]["vector_store"]["chroma_dir"]

    def _load_config(self):
        with open(self.project_root / "config.yaml", "r") as f:
            return yaml.safe_load(f)

    def check_raw_data(self) -> bool:
        print("\n[CHECK] Raw data")

        if not self.raw_dir.exists():
            print("❌ Raw data directory missing")
            return False

        files = list(self.raw_dir.glob("*"))
        if not files:
            print("❌ Raw data directory empty")
            return False

        print(f"✅ Raw files found: {len(files)}")
        return True

    def check_chromadb(self) -> bool:
        print("\n[CHECK] ChromaDB")

        if not self.chroma_dir.exists():
            print("⚠️ ChromaDB directory missing")
            return False

        sqlite_files = list(self.chroma_dir.glob("*.sqlite3"))
        index_dirs = [p for p in self.chroma_dir.iterdir() if p.is_dir()]

        if sqlite_files or index_dirs:
            print("✅ ChromaDB detected")
            return True

        print("⚠️ ChromaDB empty")
        return False

    def decide(self) -> str:
        print("\n==============================")
        print(" SYSTEM CHECK")
        print("==============================")

        if not self.check_raw_data():
            return "stop"

        if self.check_chromadb():
            print("\n➡️ Decision: QUERY")
            return "query"

        print("\n➡️ Decision: INGEST")
        return "ingest"
