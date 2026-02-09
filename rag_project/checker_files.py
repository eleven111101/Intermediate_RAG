from pathlib import Path
import yaml
from utils.logger import setup_logger

logger = setup_logger("SYSTEM", "system.log")


class SystemChecker:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.config = self._load_config()
        self.raw_dir = project_root / self.config["paths"]["data"]["raw_dir"]
        self.chroma_dir = project_root / self.config["paths"]["vector_store"]["chroma_dir"]

    def _load_config(self):
        with open(self.project_root / "config.yaml", "r") as f:
            return yaml.safe_load(f)

    def db_info(self) -> dict:
        logger.info("Checking Vector DB status")

        if not self.chroma_dir.exists():
            logger.warning("Chroma directory not found")
            return {"exists": False, "message": "Vector DB not found"}

        sqlite_files = list(self.chroma_dir.glob("*.sqlite3"))
        index_dirs = [p for p in self.chroma_dir.iterdir() if p.is_dir()]

        if not sqlite_files and not index_dirs:
            logger.warning("Chroma directory empty")
            return {"exists": False, "message": "Vector DB empty"}

        logger.info("Vector DB loaded successfully")
        return {
            "exists": True,
            "path": str(self.chroma_dir),
            "collections": len(index_dirs)
        }
