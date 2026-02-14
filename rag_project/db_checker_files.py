from pathlib import Path
import yaml
import shutil
from utils.logger import setup_logger

logger = setup_logger("SYSTEM", "system.log")


class SystemChecker:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.config = self._load_config()

        self.raw_dir = project_root / self.config["paths"]["data"]["raw_dir"]
        self.chroma_dir = project_root / self.config["paths"]["vector_store"]["chroma_dir"]

    # ----------------------------------------
    # Load config
    # ----------------------------------------
    def _load_config(self):
        with open(self.project_root / "config.yaml", "r") as f:
            return yaml.safe_load(f)

    # ----------------------------------------
    # Clear DB if enabled in config
    # ----------------------------------------
    def clear_db_if_enabled(self):
        auto_clear = self.config.get("database", {}).get("auto_clear_on_start", False)

        if auto_clear:
            if self.chroma_dir.exists():
                shutil.rmtree(self.chroma_dir)
                logger.warning("Vector DB cleared (auto_clear_on_start = true)")
            else:
                logger.info("No Vector DB found to clear.")

    # ----------------------------------------
    # Check if DB exists (strict check)
    # ----------------------------------------
    def db_exists(self) -> bool:
        if not self.chroma_dir.exists():
            return False

        sqlite_files = list(self.chroma_dir.glob("*.sqlite3"))
        index_dirs = [p for p in self.chroma_dir.iterdir() if p.is_dir()]

        return bool(sqlite_files or index_dirs)

    # ----------------------------------------
    # DB Info (optional UI use)
    # ----------------------------------------
    def db_info(self) -> dict:
        exists = self.db_exists()

        if not exists:
            return {"exists": False, "message": "Vector DB not found"}

        return {
            "exists": True,
            "path": str(self.chroma_dir),
        }
