from pathlib import Path
import yaml

# Project root = INTERMEDIATE_RAG
PROJECT_ROOT = Path(__file__).resolve().parents[1]

with open(PROJECT_ROOT / "config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

RAW_DIR = PROJECT_ROOT / CONFIG["paths"]["data"]["raw_dir"]
CHROMA_DIR = PROJECT_ROOT / CONFIG["paths"]["vector_store"]["chroma_dir"]
