from pathlib import Path
import sys
import traceback

print("=== MAIN STARTED ===")

PROJECT_ROOT = Path(__file__).resolve().parent
print("PROJECT_ROOT:", PROJECT_ROOT)

RAW_DIR = PROJECT_ROOT / "rag_project/data/raw"
print("RAW_DIR EXISTS:", RAW_DIR.exists())
print("RAW_DIR CONTENTS:", list(RAW_DIR.iterdir()))

try:
    from rag_project.rag.loaders.data_loader import DataLoader
    from utils.logger import setup_logger

    logger = setup_logger(
        name="TEST",
        log_dir=PROJECT_ROOT / "log",
        log_file="test.log",
        level="INFO",
    )

    loader = DataLoader(
        raw_data_dir=RAW_DIR,
        logger=logger,
    )

    docs = loader.load()

    print("DOCUMENTS LOADED:", len(docs))

    if docs:
        print("FIRST 200 CHARS:\n")
        print(docs[0].page_content[:200])

except Exception:
    print("=== ERROR ===")
    traceback.print_exc()
    sys.exit(1)

print("=== MAIN FINISHED ===")
