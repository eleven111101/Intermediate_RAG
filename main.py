from pathlib import Path
import subprocess
import sys

# ------------------------------------------------------------
# Ensure project root is importable
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag_project.checker_files import SystemChecker


def main():
    # Initialize system checker
    checker = SystemChecker(PROJECT_ROOT)
    decision = checker.decide()

    # Always use the SAME Python that is running this file
    python_exec = sys.executable

    if decision == "ingest":
        print("\n➡️ Running INGESTION pipeline...\n")
        subprocess.run(
            [python_exec, str(PROJECT_ROOT / "rag_project/scripts/ingestion.py")],
            check=True
        )

    elif decision == "query":
        print("\n➡️ Running QUERY pipeline...\n")
        subprocess.run(
            [python_exec, str(PROJECT_ROOT / "rag_project/scripts/query_rag.py")],
            check=True
        )

    else:
        print("\n❌ System not ready. Fix issues and rerun.")


if __name__ == "__main__":
    main()
