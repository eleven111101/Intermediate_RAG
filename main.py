from pathlib import Path
import subprocess
import sys

from rag_project.checker_files import SystemChecker


def main():
    # Absolute project root (where main.py lives)
    project_root = Path(__file__).resolve().parent

    # Initialize system checker
    checker = SystemChecker(project_root)
    decision = checker.decide()

    # Always use the SAME Python that is running this file
    python_exec = sys.executable

    if decision == "ingest":
        print("\n➡️ Running INGESTION pipeline...\n")
        subprocess.run(
            [python_exec, "rag_project/scripts/ingestion.py"],
            check=True
        )

    elif decision == "query":
        print("\n➡️ Running QUERY pipeline...\n")
        subprocess.run(
            [python_exec, "rag_project/scripts/query_rag.py"],
            check=True
        )

    else:
        print("\n❌ System not ready. Fix issues and rerun.")


if __name__ == "__main__":
    main()
