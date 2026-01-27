from pathlib import Path
import subprocess

from rag_project.checker_files import SystemChecker


def main():
    project_root = Path(__file__).resolve().parent
    checker = SystemChecker(project_root)

    decision = checker.decide()

    if decision == "ingest":
        subprocess.run(["python", "rag_project/scripts/ingestion.py"], check=True)

    elif decision == "query":
        subprocess.run(["python", "rag_project/scripts/query_rag.py"], check=True)

    else:
        print("\n❌ System not ready. Fix issues and rerun.")


if __name__ == "__main__":
    main()
