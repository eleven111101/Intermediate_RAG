import yaml
import subprocess
import sys
import time
import requests
from pathlib import Path

from rag_project.db_checker_files import SystemChecker

PROJECT_ROOT = Path(__file__).resolve().parent


def load_config():
    with open(PROJECT_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def wait_for_api(host, port, timeout=60):
    url = f"http://127.0.0.1:{port}/docs"
    start = time.time()

    while time.time() - start < timeout:
        try:
            requests.get(url)
            print("API is ready.")
            return
        except Exception:
            time.sleep(1)

    raise RuntimeError("API did not start in time.")


def main():
    config = load_config()

    # -----------------------------------
    # System Checker
    # -----------------------------------
    checker = SystemChecker(PROJECT_ROOT)
    checker.clear_db_if_enabled()

    print("\nStarting RAG System...\n")

    # -----------------------------------
    # Select POC
    # -----------------------------------
    active_poc = config["app"]["active_poc"]
    poc_config = config["app"][active_poc]

    api_module = poc_config["api_module"]
    streamlit_entry = poc_config["streamlit_entry"]

    host = config["api"]["host"]
    port = config["api"]["port"]

    # -----------------------------------
    # Launch FastAPI
    # -----------------------------------
    print("Launching FastAPI...")

    api_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            api_module,
            "--host",
            host,
            "--port",
            str(port),
        ]
    )

    wait_for_api(host, port)

    # -----------------------------------
    # Launch Streamlit
    # -----------------------------------
    print("Launching Streamlit...")

    streamlit_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(PROJECT_ROOT / streamlit_entry),
        ]
    )

    print("\nSystem running.")
    print("Press CTRL+C to shutdown.\n")

    try:
        api_process.wait()
        streamlit_process.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        api_process.terminate()
        streamlit_process.terminate()


if __name__ == "__main__":
    main()
