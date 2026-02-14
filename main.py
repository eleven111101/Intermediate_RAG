from pathlib import Path
import subprocess
import sys
import yaml
import time
import signal
import webbrowser
import requests

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag_project.db_checker_files import SystemChecker


def wait_for_api(host: str, port: int, timeout: int = 30):
    connect_host = "127.0.0.1" if host == "0.0.0.0" else host
    url = f"http://{connect_host}:{port}/docs"

    start = time.time()

    while True:
        try:
            requests.get(url)
            print("API is ready.")
            return
        except requests.exceptions.ConnectionError:
            if time.time() - start > timeout:
                raise RuntimeError("API did not start in time.")
            time.sleep(1)


def main():
    start_time = time.time()

    checker = SystemChecker(PROJECT_ROOT)

    with open(PROJECT_ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)

    python_exec = sys.executable

    # ----------------------------------------
    # Optional DB Clear
    # ----------------------------------------
    checker.clear_db_if_enabled()

    print("\nStarting RAG System...\n")

    # ----------------------------------------
    # Start FastAPI
    # ----------------------------------------
    print("Launching FastAPI...")
    api_process = subprocess.Popen(
        [
            python_exec,
            "-m",
            "uvicorn",
            "rag_project.api.app:app",
            "--host",
            config["api"]["host"],
            "--port",
            str(config["api"]["port"]),
        ]
    )

    # Wait for API ready
    wait_for_api(config["api"]["host"], config["api"]["port"])

    # ----------------------------------------
    # Start Streamlit
    # ----------------------------------------
    print("Launching Streamlit...")
    streamlit_process = subprocess.Popen(
        [
            python_exec,
            "-m",
            "streamlit",
            "run",
            str(PROJECT_ROOT / "frontend/streamlit_app.py"),
        ]
    )

    # ----------------------------------------
    # Auto-open browser
    # ----------------------------------------
    time.sleep(2)

    total_time = round(time.time() - start_time, 2)

    print(f"\nSystem ready in {total_time} seconds.")
    print("Use Streamlit to run ingestion or queries.")
    print("Press CTRL+C to shutdown.\n")

    # ----------------------------------------
    # Graceful shutdown handler
    # ----------------------------------------
    def shutdown_handler(signum, frame):
        print("\nShutting down system...\n")
        api_process.terminate()
        streamlit_process.terminate()

        api_process.wait()
        streamlit_process.wait()

        print("Shutdown complete.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown_handler(None, None)


if __name__ == "__main__":
    main()
