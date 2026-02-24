import yaml
import subprocess
import sys
import time
import requests
from pathlib import Path

from rag_project.db_checker_files import SystemChecker

PROJECT_ROOT = Path(__file__).resolve().parent


# ------------------------------------------------------------
# Load Config
# ------------------------------------------------------------
def load_config():
    with open(PROJECT_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------
# Wait for API (Health Check Based)
# ------------------------------------------------------------
def wait_for_api(host, port, timeout=90):
    url = f"http://127.0.0.1:{port}/health"
    start = time.time()

    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                print("API is ready.")
                return
        except Exception:
            pass

        time.sleep(1)

    raise RuntimeError("API did not start in time.")


# ------------------------------------------------------------
# Graceful Shutdown
# ------------------------------------------------------------
def shutdown_process(proc, name):
    if proc and proc.poll() is None:
        print(f"Stopping {name}...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    config = load_config()

    # -----------------------------------
    # System Checker
    # -----------------------------------
    checker = SystemChecker(PROJECT_ROOT)
    checker.clear_db_if_enabled()

    print("\nStarting RAG System...\n")

    # -----------------------------------
    # Select Active POC
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
        ],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    # Wait until API is healthy
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
        ],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    print("\nSystem running.")
    print("Press CTRL+C to shutdown.\n")

    try:
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down...\n")
        shutdown_process(api_process, "FastAPI")
        shutdown_process(streamlit_process, "Streamlit")
        print("Shutdown complete.")


if __name__ == "__main__":
    main()