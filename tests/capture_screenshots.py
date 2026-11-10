"""Capture full-fidelity screenshots of all four application sections using headless Edge."""
from pathlib import Path
import subprocess
import time
import urllib.request
import urllib.parse
import os

EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
SCREENSHOTS_DIR = Path("docs/screenshots")
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

PAGES = [
    ("Document Review", "document_review.png"),
    ("Research Evaluation", "research_evaluation.png"),
    ("Demo Centre", "demo_centre.png"),
    ("Methodology", "methodology.png"),
]

def main():
    print("Launching Streamlit server in background...")
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    
    proc = subprocess.Popen([
        r".venv\Scripts\python.exe",
        "-m", "streamlit",
        "run", "app.py",
        "--server.port=8501",
        "--server.address=127.0.0.1",
        "--server.headless=true",
        "--browser.gatherUsageStats=false"
    ], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        # Wait for server
        ready = False
        for attempt in range(20):
            time.sleep(1)
            try:
                with urllib.request.urlopen("http://127.0.0.1:8501/_stcore/health", timeout=2) as resp:
                    if resp.status == 200:
                        ready = True
                        print(f"Streamlit server healthy after {attempt+1}s.")
                        break
            except Exception:
                pass

        if not ready:
            print("Server failed to respond.")
            return

        for page_title, filename in PAGES:
            out_file = SCREENSHOTS_DIR / filename
            encoded_page = urllib.parse.quote(page_title)
            target_url = f"http://127.0.0.1:8501/?page={encoded_page}"

            cmd = [
                EDGE_PATH,
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--virtual-time-budget=8000",
                "--window-size=1280,950",
                f"--screenshot={out_file.resolve()}",
                target_url
            ]
            print(f"Capturing '{page_title}' to {out_file.name}...")
            res = subprocess.run(cmd, capture_output=True, timeout=25)
            if out_file.exists():
                size = out_file.stat().st_size
                print(f"  -> Success: {out_file.name} ({size:,} bytes, Edge code {res.returncode})")
            time.sleep(1)

    finally:
        print("Stopping Streamlit server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        print("Completed screenshot capture.")

if __name__ == "__main__":
    main()
