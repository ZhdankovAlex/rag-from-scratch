"""
`python -m web` — serve the UI on 127.0.0.1 and open it in the browser.
"""
import argparse
import os
import threading
import webbrowser

import uvicorn

# Hugging Face warns about symlinks on Windows without Developer Mode; harmless.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

parser = argparse.ArgumentParser(description="Local web UI for the RAG pipeline.")
parser.add_argument("--port", type=int, default=8770, help="port to listen on (default: 8770)")
parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
args = parser.parse_args()

if not args.no_browser:
    threading.Timer(1.5, webbrowser.open, [f"http://127.0.0.1:{args.port}/"]).start()
# Local address only: the UI can re-index and run the model, it is not meant to be exposed.
uvicorn.run("web.app:app", host="127.0.0.1", port=args.port, log_level="warning")
