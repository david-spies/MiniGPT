#!/usr/bin/env python3
"""
scripts/serve.py — Local development server for MiniGPT browser app.

Fixes:
  - .onnx files served as application/octet-stream
  - .wasm files served as application/wasm
  - Cross-Origin headers for SharedArrayBuffer support
  - Silent favicon.ico 404 (no traceback spam)
  - Robust log_message that handles HTTPStatus enum args

Usage:
  python scripts/serve.py              # serves web/ on port 8080
  python scripts/serve.py --port 3000  # custom port
  python scripts/serve.py --dir .      # serve from project root
"""
import argparse
import http.server
import os
import socketserver
import sys
import webbrowser
from pathlib import Path


class MiniGPTHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler with correct MIME types and clean logging."""

    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".onnx": "application/octet-stream",
        ".wasm": "application/wasm",
        ".js":   "application/javascript",
        ".mjs":  "application/javascript",
        ".json": "application/json",
        ".txt":  "text/plain",
        ".md":   "text/markdown",
        ".html": "text/html; charset=utf-8",
        ".css":  "text/css",
        "":      "application/octet-stream",
    }

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        super().end_headers()

    def do_GET(self):
        # Silently swallow favicon requests — browsers always ask for it
        if self.path == "/favicon.ico":
            self.send_response(204)  # No Content
            self.end_headers()
            return
        super().do_GET()

    def log_message(self, fmt, *args):
        # args[0] is the request line (str) when called from log_request,
        # but may be an HTTPStatus or other type when called from send_error.
        # Always convert to string first to avoid TypeError.
        first = str(args[0]) if args else ""
        status = str(args[1]) if len(args) > 1 else ""

        if ".onnx" in first:
            print(f"  \033[32m[MODEL]\033[0m  {first} → {status}")
        elif ".wasm" in first:
            print(f"  \033[34m[WASM]\033[0m   {first} → {status}")
        elif ".js" in first:
            print(f"  \033[33m[JS]\033[0m     {first} → {status}")
        elif "favicon" in first or "404" in status:
            pass  # suppress noisy 404s for missing browser assets
        else:
            print(f"  \033[90m[HTTP]\033[0m   {first} → {status}")

    def log_error(self, fmt, *args):
        # Suppress the default stderr error logging entirely —
        # our log_message above already handles what we want to show.
        pass


def main():
    parser = argparse.ArgumentParser(
        description="MiniGPT local dev server"
    )
    parser.add_argument("--port", "-p", type=int, default=8080)
    parser.add_argument("--dir",  "-d", default="web")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    serve_dir = Path(args.dir).resolve()
    if not serve_dir.exists():
        print(f"Error: directory '{args.dir}' does not exist.")
        print("Run 'python main.py export' first.")
        sys.exit(1)

    os.chdir(serve_dir)

    model_path = serve_dir / "assets" / "mini_gpt_quant.onnx"
    url = f"http://localhost:{args.port}"

    with socketserver.TCPServer(("", args.port), MiniGPTHandler) as httpd:
        httpd.allow_reuse_address = True

        print(f"\n{'─' * 50}")
        print(f"  ⚡ MiniGPT Dev Server")
        print(f"{'─' * 50}")
        print(f"  Serving : {serve_dir}")
        print(f"  URL     : \033[4m{url}\033[0m")
        print(f"  ONNX    : {'✓ found' if model_path.exists() else '✗ missing — run: python main.py export'}")
        print(f"{'─' * 50}")
        print(f"  Press Ctrl+C to stop\n")

        if not args.no_open:
            webbrowser.open(url)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\nServer stopped.")


if __name__ == "__main__":
    main()
