#!/usr/bin/env python3
"""
scripts/serve.py — Local development server for MiniGPT browser app.

Fixes the #1 deployment gotcha: browsers (and Python's built-in http.server)
don't always serve .onnx files with the correct MIME type, which can cause
ONNX Runtime Web to reject the model file.

This server explicitly sets:
  .onnx → application/octet-stream
  .wasm → application/wasm
  .js   → application/javascript

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


# ── Custom MIME Handler ────────────────────────────────────────────────────────

class MiniGPTHandler(http.server.SimpleHTTPRequestHandler):
    """Extends SimpleHTTPRequestHandler with correct MIME types for ML files."""

    # Override / extend default MIME map
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".onnx":    "application/octet-stream",
        ".wasm":    "application/wasm",
        ".js":      "application/javascript",
        ".mjs":     "application/javascript",
        ".json":    "application/json",
        ".txt":     "text/plain",
        ".md":      "text/markdown",
        ".html":    "text/html; charset=utf-8",
        ".css":     "text/css",
        ".map":     "application/json",
        "":         "application/octet-stream",
    }

    def end_headers(self):
        # Required for SharedArrayBuffer (used by some WASM threading setups)
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        # Cache control — no-cache for development
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        super().end_headers()

    def log_message(self, fmt, *args):
        # Cleaner log output
        path = args[0] if args else ""
        status = args[1] if len(args) > 1 else ""
        if ".onnx" in path:
            print(f"  \033[32m[MODEL]\033[0m  {path} → {status}")
        elif ".wasm" in path:
            print(f"  \033[34m[WASM]\033[0m   {path} → {status}")
        elif ".js" in path:
            print(f"  \033[33m[JS]\033[0m     {path} → {status}")
        else:
            print(f"  \033[90m[HTTP]\033[0m   {path} → {status}")


# ── Server Setup ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MiniGPT local dev server with correct ONNX/WASM MIME types"
    )
    parser.add_argument(
        "--port", "-p", type=int, default=8080,
        help="Port to serve on (default: 8080)"
    )
    parser.add_argument(
        "--dir", "-d", default="web",
        help="Directory to serve (default: web/)"
    )
    parser.add_argument(
        "--no-open", action="store_true",
        help="Don't auto-open browser"
    )
    args = parser.parse_args()

    serve_dir = Path(args.dir).resolve()
    if not serve_dir.exists():
        print(f"Error: directory '{args.dir}' does not exist.")
        print(f"Run 'python main.py export' first to generate web/assets/mini_gpt_quant.onnx")
        sys.exit(1)

    os.chdir(serve_dir)

    # Check for model file
    model_path = serve_dir / "assets" / "mini_gpt_quant.onnx"
    if not model_path.exists():
        print(f"\n⚠️  Model not found at {model_path}")
        print("   Run: python main.py export")
        print("   before starting the server.\n")

    url = f"http://localhost:{args.port}"

    with socketserver.TCPServer(("", args.port), MiniGPTHandler) as httpd:
        httpd.allow_reuse_address = True

        print(f"\n{'─' * 50}")
        print(f"  ⚡ MiniGPT Dev Server")
        print(f"{'─' * 50}")
        print(f"  Serving : {serve_dir}")
        print(f"  URL     : \033[4m{url}\033[0m")
        print(f"  ONNX    : {'✓ found' if model_path.exists() else '✗ missing (run export first)'}")
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
