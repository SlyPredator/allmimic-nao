#!/usr/bin/env python3
"""Local development HTTP server with COOP/COEP headers for high-performance WebAssembly.

Usage:
    python web_dep/serve.py --port 8080
"""

import argparse
import http.server
import os
import socketserver
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PUBLIC_DIR = Path(__file__).resolve().parent / "public"


class WasmHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PUBLIC_DIR), **kwargs)

    def end_headers(self):
        # Enable SharedArrayBuffer and WebAssembly multithreading
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache, must-revalidate")
        super().end_headers()

    def translate_path(self, path):
        # Route /assets/... requests directly to repository root assets/
        clean_path = path.split("?", 1)[0].split("#", 1)[0]
        if clean_path.startswith("/assets/"):
            rel = clean_path[len("/assets/"):]
            return str(ROOT / "assets" / rel)
        return super().translate_path(path)

    def guess_type(self, path):
        # Ensure correct MIME type for WebAssembly and ONNX
        if path.endswith(".wasm"):
            return "application/wasm"
        if path.endswith(".onnx"):
            return "application/octet-stream"
        if path.endswith(".bin"):
            return "application/octet-stream"
        return super().guess_type(path)


def main():
    parser = argparse.ArgumentParser(description="Serve web deployment locally")
    parser.add_argument("--port", type=int, default=8080, help="Port to serve on (default: 8080)")
    args = parser.parse_args()

    # Allow fast address reuse on restarts
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", args.port), WasmHTTPRequestHandler) as httpd:
        print(f"\n" + "=" * 60)
        print(f" SuperMimic Web Visualizer is live!")
        print(f" Serving directory : {PUBLIC_DIR}")
        print(f" URL                : http://localhost:{args.port}/")
        print(f"=" * 60 + "\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")


if __name__ == "__main__":
    main()
