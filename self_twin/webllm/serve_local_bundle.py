#!/usr/bin/env python3
"""Serve the ignored MLC bundle on localhost for WebLLM browser testing."""

from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


HERE = Path(__file__).resolve().parent
BUNDLE = HERE / "hf-upload"


class CorsHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        super().end_headers()

    def translate_path(self, path: str) -> str:
        # WebLLM accepts a model URL as-is only if it has the "…/resolve/<rev>/" shape.
        prefix = "/resolve/local/"
        if path.startswith(prefix):
            path = "/" + path[len(prefix):]
        return super().translate_path(path)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--bundle", default="", help="folder of hf-upload/ to serve, e.g. kobajin-clean-v2 (default: v3 at the root)")
    args = parser.parse_args()
    bundle = BUNDLE / args.bundle
    if not (bundle / "mlc-chat-config.json").is_file():
        raise SystemExit(f"WebLLM bundle is missing: {bundle}")

    handler = partial(CorsHandler, directory=str(bundle))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving local WebLLM bundle {bundle} at http://{args.host}:{args.port}/resolve/local/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
