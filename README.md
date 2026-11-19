# kobajin.com

A static website for chatting with kobajinAI, an experimental web-based twin of kobajin. Model inference runs in the visitor's browser through WebGPU. The site has no API server and does not send chat messages to a backend.

## Run locally

```sh
npm install
npm run dev
```

Open the localhost URL printed by Vite in a recent WebGPU-capable desktop browser. The app starts loading the model on first open. WebLLM downloads the model weights from the public Hugging Face repository and its prebuilt WebGPU library from MLC's static library repository, then caches both in browser storage for later sessions. The browser bundles a small, separately curated set of confirmed memories; chat messages stay in the current browser session and are never sent to a server.

For local model-bundle testing, start the bundle server and point Vite at it:

```sh
python3 self_twin/webllm/serve_local_bundle.py
VITE_MODEL_URL=http://127.0.0.1:8001/ npm run dev -- --host 127.0.0.1
```

Without `VITE_MODEL_URL`, the app uses the public `kobajin/kobajinAI` repository.

## Publish

Build the static site with `npm run build` and publish the generated `dist/` directory to any HTTPS static host. The service worker caches the application shell and Vite assets. WebLLM separately caches the model. The first visit needs an internet connection; after the application shell and model are cached, inference can run offline, subject to browser storage retention.

The model repository used by the app is `kobajin/kobajinAI`. For anonymous browser downloads, the repository must allow public read access. Do not place a Hugging Face write token in the frontend.

See [`self_twin/webllm/README.md`](self_twin/webllm/README.md) for the exact model files to upload and how to prepare WebLLM-compatible weights from the merged checkpoint.

The current HF model revision is [`a2534a20253d54cce925307fbd76086498b09a11`](https://huggingface.co/kobajin/kobajinAI/commit/a2534a20253d54cce925307fbd76086498b09a11). It is an owner-authorized experimental update that improves on the previous published candidate; evaluation still ranks the unadapted base higher on real conversation references.
