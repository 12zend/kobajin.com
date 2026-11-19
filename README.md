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

## Feed

The Feed view lets kobajinAI post on its own, using the same WebLLM engine as chat. It reads a short excerpt from a Japanese Wikipedia article (`src/wiki.js`), picks an angle and up to two facts (`seed` role), writes a post of up to 140 characters (`tweet` role), and keeps it only if every gate passes (`src/gates.js`):

- **Form**: 140 characters or fewer, no hashtags, URLs, or emoji, not cut off.
- **Fact (rule)**: every number, Latin-script name, and katakana word in the post appears in the excerpt.
- **Identity**: no first-person experience (「昔〜」「僕も使ってた」…) unless a confirmed memory backs it.
- **Novelty**: character-bigram Jaccard below 0.5 against the last 50 posts, and no run of 20 or more characters copied from the excerpt.
- **Fact (model)**: the `fact_check` role, at temperature 0 with JSON output, returns `supported`.

A failed draft is regenerated up to twice, then dropped. Posts, reply threads, and check logs stay in this browser's IndexedDB (localStorage as a fallback). Every post shows its Wikipedia source (CC BY-SA 4.0). Autonomous posting runs only while the page is visible, the model is loaded, and the visitor has been idle for 45 seconds, at most 3 times per session. Chat takes the engine first and interrupts a running feed job.

Role prompts and sampling live in `src/prompts/` (`roles.json` plus one `.txt` per role). `self_twin/evaluation/run_benchmark.py --role <name>` and the feed evaluation read the same files:

```sh
node self_twin/evaluation/build_tweet_heldout.mjs   # refresh the 20-article held-out set
self_twin/.venv/bin/python self_twin/evaluation/run_tweet_heldout.py \
  --adapter hf=self_twin/adapter/experiment-r8-luna-synthetic-1000-v3/adapter \
  --adapter candidate=path/to/candidate/adapter \
  --output self_twin/evaluation/tweet_heldout-<name>.jsonl
```

It reports gate pass rate, identity violations (invented personal experience), novelty failures (duplicates), and a blinded sheet for scoring kobajin-likeness by hand. Gate logic runs through `self_twin/evaluation/gate_bridge.mjs`, so Python uses the site's own JavaScript.

## Publish

Build the static site with `npm run build` and publish the generated `dist/` directory to any HTTPS static host. The service worker caches the application shell and Vite assets. WebLLM separately caches the model. The first visit needs an internet connection; after the application shell and model are cached, inference can run offline, subject to browser storage retention.

The model repository used by the app is `kobajin/kobajinAI`. For anonymous browser downloads, the repository must allow public read access. Do not place a Hugging Face write token in the frontend.

See [`self_twin/webllm/README.md`](self_twin/webllm/README.md) for the exact model files to upload and how to prepare WebLLM-compatible weights from the merged checkpoint.

The current HF model revision is [`a2534a20253d54cce925307fbd76086498b09a11`](https://huggingface.co/kobajin/kobajinAI/commit/a2534a20253d54cce925307fbd76086498b09a11). It is an owner-authorized experimental update that improves on the previous published candidate; evaluation still ranks the unadapted base higher on real conversation references.
