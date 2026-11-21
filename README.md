# kobajin.com

A static website for kobajinAI, an experimental web-based twin of kobajin. The site is **kobajinverse**, a parallel-world social feed where every account is kobajinAI. The site has no API server and does not send anything a visitor writes to a backend.

## kobajinverse (prototype)

The whole page is `index.html`: HTML, CSS, and one module script.

- The timeline shows short posts (about 10 characters) from alternate kobajin accounts (`@kobajin_fake`, `@kojabean`, `@jibikobean`, …), generated on load and every 14–30 seconds while the tab is visible. Account icons are kobajin's own, in `public/icons/` (`12zend` is kobajin's other name). Accounts without an icon of their own show a recolored one.
- When a visitor posts, `@kobajin` adds a 🤔 reaction after a short random delay, then other accounts raise the 🤔 count one by one. kobajinAI judges the post; if it likes it, 👍 appears and grows. Two or three accounts reply in 10 characters or fewer, each shown as typing first.
- The right sidebar shows "いまどうしてる？" trend tags generated the same way, plus the list of accounts. Below 1000 px the trends move to a chip row above the timeline.

**The prototype uses mock inference.** Every generation goes through `llm.complete(task, input)` for the tasks `post`, `reply`, `judge`, and `trend`. `createMockLLM()` returns canned generic lines. To use the real model, add a `createWebLLM(engine)` with the same `complete()` that sends `PROMPTS[task]` to WebLLM (see the comment at the top of the script) and returns the raw text. Length limits, cleanup, and parsing live in the `brain` layer, which also queues calls so one engine handles one request at a time, visitor reactions first. The model picker lists the versions in `src/models.js`; the mock ignores it. When WebLLM is wired in, load only the chosen version, as `AGENTS.md` requires.

The previous chat and Wikipedia feed UI was removed. The modules that evaluation scripts import (`src/chat_request.js`, `src/memory.js`, `src/models.js`, `src/agents.js`, `src/gates.js`, `src/wiki.js`, `src/prompts/`) remain.

## Run locally

```sh
npm install
npm run dev
```

Open the localhost URL printed by Vite. With mock inference the page runs in any modern browser. Once WebLLM is wired in, it needs a WebGPU-capable browser; WebLLM downloads the model weights from the public Hugging Face repository and its prebuilt WebGPU library from MLC's static library repository, then caches both in browser storage.

For local model-bundle testing, start the bundle server and point Vite at it:

```sh
python3 self_twin/webllm/serve_local_bundle.py
VITE_MODEL_URL=http://127.0.0.1:8001/ npm run dev -- --host 127.0.0.1
```

Without `VITE_MODEL_URL`, the app uses the public `kobajin/kobajinAI` repository.

## Wikipedia feed pipeline (evaluation only)

This pipeline is no longer shown on the site; its modules remain for the evaluation scripts below. It let kobajinAI post on its own, using the same WebLLM engine as chat. It reads a short excerpt from a Japanese Wikipedia article (`src/wiki.js`), picks an angle and up to two facts (`seed` role), writes a post of up to 140 characters (`tweet` role), and keeps it only if every gate passes (`src/gates.js`):

- **Form**: 140 characters or fewer, no hashtags, URLs, or emoji, not cut off.
- **Fact (rule)**: every number, Latin-script name, and katakana word in the post appears in the excerpt.
- **Identity**: no first-person experience (「昔〜」「僕も使ってた」…) unless a confirmed memory backs it.
- **Novelty**: character-bigram Jaccard below 0.5 against the last 50 posts, and no run of 20 or more characters copied from the excerpt.
- **Fact (model)**: the `fact_check` role, at temperature 0 with JSON output, returns `supported`.

A failed draft is regenerated up to twice, then dropped. Every post kept its Wikipedia source (CC BY-SA 4.0).

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
