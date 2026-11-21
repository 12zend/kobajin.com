# WebLLM model bundle for `kobajin/kobajinAI`

The site lets visitors choose among every published kobajinAI version. `src/models.js` lists them. Each record points at the public `kobajin/kobajinAI` Hugging Face repository, pinned to an immutable revision and the version's folder, with a versioned WebLLM `model_id`. v1–v3 are Qwen3.5-4B `q4f16_1` bundles and reuse WebLLM's built-in Qwen3.5-4B WebGPU library. v4, the light version, is an LFM2.5-1.2B-JP `q4f16_1` bundle and reuses WebLLM's built-in LFM2.5-1.2B-Instruct library. The repository therefore needs MLC-converted weights, not the original Transformers `.safetensors` files alone.

The owner decided on 2026-11-21 to publish every trained version (see `AGENTS.md`). Evaluation decides each version's release notes and the site's default, not whether it is uploaded. The default is the version named in `src/closest_model.json`, the one currently judged closest to kobajin.

## Published versions

| Version | Folder at revision | Source adapter | Notes |
|---|---|---|---|
| v1 | `v1/` at [`8d6dab1`](https://huggingface.co/kobajin/kobajinAI/commit/8d6dab15195505c8c92c2d9ceb99f2551239f2d4) | `self_twin/adapter/experiment-r8-kobajin-clean-v1-lr5e5/adapter` | Real chats only. Natural voice; ignored retrieved memories (0/3) and overused 「うんこ」. |
| v2 | `v2/` at [`1f9287d`](https://huggingface.co/kobajin/kobajinAI/commit/1f9287d837eb311f5d0ab9262240ab2ed362d87c) | `self_twin/adapter/experiment-r8-kobajin-clean-v2-memory/adapter` | Real chats plus memory-grounded rows; 2/3 fixed memory questions answered with the memory. The site's default since 2026-11-21. |
| v3 | `v3/` at [`0bc0bea`](https://huggingface.co/kobajin/kobajinAI/commit/0bc0bea14368ef51162f68ed5c545633ee567f15); the site loads the same files from the repository root at `a2534a20253d54cce925307fbd76086498b09a11` | `self_twin/adapter/experiment-r8-luna-synthetic-1000-v3/adapter` | Mostly Luna-written synthetic examples. |
| v4 | `v4/` at [`5c03457`](https://huggingface.co/kobajin/kobajinAI/commit/5c03457fbc811484733abae9e1fa1b55f42e4059) | `self_twin/adapter/experiment-r8-kobajin-clean-v2-memory-lfm25-1.2b-jp/adapter` | Light version: LFM2.5-1.2B-JP trained like v2. 0.66 GB; shorter replies and less like kobajin than v2. Chosen first on low-memory devices. LFM Open License v1.0 (`v4/LICENSE`, `v4/NOTICE.md`). |

The same repository also has GGUF `Q4_K_M` files of every version in `gguf/` for llama.cpp; the site does not use them (see `self_twin/gguf/README.md`). The repository's `README.md` is the model card in `hf_model_card.md`; upload it again when a version is added.

Each merged checkpoint is in `self_twin/merged/` (`qwen35_4b_self_twin_*`, and `lfm25_1_2b_jp_self_twin_kobajin_clean_v2_20261122` for v4) with a `merge_receipt.json` (base revision, adapter hash). v1, v2 and v4 were merged with `training/merge_adapter.py --owner-release vN` (v4 with `--model-id LiquidAI/LFM2.5-1.2B-JP-202606`). Evaluation results are in `self_twin/training/training_report.md`.

## v3 release (2026-11-19)

The v3 bundle was merged from `self_twin/adapter/experiment-r8-luna-synthetic-1000-v3/adapter` into the pinned Qwen3.5-4B base at `self_twin/merged/qwen35_4b_self_twin_experimental_luna_synthetic_v3_20261119/`, then converted to WebLLM `q4f16_1` at `self_twin/webllm/hf-upload/luna-v3/`.

- 76 `params_shard_*.bin` files; quantized parameter size is about 2.205 GB
- `tensor-cache.json`
- `mlc-chat-config.json` with a 4,096-token context and 1,024-token prefill chunks
- `tokenizer.json` and `tokenizer_config.json`
- Hugging Face revision: [`a2534a20253d54cce925307fbd76086498b09a11`](https://huggingface.co/kobajin/kobajinAI/commit/a2534a20253d54cce925307fbd76086498b09a11)

`src/models.js` pins v3 to this immutable revision and uses a versioned WebLLM `model_id`. Never change a published record's URL: WebLLM caches weights by model URL, so replacing files behind `resolve/main/` can leave stale weights in a browser. `VITE_MODEL_URL` points the default version at a local bundle for testing; confirm it serves the intended bundle when using it.

The owner authorized this as an experimental replacement after same-prompt evaluation improved over the previous published candidate: real-conversation char ROUGE-L-F1 `0.0492 → 0.0999`, median reply length `5 → 34` characters, and held-out synthetic char ROUGE-L-F1 `0.0746 → 0.2436`. The unadapted base remains higher on real-conversation references (`0.1213`), so the quality gate is still marked not passed. This release is not described as better than the base model.

The MLC Qwen3.5 text loader intentionally ignores the model's vision weights. This site is text chat. The generated config uses the `qwen3_5` conversation template so its stop tokens match this tokenizer. The site reuses WebLLM's prebuilt Qwen3.5-4B WebGPU library, so a `.wasm` file does not need to be compiled or uploaded.

## Recreate the bundle

This machine needed the MLC nightly pair `mlc-llm-nightly-cpu==0.26.dev6` and `mlc-ai-nightly-cpu==0.26.dev246`, `apache-tvm-ffi==0.1.13`, and `psutil`. The package dylibs need local ad-hoc code signatures on this macOS build, and the published Qwen3.5 converter needs two small source workarounds in `mlc_llm/nn/rnn_state.py` and `mlc_llm/model/qwen35/qwen35_loader.py` before conversion. Those fixes are in `self_twin/mlc-venv/`.

Each `self_twin/webllm/mlc-input/<name>/` folder symlinks a merged checkpoint and adds `"context_window_size": 4096` to its `config.json` for the MLC runtime (`luna-v3`, `kobajin-clean-v1`, `kobajin-clean-v2`). For v3, from the repository root:

```sh
export MLC_LLM_HOME="$PWD/self_twin/mlc-cache"
self_twin/mlc-venv/bin/python -m mlc_llm convert_weight self_twin/webllm/mlc-input/luna-v3 \
  --quantization q4f16_1 --device cpu --output self_twin/webllm/hf-upload/luna-v3

self_twin/mlc-venv/bin/python -m mlc_llm gen_config self_twin/webllm/mlc-input/luna-v3 \
  --quantization q4f16_1 --conv-template qwen3_5 \
  --context-window-size 4096 --prefill-chunk-size 1024 \
  --output self_twin/webllm/hf-upload/luna-v3
```

## LFM2.5-1.2B-JP conversion (low-memory version, 2026-11-22)

The low-memory version is trained on `LiquidAI/LFM2.5-1.2B-JP-202606` (see `training/training_report.md`). The MLC nightly installed here (0.26.dev6) predates MLC's LFM2 support, so with the owner's approval (2026-11-22) the Python side of MLC commit [`87a90ce`](https://github.com/mlc-ai/mlc-llm/commit/87a90ce0028901e17ab50ff3ee3caea33c204a5f) was added to `self_twin/mlc-venv/`: `model/lfm2/` (model and loader), `conversation_template/lfm2.py`, the `lfm2` entries in `model/model.py`, `conversation_template/__init__.py` and `interface/gen_config.py`. Two local adjustments make it run on this older TVM: `lfm2_model.py` imports `tirx` from `tvm` instead of `tvm_tirx`, and `lfm2_loader.py` passes dtypes to NumPy as strings (as the Qwen3.5 loader does). The commit's C++ changes are runtime-only; WebLLM's prebuilt LFM2.5 library already has them.

A test conversion of the untrained base (`webllm/mlc-input/lfm25-jp-base`, `--quantization q4f16_1 --conv-template lfm2 --context-window-size 4096 --prefill-chunk-size 1024`) gives a 633 MB bundle (1.17 B parameters, 4.5 bits each). Its 185 tensors match the names, shapes and dtypes of MLC's published `mlc-ai/LFM2.5-1.2B-Instruct-q4f16_1-MLC`, so a fine-tuned JP bundle can reuse WebLLM's prebuilt `LFM2.5-1.2B-Instruct-q4f16_1` library (about 1.0 GB of VRAM by WebLLM's estimate). The remaining config differences are the context window (WebLLM overrides it to 4,096; the site uses 2,048) and `render_empty_system_message`, which does not matter because the site always sends a system prompt. 

### v4 release (2026-11-22)

v4 was merged into the pinned base at `self_twin/merged/lfm25_1_2b_jp_self_twin_kobajin_clean_v2_20261122/` and converted from `webllm/mlc-input/kobajin-clean-v2-lfm25/` to `webllm/hf-upload/kobajin-clean-v2-lfm25/` with the same commands as the base test (`webllm/build-v4.log`). The input folder links the merged `model.safetensors` but keeps the base model's own `config.json` (plus `context_window_size`), tokenizer files and chat template: Transformers 5 rewrites `config.json` on save with renamed fields (`tie_word_embeddings`, no `block_ff_dim` or `rope_theta`), and LoRA does not change the architecture. The bundle has the base test's 185 tensors with the same shapes, and the same `mlc-chat-config.json`.

- 27 `params_shard_*.bin` files, 0.66 GB; `tensor-cache.json`; `mlc-chat-config.json` (`lfm2` template, 4,096-token context, 1,024-token prefill chunks); the base model's `tokenizer.json` and `tokenizer_config.json`, unchanged
- `LICENSE`: the base model's LFM Open License v1.0, unchanged. `NOTICE.md`: states that kobajin changed the files (fine-tuned, merged, converted) and when, as section 4 of the license asks
- Hugging Face revision: [`5c03457fbc811484733abae9e1fa1b55f42e4059`](https://huggingface.co/kobajin/kobajinAI/commit/5c03457fbc811484733abae9e1fa1b55f42e4059), folder `v4/`; all 33 files checked against the local bundle by size and sha256 (`webllm/upload-v4.log`)

Parity on the 8 fixed prompts (`evaluation/webllm_parity/v4-local-greedy-compare.json`, local bundle with `cache=memory`): the prompt equals Transformers for 8/8, the greedy reply is identical for 4/8 (v2: 4/8), Transformers' top token is WebLLM's token for 94.5 % of positions (v2: 98.7 %), and the mean absolute log-probability difference is 0.20 (v2: 0.094). Running Transformers with the bundle's own dequantized 4-bit weights brings that difference to 0.015 (97.4 % top-1), so the conversion is right and the gap is the 4-bit rounding, which the 1.2B model feels more than the 4B. With the site's decoding (`v4-local-sampled-compare.json`), 6 of 12 WebLLM replies to 「好きな食べ物は何？」 used the memory candidate (Transformers 4/12; v2's WebLLM bundle 1/12).

The published files were then loaded on the dev site with a phone-sized viewport: the start screen chose v4 for the emulated phone, downloaded it from `5c03457…/v4/` with WebLLM's prebuilt LFM2.5 library, and wrote posts without console errors. Evaluation results are in `training/training_report.md`.

### Site support for other base models (2026-11-22)

Each record in `src/models.js` names a `family` (`qwen3.5-4b`, `qwen3.5-2b`, `lfm2.5-1.2b`). The family decides the prebuilt WebLLM library the version runs on, its download size and GPU memory (shown on the start screen), and the request options. Qwen3.5 versions keep the conv template override in `src/chat_request.js` and send `enable_thinking: false`. LFM2.5 versions keep the bundle's own `lfm2` template and send nothing extra: WebLLM inserts an empty `<think>` block for any model that gets `enable_thinking: false`, and LFM2.5 was trained without one. The start screen shows the chosen version's size and memory needs, suggests a light version to devices that report less than 8 GB (once one is published), and says up front when the GPU lacks `shader-f16`, which every q4f16_1 bundle needs (dev preview: `?no-f16`).

For local testing, `VITE_LOCAL_MODEL=<family>=<bundle URL>` (dev server only) adds a version named `local`, for example `VITE_LOCAL_MODEL=lfm2.5-1.2b=http://127.0.0.1:8001/resolve/local/` with `serve_local_bundle.py --bundle lfm25-jp-base-test`. Checked in the desktop app's browser pane with the untrained LFM2.5-JP bundle:

- The parity page (`?version=local&mode=greedy`, `evaluation/webllm_parity/lfm25-jp-base-local-greedy.json`): WebLLM's prompt equals the training template's rendering (`training/chat_templates/lfm2_5_jp_training.jinja`, BOS included) in text and token ids for 8/8 prompts.
- The site: the local version downloads (0.66 GB), comes online and writes posts and reactions, with no console errors.
- v2 after the change, through the parity page with the local v2 bundle and `cache=memory` (`v2-family-refactor-greedy.json`): the prompt still equals Transformers with `enable_thinking=False` for 8/8 prompts.

Not yet checked on a 4 GB tablet or a phone. The light version's memory figure is WebLLM's estimate.

## wllama (GGUF in the browser) considered, not adopted (2026-11-21)

[wllama](https://github.com/ngxson/wllama) 3.13.0 runs llama.cpp in the browser with WebGPU and loads `qwen35` GGUF, so the v2 `Q4_K_M` file (split into 512 MB parts with `llama-gguf-split`) works. It was compared with the site's WebLLM v2 bundle in the same Chromium browser pane on the same Mac, from a local server: greedy, no repetition penalty, 128 tokens, the 4 parity prompts with the site's system prompt (about 334 prompt tokens).

| | WebLLM `q4f16_1` (site) | wllama GGUF `Q4_K_M` |
|---|---|---|
| Download | 2.39 GB | 2.71 GB (+13 %) |
| Prompt processing | about 358 tokens/s | about 130 tokens/s |
| Time to first token | 0.93 s | 2.5 s |
| Generation | 30.9–32.5 tokens/s | 38.3–39.9 tokens/s |

The first WebLLM prompt (108 tokens/s, 3.1 s) is left out of the WebLLM column: it includes one-time GPU shader setup. wllama gave the same speeds without COOP/COEP headers (kobajin.com, served by Cloudflare, sends none), because all layers run on WebGPU. Its log showed flash attention disabled for lack of WebGPU support.

wllama generates faster but reads the prompt about 2.7 times slower. Site replies are short (kobajin's real replies have a median of 9.5 characters) and the site's prompts are longer than these (memories, conversation log), so a typical reply takes longer in wllama (about 2.8 s against 1.2 s for 10 tokens here), and the download is larger. The owner asked not to adopt it if it is heavier, so the site stays on WebLLM. This was one machine and one browser.

## Upload to Hugging Face

Upload each version's **contents** to its own folder of `kobajin/kobajinAI` (`v1/`, `v2/`, `v4/`, later `v5/`, …). Include `mlc-chat-config.json`, `tensor-cache.json`, every `params_shard_*.bin`, and both tokenizer files. A version trained on LFM2.5 also needs the base model's `LICENSE` and a `NOTICE.md` saying what was changed (LFM Open License v1.0, section 4). Exclude the optional `tensor-cache-b16.json`. Keep all shards together; do not substitute raw merged `.safetensors` files. The WebLLM model record fetches its prebuilt WebGPU library separately from MLC; `.wasm` is not part of this upload. v3 was uploaded to the repository root (`a2534a2`) and moved to `v3/` in `0bc0bea` on 2026-11-21; all 80 files are byte-identical (checked by size and hash). `src/models.js` still points v3 at the root of `a2534a2`, which stays downloadable because Hugging Face revisions are immutable; changing the URL would make browsers that cached v3 download it again.

```sh
self_twin/mlc-venv/bin/hf upload kobajin/kobajinAI self_twin/webllm/hf-upload/kobajin-clean-v2 v2 \
  --exclude tensor-cache-b16.json \
  --commit-message "Add v2 WebLLM bundle"
```

`hf upload` adds the selected folder's contents at the given repository path and leaves the other files in place, so earlier versions stay downloadable. Hugging Face's [upload guide](https://huggingface.co/docs/huggingface_hub/en/guides/cli) documents this syntax. After uploading, list the folder's files at the new revision, then add a record to `src/models.js` with that revision.

The repository is public, so the browser app can download anonymously. Do not put a Hugging Face write token in the website. The browser app includes a small curated public-memory file at `src/public_memories.json`; it does not bundle the private memory store or Discord history.

## WebLLM and Transformers parity

Evaluations run the merged BF16 checkpoint in Transformers; visitors run the q4f16_1 bundle in WebLLM. Three differences that made the site differ from what evaluation measured were found and fixed on 2026-11-21:

1. **Prompt.** The bundles' `mlc-chat-config.json` (MLC's `qwen3_5` template) has `"<|im_start|>assistant\n<think>"` as the assistant role. With `enable_thinking: false`, WebLLM appends its own `<think>\n\n</think>\n\n`, so every site prompt ended in `assistant\n<think>\n<think>\n\n</think>\n\n`, which the model never saw in training. `src/chat_request.js` now overrides the role to `<|im_start|>assistant` at load time (the published bundle files are unchanged), and the prompt is character for character the one Transformers builds with `enable_thinking=False`.
2. **Repetition penalty.** WebLLM penalizes only tokens generated in the current reply; Transformers also penalized every prompt token, including the memory candidates. `inference/generation.py` now does what WebLLM does.
3. **top-k.** Transformers silently sampled with `top_k=50`; WebLLM has no top-k. `generation.py` now disables it.

To measure the remaining difference, open the dev-only page `/self_twin/webllm/parity/` (`npm run dev`). It builds prompts with the site's own code and runs `parity/prompts.json` through the real WebLLM engine, then saves the run to `self_twin/evaluation/webllm_parity/`. `self_twin/evaluation/webllm_parity.py <run>.json` compares it with Transformers. Query options are documented at the top of `parity/parity.js`. A browser whose site storage can't hold the 2.2 GB bundle (the desktop app's browser pane stops at about 1.5 GB) can use `cache=memory` with a local bundle from `serve_local_bundle.py --bundle <folder>`.

v2, greedy decoding, 8 prompts, the local bundle files verified identical to `v2/` at `1f9287d`:

| | Prompt equals Transformers | Same greedy reply | Transformers' top token = WebLLM's token | Mean abs. log-prob difference |
|---|---|---|---|---|
| Bundle template (before) | 0/8 | 5/8 | 96.4 % | 0.093 |
| Site override (after) | 8/8 | 4/8 | 98.7 % | 0.094 |

The remaining difference is q4f16_1 rounding: WebLLM's first-token probabilities for 「好きな食べ物は何？」 match Transformers to within 0.001 (「う」 0.042 against 0.043), but most prompts here have a flat distribution, so small shifts change which reply is sampled. Greedy replies without repetition penalty loop, as they do in Transformers; that run is a measurement, not the site's decoding.

### Rounding into q4f16_1

`webllm/q4_rounding.py check` simulates q4f16_1 on the Transformers model and measures how far each way of rounding moves it from BF16. Its simulation of MLC's own rounding matches what WebLLM showed (top-1 agreement 96.3 % on the parity prompts against 96.4–98.7 % measured in the browser). v2, 60 held-out real replies from `kobajin_clean_v2_validation.jsonl` (734 token positions, teacher-forced) and the 8 parity prompts; GPTQ calibrated on 64 conversations from the training split (`evaluation/webllm_parity/v2-q4-rounding-check.json`):

| Rounding | Held-out KL from BF16 | Held-out top-1 agreement | Parity-prompt KL | Parity top-1 |
|---|---|---|---|---|
| float16, no quantization (earlier run) | — | — | 0.00014 | 99.5 % |
| MLC `convert_weight` (published bundles) | 0.079 | 80.8 % | 0.0177 | 96.3 % |
| max-abs scale shrunk for least squared error (earlier run) | — | — | 0.0192 | 96.7 % |
| GPTQ, output head by least squared error | **0.045** | **85.3 %** | **0.0107** | **97.7 %** |
| GPTQ including the output head | 0.045 | 85.1 % | 0.0107 | 97.7 % |

Almost all of the difference is the 4-bit rounding, not float16 arithmetic. GPTQ keeps the q4f16_1 layout, so the result still runs on WebLLM's prebuilt Qwen3.5-4B library: `q4_rounding.py export` writes a checkpoint whose weights already lie on the GPTQ grid, `mlc_llm convert_weight` turns it into a bundle as usual, and `q4_rounding.py verify` checks the bundle holds those weights.

A GPTQ v2 candidate was built this way (128 calibration conversations from the training split) at `merged/qwen35_4b_self_twin_kobajin_clean_v2_20261121_q4gptq/` and `webllm/hf-upload/kobajin-clean-v2-q4gptq/` (local only, not published). MLC's converter keeps GPTQ's levels except for float16 rounding of the scales, which adds about 1/1000 of the rounding error itself.

In the browser it was measured teacher-forced (`evaluation/webllm_forced.py`, the parity page's `mode=forced`): Transformers BF16 sampled 3 replies to each parity prompt, and both bundles scored the same 177 positions in WebLLM.

| 177 site-prompt positions | WebLLM top token = BF16 top token | Mean abs. log-prob difference | Same weights in Transformers: top token = BF16 |
|---|---|---|---|
| Published v2 bundle (MLC rounding) | 83.6 % | 0.206 | 85.9 % |
| GPTQ candidate | 83.1 % | 0.188 | 81.9 % |

WebLLM's own kernels add little: its top token equals that of Transformers running the same rounded weights at 94–96 % of these positions. But on the site's prompts GPTQ was not closer to BF16, while on the 60 held-out conversations above it was clearly closer. A larger held-out run in the site's exact format (731 positions, production system prompt; `--heldout` in `webllm_forced.py`, output git-ignored) was stopped before finishing because it made the machine too slow. Until that run decides it, the GPTQ candidate is not published and the site keeps the published bundles.

With the site's decoding (12 replies per parity prompt; `evaluation/webllm_parity/v2-site-sampled-compare.json`), WebLLM and Transformers behave alike. For the three memory questions, WebLLM's replies used the memory in 1 of 36 and Transformers' in 2 of 36. For 「好きな食べ物は何？」 both mostly name a made-up food (WebLLM: 納豆, 牛丼, spaghetti, …; Transformers: カレー, 卵かけご飯, …). Replies such as 「藕」 are therefore v2's own behavior on a bare question, not a WebLLM error: v2 learned to quote a memory that also appears in the real reply, and rarely does so for a question with no conversation around it.

## Runtime details

- Public read access is required for anonymous browser downloads. This static app intentionally has no token handling.
- For a first-time visitor nothing loads when the site opens. Only the version the visitor has selected is downloaded (once, into the browser Cache API) and loaded into memory, when the visitor starts it. After a version has come online once, later visits load it from the cache as soon as the page opens, until the visitor stops it. Switching versions unloads the previous one from memory; its weights stay cached. Already cached versions are marked "saved" in the picker. WebLLM downloads weights from Hugging Face and its prebuilt WebGPU library from MLC's static library repository.
- The website is static and can be hosted over HTTPS. Chat generation happens locally in WebGPU; the first download needs internet access. Browser storage can be evicted or cleared.
- The published v3 model is an owner-authorized experimental update over the prior HF candidate. Evaluation records still mark the base-model quality gate as not passed; the merge receipt retains that status.

References: [WebLLM custom models and caching](https://webllm.mlc.ai/docs/user/advanced_usage.html), [WebLLM Qwen3.5-4B model entry](https://webllm.mlc.ai/models/), [MLC weight conversion](https://llm.mlc.ai/docs/compilation/convert_weights.html).
