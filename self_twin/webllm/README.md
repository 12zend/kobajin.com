# WebLLM model bundle for `kobajin/kobajinAI`

The site lets visitors choose among every published kobajinAI version. `src/models.js` lists them. Each record points at the public `kobajin/kobajinAI` Hugging Face repository, pinned to an immutable revision and the version's folder, with a versioned WebLLM `model_id`. All versions are Qwen3.5-4B `q4f16_1` bundles and reuse WebLLM's built-in Qwen3.5-4B WebGPU library. The repository therefore needs MLC-converted weights, not the original Transformers `.safetensors` files alone.

The owner decided on 2026-11-21 to publish every trained version (see `AGENTS.md`). Evaluation decides each version's release notes and the site's default, not whether it is uploaded. The default is the version named in `src/closest_model.json`, the one currently judged closest to kobajin.

## Published versions

| Version | Folder at revision | Source adapter | Notes |
|---|---|---|---|
| v1 | `v1/` at [`8d6dab1`](https://huggingface.co/kobajin/kobajinAI/commit/8d6dab15195505c8c92c2d9ceb99f2551239f2d4) | `self_twin/adapter/experiment-r8-kobajin-clean-v1-lr5e5/adapter` | Real chats only. Natural voice; ignored retrieved memories (0/3) and overused 「うんこ」. |
| v2 | `v2/` at [`1f9287d`](https://huggingface.co/kobajin/kobajinAI/commit/1f9287d837eb311f5d0ab9262240ab2ed362d87c) | `self_twin/adapter/experiment-r8-kobajin-clean-v2-memory/adapter` | Real chats plus memory-grounded rows; 2/3 fixed memory questions answered with the memory. The site's default since 2026-11-21. |
| v3 | `v3/` at [`0bc0bea`](https://huggingface.co/kobajin/kobajinAI/commit/0bc0bea14368ef51162f68ed5c545633ee567f15); the site loads the same files from the repository root at `a2534a20253d54cce925307fbd76086498b09a11` | `self_twin/adapter/experiment-r8-luna-synthetic-1000-v3/adapter` | Mostly Luna-written synthetic examples. |

The same repository also has GGUF `Q4_K_M` files of every version in `gguf/` for llama.cpp; the site does not use them (see `self_twin/gguf/README.md`). The repository's `README.md` is the model card in `hf_model_card.md`; upload it again when a version is added.

Each merged checkpoint is in `self_twin/merged/qwen35_4b_self_twin_*` with a `merge_receipt.json` (base revision, adapter hash). v1 and v2 were merged with `training/merge_adapter.py --owner-release vN`. Evaluation results are in `self_twin/training/training_report.md`.

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

## Upload to Hugging Face

Upload each version's **contents** to its own folder of `kobajin/kobajinAI` (`v1/`, `v2/`, later `v4/`, …). Include `mlc-chat-config.json`, `tensor-cache.json`, every `params_shard_*.bin`, and both tokenizer files. Exclude the optional `tensor-cache-b16.json`. Keep all shards together; do not substitute raw merged `.safetensors` files. The WebLLM model record fetches its prebuilt WebGPU library separately from MLC; `.wasm` is not part of this upload. v3 was uploaded to the repository root (`a2534a2`) and moved to `v3/` in `0bc0bea` on 2026-11-21; all 80 files are byte-identical (checked by size and hash). `src/models.js` still points v3 at the root of `a2534a2`, which stays downloadable because Hugging Face revisions are immutable; changing the URL would make browsers that cached v3 download it again.

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
- Nothing loads when the site opens. Only the version the visitor has selected is downloaded (once, into the browser Cache API) and loaded into memory, when the visitor sends a message or presses Bring online. Switching versions unloads the previous one from memory; its weights stay cached. Already cached versions are marked "saved" in the picker. WebLLM downloads weights from Hugging Face and its prebuilt WebGPU library from MLC's static library repository.
- The website is static and can be hosted over HTTPS. Chat generation happens locally in WebGPU; the first download needs internet access. Browser storage can be evicted or cleared.
- The published v3 model is an owner-authorized experimental update over the prior HF candidate. Evaluation records still mark the base-model quality gate as not passed; the merge receipt retains that status.

References: [WebLLM custom models and caching](https://webllm.mlc.ai/docs/user/advanced_usage.html), [WebLLM Qwen3.5-4B model entry](https://webllm.mlc.ai/models/), [MLC weight conversion](https://llm.mlc.ai/docs/compilation/convert_weights.html).
