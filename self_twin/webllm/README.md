# WebLLM model bundle for `kobajin/kobajinAI`

The site lets visitors choose among every published kobajinAI version. `src/models.js` lists them. Each record points at the public `kobajin/kobajinAI` Hugging Face repository, pinned to an immutable revision and the version's folder, with a versioned WebLLM `model_id`. All versions are Qwen3.5-4B `q4f16_1` bundles and reuse WebLLM's built-in Qwen3.5-4B WebGPU library. The repository therefore needs MLC-converted weights, not the original Transformers `.safetensors` files alone.

The owner decided on 2026-11-21 to publish every trained version (see `AGENTS.md`). Evaluation decides each version's release notes and the site's default, not whether it is uploaded. The default is the version named in `src/closest_model.json`, the one currently judged closest to kobajin.

## Published versions

| Version | Folder at revision | Source adapter | Notes |
|---|---|---|---|
| v1 | `v1/` at [`8d6dab1`](https://huggingface.co/kobajin/kobajinAI/commit/8d6dab15195505c8c92c2d9ceb99f2551239f2d4) | `self_twin/adapter/experiment-r8-kobajin-clean-v1-lr5e5/adapter` | Real chats only. Natural voice; ignored retrieved memories (0/3) and overused 「うんこ」. |
| v2 | `v2/` at [`1f9287d`](https://huggingface.co/kobajin/kobajinAI/commit/1f9287d837eb311f5d0ab9262240ab2ed362d87c) | `self_twin/adapter/experiment-r8-kobajin-clean-v2-memory/adapter` | Real chats plus memory-grounded rows; 2/3 fixed memory questions answered with the memory. The site's default since 2026-11-21. |
| v3 | repository root at `a2534a20253d54cce925307fbd76086498b09a11` | `self_twin/adapter/experiment-r8-luna-synthetic-1000-v3/adapter` | Mostly Luna-written synthetic examples. |

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

Upload each version's **contents** to its own folder of `kobajin/kobajinAI` (`v1/`, `v2/`, later `v4/`, …). Include `mlc-chat-config.json`, `tensor-cache.json`, every `params_shard_*.bin`, and both tokenizer files. Exclude the optional `tensor-cache-b16.json`. Keep all shards together; do not substitute raw merged `.safetensors` files. The WebLLM model record fetches its prebuilt WebGPU library separately from MLC; `.wasm` is not part of this upload. v3 is at the repository root of revision `a2534a2` and stays there.

```sh
self_twin/mlc-venv/bin/hf upload kobajin/kobajinAI self_twin/webllm/hf-upload/kobajin-clean-v2 v2 \
  --exclude tensor-cache-b16.json \
  --commit-message "Add v2 WebLLM bundle"
```

`hf upload` adds the selected folder's contents at the given repository path and leaves the other files in place, so earlier versions stay downloadable. Hugging Face's [upload guide](https://huggingface.co/docs/huggingface_hub/en/guides/cli) documents this syntax. After uploading, list the folder's files at the new revision, then add a record to `src/models.js` with that revision.

The repository is public, so the browser app can download anonymously. Do not put a Hugging Face write token in the website. The browser app includes a small curated public-memory file at `src/public_memories.json`; it does not bundle the private memory store or Discord history.

## Runtime details

- Public read access is required for anonymous browser downloads. This static app intentionally has no token handling.
- Nothing loads when the site opens. Only the version the visitor has selected is downloaded (once, into the browser Cache API) and loaded into memory, when the visitor sends a message or presses Bring online. Switching versions unloads the previous one from memory; its weights stay cached. Already cached versions are marked "saved" in the picker. WebLLM downloads weights from Hugging Face and its prebuilt WebGPU library from MLC's static library repository.
- The website is static and can be hosted over HTTPS. Chat generation happens locally in WebGPU; the first download needs internet access. Browser storage can be evicted or cleared.
- The published v3 model is an owner-authorized experimental update over the prior HF candidate. Evaluation records still mark the base-model quality gate as not passed; the merge receipt retains that status.

References: [WebLLM custom models and caching](https://webllm.mlc.ai/docs/user/advanced_usage.html), [WebLLM Qwen3.5-4B model entry](https://webllm.mlc.ai/models/), [MLC weight conversion](https://llm.mlc.ai/docs/compilation/convert_weights.html).
