# WebLLM model bundle for `kobajin/kobajinAI`

`src/main.js` uses WebLLM's built-in Qwen3.5-4B WebGPU library and points the model record at the public `kobajin/kobajinAI` Hugging Face repository. The repository therefore needs MLC-converted weights, not the original Transformers `.safetensors` files alone.

## Current experimental release

The currently published bundle was merged from `self_twin/adapter/experiment-r8-luna-synthetic-1000-v3/adapter` into the pinned Qwen3.5-4B base at `self_twin/merged/qwen35_4b_self_twin_experimental_luna_synthetic_v3_20261119/`, then converted to WebLLM `q4f16_1` at `self_twin/webllm/hf-upload/luna-v3/`.

- 76 `params_shard_*.bin` files; quantized parameter size is about 2.205 GB
- `tensor-cache.json`
- `mlc-chat-config.json` with a 4,096-token context and 1,024-token prefill chunks
- `tokenizer.json` and `tokenizer_config.json`
- Hugging Face revision: [`a2534a20253d54cce925307fbd76086498b09a11`](https://huggingface.co/kobajin/kobajinAI/commit/a2534a20253d54cce925307fbd76086498b09a11)

The production default in `src/main.js` pins the model URL to this immutable revision and uses a versioned WebLLM `model_id`. Keep both in sync when publishing a new model: WebLLM caches weights by model URL, so replacing files behind `resolve/main/` can leave stale weights in a browser. `VITE_MODEL_URL` is a local bundle testing override; confirm it serves the intended bundle when using it.

The owner authorized this as an experimental replacement after same-prompt evaluation improved over the previous published candidate: real-conversation char ROUGE-L-F1 `0.0492 → 0.0999`, median reply length `5 → 34` characters, and held-out synthetic char ROUGE-L-F1 `0.0746 → 0.2436`. The unadapted base remains higher on real-conversation references (`0.1213`), so the quality gate is still marked not passed. This release is not described as better than the base model.

The MLC Qwen3.5 text loader intentionally ignores the model's vision weights. This site is text chat. The generated config uses the `qwen3_5` conversation template so its stop tokens match this tokenizer. The site reuses WebLLM's prebuilt Qwen3.5-4B WebGPU library, so a `.wasm` file does not need to be compiled or uploaded.

## Recreate the bundle

This machine needed the MLC nightly pair `mlc-llm-nightly-cpu==0.26.dev6` and `mlc-ai-nightly-cpu==0.26.dev246`, `apache-tvm-ffi==0.1.13`, and `psutil`. The package dylibs need local ad-hoc code signatures on this macOS build, and the published Qwen3.5 converter needs two small source workarounds in `mlc_llm/nn/rnn_state.py` and `mlc_llm/model/qwen35/qwen35_loader.py` before conversion. Those fixes are in `self_twin/mlc-venv/`.

`self_twin/webllm/mlc-input/luna-v3/` points at the v3 merged safetensors and supplies the MLC runtime's 4,096-token context value. From the repository root:

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

Upload the **contents** of `self_twin/webllm/hf-upload/luna-v3/` to the **root** of `kobajin/kobajinAI`. Include `mlc-chat-config.json`, `tensor-cache.json`, every `params_shard_*.bin`, and both tokenizer files. Exclude the optional `tensor-cache-b16.json`. Keep all shards together; do not substitute raw merged `.safetensors` files. The WebLLM model record fetches its prebuilt WebGPU library separately from MLC; `.wasm` is not part of this upload.

The v3 bundle was uploaded in commit `a2534a20253d54cce925307fbd76086498b09a11`. To publish a later evaluated candidate:

```sh
self_twin/mlc-venv/bin/hf upload kobajin/kobajinAI self_twin/webllm/hf-upload/luna-v3 . \
  --exclude tensor-cache-b16.json \
  --commit-message "Replace WebLLM bundle with evaluated candidate"
```

`hf upload` adds the selected folder's contents at repository path `.` and leaves unrelated files in the repository in place. Hugging Face's [upload guide](https://huggingface.co/docs/huggingface_hub/en/guides/cli) documents this syntax.

The repository is public, so the browser app can download anonymously. Do not put a Hugging Face write token in the website. The browser app includes a small curated public-memory file at `src/public_memories.json`; it does not bundle the private memory store or Discord history.

## Runtime details

- Public read access is required for anonymous browser downloads. This static app intentionally has no token handling.
- The site starts loading the configured model when it opens. WebLLM downloads weights from Hugging Face and its prebuilt WebGPU library from MLC's static library repository, then caches those artifacts with the browser Cache API.
- The website is static and can be hosted over HTTPS. Chat generation happens locally in WebGPU; the first download needs internet access. Browser storage can be evicted or cleared.
- The published v3 model is an owner-authorized experimental update over the prior HF candidate. Evaluation records still mark the base-model quality gate as not passed; the merge receipt retains that status.

References: [WebLLM custom models and caching](https://webllm.mlc.ai/docs/user/advanced_usage.html), [WebLLM Qwen3.5-4B model entry](https://webllm.mlc.ai/models/), [MLC weight conversion](https://llm.mlc.ai/docs/compilation/convert_weights.html).
