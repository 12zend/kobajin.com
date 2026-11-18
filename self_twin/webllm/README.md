# WebLLM model bundle for `kobajin/kobajinAI`

`src/main.js` uses WebLLM's built-in Qwen3.5-4B WebGPU library and points the model record at the public `kobajin/kobajinAI` Hugging Face repository. The repository therefore needs MLC-converted weights, not the original Transformers `.safetensors` files alone.

## Converted bundle

The LoRA was merged into a separate BF16 Transformers checkpoint at
`self_twin/merged/qwen35_4b_self_twin_experimental_20261118/`. The WebLLM bundle is now converted at
`self_twin/webllm/hf-upload/` with MLC `q4f16_1`:

- 76 `params_shard_*.bin` files, about 2.2 GB total
- `tensor-cache.json`
- `mlc-chat-config.json` with a 4,096-token context and 1,024-token prefill chunks
- `tokenizer.json` and `tokenizer_config.json`

The MLC Qwen3.5 text loader intentionally ignores the model's vision weights. This site is text chat.
The generated config uses the `qwen3_5` conversation template so its stop tokens match this tokenizer.
The site reuses WebLLM's prebuilt Qwen3.5-4B WebGPU library, so a `.wasm` file does not need to be
compiled or uploaded.

## Recreate the bundle

This machine needed the MLC nightly pair `mlc-llm-nightly-cpu==0.26.dev6` and
`mlc-ai-nightly-cpu==0.26.dev246`, `apache-tvm-ffi==0.1.13`, and `psutil`. The package dylibs need
local ad-hoc code signatures on this macOS build, and the published Qwen3.5 converter needs two small
source workarounds in `mlc_llm/nn/rnn_state.py` and `mlc_llm/model/qwen35/qwen35_loader.py` before
conversion. Those fixes were applied in `self_twin/mlc-venv/` for this output.

The MLC input in `self_twin/webllm/mlc-input/` points at the merged safetensors and supplies the MLC
runtime's 4,096-token context value. From the repository root, the commands are:

```sh
export MLC_LLM_HOME="$PWD/self_twin/mlc-cache"
self_twin/mlc-venv/bin/python -m mlc_llm convert_weight self_twin/webllm/mlc-input \
  --quantization q4f16_1 --device cpu --output self_twin/webllm/hf-upload

self_twin/mlc-venv/bin/python -m mlc_llm gen_config self_twin/webllm/mlc-input \
  --quantization q4f16_1 --conv-template qwen3_5 \
  --context-window-size 4096 --prefill-chunk-size 1024 \
  --output self_twin/webllm/hf-upload
```

## Upload to Hugging Face

Upload the **contents** of `self_twin/webllm/hf-upload/` to the **root** of `kobajin/kobajinAI`:

- `mlc-chat-config.json`
- `tensor-cache.json`
- every `params_shard_*.bin` file
- each tokenizer file listed in `tokenizer_files` inside this bundle's `mlc-chat-config.json`: `tokenizer.json` and `tokenizer_config.json`

Keep every shard together. Do not upload a partial set. Do not upload the raw merged `.safetensors` as a substitute for these runtime files. The WebLLM model record reuses its prebuilt Qwen3.5-4B WebGPU library, which WebLLM fetches separately from MLC's static library repository; the `.wasm` file is not part of this upload.

### CLI (recommended)

From the repository root, log in and upload the folder contents to the model repository root:

```sh
self_twin/mlc-venv/bin/hf upload kobajin/kobajinAI self_twin/webllm/hf-upload . \
  --commit-message "Add WebLLM q4f16_1 model"
```

`hf upload` adds the files in that local folder at the repository path `.` and leaves unrelated files in the repository in place. Hugging Face's [upload guide](https://huggingface.co/docs/huggingface_hub/en/guides/cli) documents this syntax.

### Web interface

Open `kobajin/kobajinAI` on Hugging Face, go to **Files and versions → Add file → Upload files**, select the files listed above from `self_twin/webllm/hf-upload/`, and commit them at the repository root. Hugging Face documents this workflow in its [model upload guide](https://huggingface.co/docs/hub/en/models-uploading).

The repository is currently public, so the browser app can download anonymously. Do not put a write token in the website. The browser app includes a small curated public-memory file at `src/public_memories.json`; it does not bundle the full private memory store or Discord history.

The original Transformers checkpoint can also be uploaded separately if you want a standard Transformers model card on Hugging Face. That upload typically includes `config.json`, `model.safetensors.index.json` (when sharded), all `model-*.safetensors`, and the tokenizer / processor / chat-template files emitted by `save_pretrained()`. WebLLM does not fetch those files.

## Runtime details

- Public read access is required for anonymous browser downloads. A private repository would require authenticated downloads and this static app intentionally has no token handling.
- The site starts loading the configured model when it opens. WebLLM downloads the weights from Hugging Face and its prebuilt WebGPU library from MLC's static library repository, then caches those artifacts with the browser Cache API. On later opens it reuses the cache. The WebLLM record keeps the Qwen3.5-specific 4K context and recurrent-state settings from the built-in runtime entry.
- The website itself can be hosted as static files over HTTPS. Chat generation happens locally in WebGPU; the model and UI require network access only for the first download. Browser storage can be evicted by the browser or cleared by the user.
- The chosen LoRA candidate is explicitly recorded as an experimental merge because the current owner evaluation did not pass the quality gate. The merged folder's receipt records this status; it is not a claim that the LoRA improved the base model.

References: [WebLLM custom models and caching](https://webllm.mlc.ai/docs/user/advanced_usage.html), [WebLLM Qwen3.5-4B model entry](https://webllm.mlc.ai/models/), [MLC weight conversion](https://llm.mlc.ai/docs/compilation/convert_weights.html).
