# GGUF Q4_K_M for `kobajin/kobajinAI`

kobajin.com loads the WebLLM `q4f16_1` bundles (see `../webllm/README.md`). The GGUF files are an extra for running kobajinAI locally with llama.cpp or other GGUF runtimes. The site does not use them, and they do not change any WebLLM bundle or pinned revision.

The Hugging Face model card is `../webllm/hf_model_card.md`, uploaded as the repository's `README.md`.

## Build

```bash
self_twin/gguf/build_gguf.sh v1 v2 v3
```

The script converts each version's merged BF16 checkpoint (`self_twin/merged/qwen35_4b_self_twin_*`) to BF16 GGUF with llama.cpp's `convert_hf_to_gguf.py`, quantizes it with `llama-quantize Q4_K_M` (no importance matrix), deletes the BF16 GGUF, and prints the sha256. `LLAMA_CPP` must point at a llama.cpp checkout at the same commit as the installed llama.cpp tools; the 2026-11-21 build used `d81235049` for both. Output goes to `build/`, which Git ignores.

`--no-mtp` is required. The merged checkpoints' `config.json` keeps `mtp_num_hidden_layers: 1`, but Transformers drops the base model's `mtp.*` tensors when it loads and saves the model, so without the flag the converter declares 33 blocks and llama.cpp fails with `tensor 'blk.32.attn_norm.weight' not found`. The GGUF files therefore have no MTP head. The vision encoder is not included either.

## 2026-11-21 release

| File | Source checkpoint | sha256 |
|---|---|---|
| `gguf/kobajinAI-v1-Q4_K_M.gguf` | `qwen35_4b_self_twin_kobajin_clean_v1_20261121` | `7baa7ae5f6c7a36147579e0c9ab54e0682641e12b0aad649d523ff81d597341d` |
| `gguf/kobajinAI-v2-Q4_K_M.gguf` | `qwen35_4b_self_twin_kobajin_clean_v2_20261121` | `8d9bb50725daa65f44c4cb0a66dd7833693de99187b579a0193a676d839e3a34` |
| `gguf/kobajinAI-v3-Q4_K_M.gguf` | `qwen35_4b_self_twin_experimental_luna_synthetic_v3_20261119` | `eb893d394dda9ef17f5d984d498597d12b6e12a309f83629456aef0d9a49c430` |

Each file is 2,708,804,032 bytes (2,572.86 MiB, 5.13 bits per weight; BF16 was 8,023.67 MiB). Hugging Face revision: GGUF files added in [`a64b0e7`](https://huggingface.co/kobajin/kobajinAI/commit/a64b0e78e9a80d5ae17677b49ac4677ab4d0d82b); model card metadata (base model, tags) updated in [`46b03fa`](https://huggingface.co/kobajin/kobajinAI/commit/46b03fab0c85e1948ce39514f903ec84bf212166). The WebLLM bundles at `8d6dab1` (`v1/`), `1f9287d` (`v2/`), and `a2534a2` (root) were checked file by file and are unchanged at `46b03fa`.

## Check against Transformers

`gguf_parity.py` runs the 8 prompts of a WebLLM parity run through `llama-server` (raw Transformers prompt, greedy, no repetition penalty, 60 tokens) and compares the replies with that run's Transformers greedy replies.

```bash
llama-server -m self_twin/gguf/build/kobajinAI-v2-Q4_K_M.gguf --port 18089 -c 4096 -ngl 99 --jinja
self_twin/mlc-venv/bin/python -I self_twin/gguf/gguf_parity.py \
  self_twin/evaluation/webllm_parity/v2-site-greedy.json \
  self_twin/evaluation/webllm_parity/v2-site-greedy-compare.json \
  self_twin/merged/qwen35_4b_self_twin_kobajin_clean_v2_20261121 18089
```

v2 result (`../evaluation/gguf_parity/v2-q4_k_m-greedy-compare.json`):

| | Prompt equals Transformers | Same greedy reply |
|---|---|---|
| WebLLM `q4f16_1` (site) | 8/8 | 4/8 |
| GGUF `Q4_K_M` | 8/8 (llama.cpp's own template) | 6/8 |

The first-token probability of 「う」 for 「好きな食べ物は何？」 is 0.042 in GGUF, against 0.043 in Transformers. The two differing replies are 「明日テストなんだけど何から勉強すればいい？」 (GGUF 「えぇ\n何のテスト？」, Transformers 「えぇ\n僕も明日テスト」) and 「最近寒すぎない？」 (GGUF 「うん」, Transformers 「えぐい」 repeated). This is 8 prompts, so it shows the conversion is sound, not that Q4_K_M is generally closer than `q4f16_1`.

v1 and v3 were only checked to load and reply with the site's decoding (temperature 0.72, top-p 1.0, repetition penalty 1.12); their replies matched each version's known character (v1 short and casual, v3 longer and advice-like).
