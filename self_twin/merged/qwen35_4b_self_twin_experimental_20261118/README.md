# Experimental merge — 2026-11-18

This folder contains a new BF16 merge of `Qwen/Qwen3.5-4B` revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` with the rank 8 / balanced 5,000-example LoRA candidate.

The source checkpoint and adapter remain unchanged. The evaluation decision is still `not_passed`; this artifact was created at the owner's request for experimentation. See `merge_receipt.json` for the adapter hash and recorded gate status.

The large `.safetensors` shards and tokenizer JSON are local artifacts and are excluded from Git. For the browser app, convert this checkpoint to WebLLM's MLC `q4f16_1` format and upload those runtime files as described in `../../webllm/README.md`.
