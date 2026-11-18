#!/usr/bin/env python3
"""Merge only an explicitly evaluated and approved local adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoProcessor

PROJECT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3.5-4B"
CACHE_DIR = PROJECT / "hf_home" / "hub"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--decision", type=Path, default=PROJECT / "evaluation" / "decision.json")
    parser.add_argument("--output", type=Path, default=PROJECT / "merged" / "qwen35_4b_self_twin")
    args = parser.parse_args()

    decision = json.loads(args.decision.read_text(encoding="utf-8"))
    if decision.get("status") != "passed":
        raise SystemExit("Evaluation decision is not passed; refusing to merge")
    decided_adapter = Path(decision.get("adapter_path", "")).resolve()
    if args.adapter.resolve() != decided_adapter:
        raise SystemExit("Requested adapter does not match the adapter that passed evaluation")
    weight_path = args.adapter / "adapter_model.safetensors"
    if not weight_path.is_file():
        raise SystemExit(f"Adapter weights not found: {weight_path}")

    receipt_path = PROJECT / "training" / "base_checkpoint.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("repo_id") != MODEL_ID or receipt.get("quantization") is not None:
        raise SystemExit("Pinned official post-trained base receipt is missing or quantized")
    revision = receipt["revision"]
    expected_revision = decision.get("base_revision")
    if expected_revision != revision:
        raise SystemExit("Evaluation decision does not match the pinned base revision")

    if args.output.exists():
        existing = [item for item in args.output.iterdir() if item.name != "README.md"]
        if existing:
            raise SystemExit(f"Refusing to overwrite an existing model directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    base = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID,
        revision=revision,
        cache_dir=CACHE_DIR,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    adapted = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    merged = adapted.merge_and_unload()
    merged.config.use_cache = True
    merged.save_pretrained(args.output, safe_serialization=True, max_shard_size="5GB")
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        revision=revision,
        cache_dir=CACHE_DIR,
        local_files_only=True,
    )
    processor.save_pretrained(args.output)

    merge_receipt = {
        "base_repo": MODEL_ID,
        "base_revision": revision,
        "adapter_path": str(args.adapter.resolve()),
        "adapter_sha256": sha256_file(weight_path),
        "evaluation_decision": str(args.decision.resolve()),
        "merge_method": "PEFT PeftModel.merge_and_unload()",
        "dtype": "bfloat16",
        "format": "Transformers Safetensors",
        "base_preserved": True,
        "tokenizer_and_chat_template_saved": True,
    }
    (args.output / "merge_receipt.json").write_text(
        json.dumps(merge_receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(merge_receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
