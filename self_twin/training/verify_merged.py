#!/usr/bin/env python3
"""Reload the merged checkpoint and compare it with the same live LoRA adapter."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoProcessor

PROJECT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3.5-4B"
CACHE_DIR = PROJECT / "hf_home" / "hub"


def generate(model, processor, messages, device):
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=96, do_sample=False, use_cache=True)
    return processor.batch_decode(output[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)[0].strip()


def load_prompts(path: Path, count: int) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()][:count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--merged", type=Path, default=PROJECT / "merged" / "qwen35_4b_self_twin")
    parser.add_argument("--prompts", type=Path, default=PROJECT / "evaluation" / "evaluation_prompts.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT / "merged" / "qwen35_4b_self_twin" / "merge_consistency.json")
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()

    receipt = json.loads((args.merged / "merge_receipt.json").read_text(encoding="utf-8"))
    revision = receipt["base_revision"]
    prompts = load_prompts(args.prompts, args.count)
    if not prompts:
        raise SystemExit("No verification prompts found")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("No CUDA or MPS device is available; refusing 4B inference on CPU")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID, revision=revision, cache_dir=CACHE_DIR, local_files_only=True
    )
    base = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID,
        revision=revision,
        cache_dir=CACHE_DIR,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device)
    adapted = PeftModel.from_pretrained(base, args.adapter, is_trainable=False).eval()
    adapter_outputs = [generate(adapted, processor, row["prompt_messages"], device) for row in prompts]
    del adapted, base
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    reloaded_processor = AutoProcessor.from_pretrained(args.merged, local_files_only=True)
    merged = AutoModelForMultimodalLM.from_pretrained(
        args.merged,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device).eval()
    merged_outputs = [generate(merged, reloaded_processor, row["prompt_messages"], device) for row in prompts]

    comparisons = [
        {
            "index": index,
            "category": row["category"],
            "adapter_response": adapter_response,
            "merged_response": merged_response,
            "exact_match": adapter_response == merged_response,
        }
        for index, (row, adapter_response, merged_response) in enumerate(
            zip(prompts, adapter_outputs, merged_outputs, strict=True)
        )
    ]
    report = {
        "status": "verified" if all(row["exact_match"] for row in comparisons) else "differences_observed",
        "base_revision": revision,
        "device": device,
        "prompt_count": len(comparisons),
        "exact_match_count": sum(row["exact_match"] for row in comparisons),
        "comparisons": comparisons,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "comparisons"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
