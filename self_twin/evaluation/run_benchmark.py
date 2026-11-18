#!/usr/bin/env python3
"""Run a fixed held-out prompt set against local base, LoRA, and LoRA+RAG."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoProcessor

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / "memory"))
from retrieve_memory import retrieve, should_retrieve  # noqa: E402
sys.path.insert(0, str(PROJECT / "inference"))
from generation import generate_assistant_response  # noqa: E402

MODEL_ID = "Qwen/Qwen3.5-4B"
DEFAULT_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
DEFAULT_CACHE = PROJECT / "hf_home" / "hub"


def load_prompts(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def get_query_text(messages: list[dict]) -> str:
    if not messages:
        return ""
    lines = str(messages[-1].get("content", "")).splitlines()
    return lines[-1] if lines else ""


def with_memory(messages: list[dict], query: str) -> tuple[list[dict], list[dict]]:
    result = copy.deepcopy(messages)
    matches = retrieve(query, limit=5) if should_retrieve(query) else []
    if matches:
        lines = []
        for item in matches:
            freshness = "古くなっている可能性あり" if item["could_be_stale"] else ""
            lines.append(
                f"- {item['content']} (出典message_id: {item['source_message_id']}; "
                f"確信度: {item['confidence']}; {freshness})"
            )
        result[0]["content"] += "\n\n今回の質問に関連する本人の記憶候補:\n" + "\n".join(lines)
    elif should_retrieve(query):
        result[0]["content"] += (
            "\n\n今回の記憶検索では確認できる本人の記録が見つからなかった。"
            "過去の事実を推測で補わない。"
        )
    return result, matches


def persist_results(path: Path, rows: list[dict]) -> None:
    """Keep completed generations if a long local benchmark is interrupted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def generate_one(model, processor, messages: list[dict], device: str, max_new_tokens: int) -> str:
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    return generate_assistant_response(model, processor, inputs, max_new_tokens)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, default=HERE / "evaluation_prompts.jsonl")
    parser.add_argument(
        "--base-results",
        type=Path,
        help="Reuse base responses from an earlier run with the exact same prompts and decoding settings.",
    )
    parser.add_argument("--output", type=Path, default=HERE / "generations.jsonl")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    if not args.adapter.is_dir():
        raise SystemExit(f"LoRA adapter directory does not exist: {args.adapter}")
    prompts = load_prompts(args.prompts)
    if not prompts:
        raise SystemExit("Evaluation prompt set is empty")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("No CUDA or MPS device is available; refusing 4B evaluation on CPU")
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        revision=args.revision,
        cache_dir=DEFAULT_CACHE,
        local_files_only=True,
    )
    if processor.tokenizer.eos_token_id is None:
        raise SystemExit("Tokenizer has no EOS token; refusing to generate unbounded chat turns")
    generation_settings = {
        "model_id": MODEL_ID,
        "revision": args.revision,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "enable_thinking": False,
        "eos_token_id": processor.tokenizer.eos_token_id,
        "stop_before_next_chat_turn": True,
    }
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID,
        revision=args.revision,
        cache_dir=DEFAULT_CACHE,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device)
    model.eval()

    reused_base = None
    if args.base_results:
        reused_base = load_prompts(args.base_results)
        if len(reused_base) != len(prompts):
            raise SystemExit("Base result count does not match the evaluation prompt count")
        for index, (prompt, previous) in enumerate(zip(prompts, reused_base)):
            if (
                previous.get("index") != index
                or previous.get("category") != prompt["category"]
                or previous.get("target_message_ids", []) != prompt.get("target_message_ids", [])
                or not isinstance(previous.get("base"), str)
                or previous.get("generation_settings") != generation_settings
            ):
                raise SystemExit(f"Base result row {index} does not match the current prompt set")

    results = []
    if reused_base is None:
        print(f"Generating base responses: {len(prompts)} prompts", flush=True)
    else:
        print(f"Reusing {len(prompts)} base responses from {args.base_results}", flush=True)
    for index, prompt in enumerate(prompts):
        base_response = (
            reused_base[index]["base"]
            if reused_base is not None
            else generate_one(model, processor, prompt["prompt_messages"], device, args.max_new_tokens)
        )
        results.append({
            "index": index,
            "category": prompt["category"],
            "target_message_ids": prompt.get("target_message_ids", []),
            "context_message_ids": prompt.get("context_message_ids", []),
            "reference_kind": prompt.get("reference_kind", "held_out_reply"),
            "reference_response": prompt.get("reference_response"),
            "expected_fact_terms": prompt.get("expected_fact_terms", []),
            "generation_settings": generation_settings,
            "base": base_response,
        })
        if reused_base is None:
            persist_results(args.output, results)
            print(f"base {index + 1}/{len(prompts)}", flush=True)

    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()
    print(f"Generating LoRA and RAG responses: {len(prompts)} prompts", flush=True)
    for index, prompt in enumerate(prompts):
        messages = prompt["prompt_messages"]
        query = get_query_text(messages)
        response = generate_one(model, processor, messages, device, args.max_new_tokens)
        rag_messages, matches = with_memory(messages, query)
        # The RAG branch is identical when the query does not ask for memory.
        # Reuse that output; run a separate generation only for retrieved facts
        # or an attempted lookup that needs an explicit unknown-memory response.
        rag_response = (
            generate_one(model, processor, rag_messages, device, args.max_new_tokens)
            if matches or should_retrieve(query)
            else response
        )
        results[index]["lora"] = response
        results[index]["lora_rag"] = rag_response
        results[index]["rag_memory_ids"] = [item["memory_id"] for item in matches]
        results[index]["rag_source_message_ids"] = [item["source_message_id"] for item in matches]
        persist_results(args.output, results)
        print(f"LoRA/RAG {index + 1}/{len(prompts)}", flush=True)

    print(f"Saved {len(results)} comparisons to {args.output}")


if __name__ == "__main__":
    main()
