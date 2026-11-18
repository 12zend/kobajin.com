#!/usr/bin/env python3
"""Draft local inference wrapper with opt-in, intent-triggered memory retrieval.

Requires a Transformers v5 runtime and a locally saved merged checkpoint. This
script never sends prompts or memory records to a remote service.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from transformers import AutoModelForMultimodalLM, AutoProcessor

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_MODEL = PROJECT / "merged" / "qwen35_4b_self_twin"
SYSTEM_PROMPT = (HERE / "system_prompt.txt").read_text(encoding="utf-8").strip()
sys.path.insert(0, str(PROJECT / "memory"))
from retrieve_memory import retrieve, should_retrieve  # noqa: E402
from generation import generate_assistant_response  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--message", required=True)
    parser.add_argument("--history", type=Path, help="Optional JSON array of role/content messages")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Merged local model not found: {args.model}")

    history = json.loads(args.history.read_text(encoding="utf-8")) if args.history else []
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    last_user_text = args.message
    for item in history:
        if item.get("role") in {"user", "assistant"} and isinstance(item.get("content"), str):
            messages.append({"role": item["role"], "content": item["content"]})
            if item["role"] == "user":
                last_user_text = item["content"]

    # Retrieval must be driven by the new query, not the last user message in history.
    last_user_text = args.message

    if should_retrieve(last_user_text):
        memories = retrieve(last_user_text, limit=5)
        if memories:
            facts = "\n".join(
                f"- {item['content']} (出典message_id: {item['source_message_id']}; 確信度: {item['confidence']})"
                for item in memories
            )
            messages[0]["content"] += "\n\n今回の質問に関係する本人の過去の記憶候補:\n" + facts
        else:
            messages[0]["content"] += "\n\n今回の記憶検索では確認できる本人の記録が見つからなかった。過去の事実を推測で補わない。"

    messages.append({"role": "user", "content": args.message})

    import torch

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    response = generate_assistant_response(model, processor, inputs, args.max_new_tokens)
    print(response)


if __name__ == "__main__":
    main()
