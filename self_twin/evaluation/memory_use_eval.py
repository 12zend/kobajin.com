#!/usr/bin/env python3
"""Held-out check of whether a model uses retrieved memories, and only when relevant.

The fixed benchmark has only three memory questions. This check builds a larger
set from the chronological test split, which no candidate saw during training.
It uses the same construction and memory format as kobajin_clean_v2:

- relevant: a real test conversation whose real reply contains a self-statement.
  The memory block quotes that statement next to one unrelated memory. A reply
  "uses" the memory when it shares a span of at least --min-span characters
  with the relevant quote.
- distractor: a real test conversation, plus memory candidates quoted from other
  conversations. Any shared span with an injected memory counts as a false use.

Prompts and decoding match the site: the production system prompt followed by
the getMemoryContext() block, temperature 0.72, top_p 1.0, repetition penalty
1.12. The outputs contain private conversation text, so --generations must stay
in a gitignored path (generations-*.jsonl); --output holds aggregates only.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoProcessor

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / "inference"))
sys.path.insert(0, str(PROJECT / "training"))
sys.path.insert(0, str(HERE))
from build_kobajin_train_v2 import (  # noqa: E402
    EVAL_MEMORY_TERMS,
    PARTICIPANT,
    SELF_STATEMENT,
    clean_target,
    memory_block,
    memory_line,
)
from generation import generate_assistant_response  # noqa: E402
from kobajin_likeness import CACHE, MODEL_ID, PROD_PROMPT, REVISION  # noqa: E402
from train import fit_context_to_length  # noqa: E402

import re  # noqa: E402


def normalize(text: str) -> str:
    return re.sub(r"[\s　。、！？!?「」『』()（）\[\]【】]", "", text).lower()


def longest_shared_span(left: str, right: str) -> int:
    a, b = normalize(left), normalize(right)
    best = 0
    previous = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        current = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                current[j] = previous[j - 1] + 1
                best = max(best, current[j])
        previous = current
    return best


def self_statement(target: str) -> str | None:
    if len(target) > 120 or "【" in target or any(term in target.lower() for term in EVAL_MEMORY_TERMS):
        return None
    for line in target.split("\n"):
        line = line.strip()
        if SELF_STATEMENT.search(line) and len(line) >= 6 and not re.search(r"[？?]\s*$", line):
            return line
    return None


def build_items(test_path: Path, count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for line in test_path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        target = row["messages"][-1]["content"]
        if PARTICIPANT in target:
            continue
        target = clean_target(target)
        if target:
            row["messages"][-1]["content"] = target
            rows.append(row)
    statements = [(row, quote) for row in rows if (quote := self_statement(row["messages"][-1]["content"]))]
    rng.shuffle(statements)
    statements = statements[:count]
    plain = [row for row in rows if not self_statement(row["messages"][-1]["content"])]
    items = []
    for index, (row, quote) in enumerate(statements):
        other_row, other_quote = rng.choice([item for item in statements if item[1] != quote])
        lines = [memory_line(quote, row, rng), memory_line(other_quote, other_row, rng)]
        rng.shuffle(lines)
        items.append({"kind": "relevant", "row": row, "quotes": [quote], "injected": [quote, other_quote], "lines": lines})
    for row in rng.sample(plain, count):
        picked = rng.sample(statements, rng.randint(1, 2))
        lines = [memory_line(q, r, rng) for r, q in picked]
        quotes = [q for _, q in picked]
        items.append({"kind": "distractor", "row": row, "quotes": quotes, "injected": quotes, "lines": lines})
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", action="append", default=[], help="name=path; base is always included")
    parser.add_argument("--test", type=Path, default=PROJECT / "dataset" / "test.jsonl")
    parser.add_argument("--items", type=int, default=40, help="items per kind")
    parser.add_argument("--min-span", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generations", type=Path, required=True)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cuda"
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION, cache_dir=CACHE, local_files_only=True)
    items = build_items(args.test, args.items, args.seed)
    prompts = []
    for item in items:
        system = f"{PROD_PROMPT}\n\n{memory_block(item['lines'])}"
        messages = [{"role": "system", "content": system}, *item["row"]["messages"][1:]]
        fitted, _ = fit_context_to_length(processor, messages, args.max_length)
        prompts.append(fitted[:-1] if fitted else None)

    adapters = [tuple(item.split("=", 1)) for item in args.adapter]
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, revision=REVISION, cache_dir=CACHE, dtype=torch.bfloat16, low_cpu_mem_usage=True, local_files_only=True
    ).to(device)
    if adapters:
        model = PeftModel.from_pretrained(model, adapters[0][1], adapter_name=adapters[0][0], is_trainable=False)
        for name, path in adapters[1:]:
            model.load_adapter(path, adapter_name=name, is_trainable=False)
    model.eval()

    report = {
        "seed": args.seed,
        "items_per_kind": args.items,
        "min_shared_span": args.min_span,
        "decoding": {"temperature": 0.72, "top_p": 1.0, "repetition_penalty": 1.12, "max_new_tokens": 160},
        "adapters": {name: path for name, path in adapters},
        "candidates": {},
    }
    records = [{"index": i, "kind": item["kind"], "memory": item["lines"], "reference": item["row"]["messages"][-1]["content"]} for i, item in enumerate(items)]
    candidates = ([] if args.skip_base else ["base"]) + [name for name, _ in adapters]
    for name in candidates:
        print(f"== {name}", flush=True)
        if name == "base":
            manager = model.disable_adapter() if adapters else torch.no_grad()
        else:
            model.set_adapter(name)
            manager = torch.no_grad()
        counts = {"relevant": [0, 0], "distractor": [0, 0]}
        with manager:
            for index, (item, prompt) in enumerate(zip(items, prompts)):
                if prompt is None:
                    continue
                torch.manual_seed(args.seed + index)
                inputs = processor.apply_chat_template(
                    prompt, add_generation_prompt=True, enable_thinking=False, tokenize=True,
                    return_dict=True, return_tensors="pt",
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}
                text = generate_assistant_response(
                    model, processor, inputs, 160, do_sample=True, temperature=0.72, top_p=1.0, repetition_penalty=1.12
                )
                used = max(longest_shared_span(text, quote) for quote in item["quotes"]) >= args.min_span
                counts[item["kind"]][0] += used
                counts[item["kind"]][1] += 1
                records[index][name] = text
                records[index][f"{name}_used"] = used
        report["candidates"][name] = {
            "relevant_memory_used": f"{counts['relevant'][0]}/{counts['relevant'][1]}",
            "distractor_memory_used": f"{counts['distractor'][0]}/{counts['distractor'][1]}",
        }
        print(f"  {json.dumps(report['candidates'][name], ensure_ascii=False)}", flush=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with args.generations.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
