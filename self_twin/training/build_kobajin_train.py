#!/usr/bin/env python3
"""Derive the real-chat-only training split used for the kobajin-likeness run.

The source splits in dataset/ stay unchanged. This script writes cleaned copies
that remove artifacts the website can never use and that pull the model toward
a few generic replies:

- Discord "(edited)" markers are stripped from targets and context.
- Targets that are only placeholders (for example 【URL】) are dropped.
- A target line that is only a Discord role mention such as @通話募集 is removed;
  a target left empty by that is dropped.
- Identical targets are capped (default 20 copies) so the most frequent
  reactions do not dominate the loss.
- Optionally, every row's system prompt is replaced with the production prompt
  the site sends, so training and inference see the same instructions.

No synthetic examples are mixed in: this run measures kobajin's own voice.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
EDITED = re.compile(r"\s*\(edited\)")
PLACEHOLDER_ONLY = re.compile(r"(\s*【[^】]+】\s*)+")
ROLE_MENTION_LINE = re.compile(r"^\s*@\S+\s*$")


def clean_target(text: str) -> str:
    lines = [EDITED.sub("", line).rstrip() for line in text.split("\n")]
    lines = [line for line in lines if line.strip() and not ROLE_MENTION_LINE.match(line)]
    return "\n".join(lines).strip()


def clean_rows(rows, system_prompt, cap, low_weight_rate, seed, stats):
    rng = random.Random(seed)
    seen: collections.Counter = collections.Counter()
    order = list(range(len(rows)))
    rng.shuffle(order)  # the cap keeps a random subset of each duplicate, not the oldest
    kept = []
    for index in order:
        row = json.loads(json.dumps(rows[index]))
        messages = row["messages"]
        target = clean_target(messages[-1]["content"])
        if not target:
            stats["dropped_empty_after_mention_strip"] += 1
            continue
        if PLACEHOLDER_ONLY.fullmatch(target):
            stats["dropped_placeholder_only"] += 1
            continue
        if cap and seen[target] >= cap:
            stats["dropped_duplicate_over_cap"] += 1
            continue
        if row.get("metadata", {}).get("low_weight_review") and rng.random() >= low_weight_rate:
            stats["dropped_low_weight_sampling"] += 1
            continue
        seen[target] += 1
        messages[-1]["content"] = target
        messages[1]["content"] = EDITED.sub("", messages[1]["content"])
        if system_prompt is not None:
            messages[0]["content"] = system_prompt
        row.setdefault("metadata", {})["kobajin_clean_v1"] = True
        kept.append((index, row))
    kept.sort(key=lambda item: item[0])  # restore chronological order
    stats["kept"] += len(kept)
    return [row for _, row in kept]


def write(path: Path, rows) -> str:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-prefix", default="kobajin_clean_v1")
    parser.add_argument("--duplicate-cap", type=int, default=20)
    parser.add_argument("--low-weight-rate", type=float, default=0.5)
    parser.add_argument("--system-prompt", choices=("source", "production"), default="source")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    dataset = PROJECT / "dataset"
    system_prompt = None
    if args.system_prompt == "production":
        system_prompt = (PROJECT / "inference" / "system_prompt.txt").read_text(encoding="utf-8").strip()
    manifest = {
        "source": "dataset/train.jsonl and dataset/validation.jsonl (real Discord chat only, no synthetic rows)",
        "duplicate_cap": args.duplicate_cap,
        "low_weight_rate": args.low_weight_rate,
        "system_prompt": args.system_prompt,
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest() if system_prompt else None,
        "seed": args.seed,
        "splits": {},
    }
    for split, rate in (("train", args.low_weight_rate), ("validation", 1.0)):
        rows = [json.loads(line) for line in (dataset / f"{split}.jsonl").open(encoding="utf-8") if line.strip()]
        stats: collections.Counter = collections.Counter(source=len(rows))
        cap = args.duplicate_cap if split == "train" else 0
        cleaned = clean_rows(rows, system_prompt, cap, rate, args.seed, stats)
        path = dataset / f"{args.output_prefix}_{split}.jsonl"
        digest = write(path, cleaned)
        categories = collections.Counter(row["metadata"]["category"] for row in cleaned)
        manifest["splits"][split] = {"file": path.name, "sha256": digest, **stats, "categories": dict(categories)}
    (dataset / f"{args.output_prefix}_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
