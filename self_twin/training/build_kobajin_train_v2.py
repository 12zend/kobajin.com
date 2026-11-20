#!/usr/bin/env python3
"""Derive the kobajin_clean_v2 split from kobajin_clean_v1.

v1 learned kobajin's voice, but its evaluation showed four problems. Each
change below targets one of them:

1. It ignored memories the site retrieves (0/3). So v2 adds rows that use the
   site's exact memory format: the production system prompt plus the
   getMemoryContext() block. A candidate memory quotes a real self-statement
   that kobajin made in the target reply, using the same wording as the
   reviewed records ("本人が「…」と発言"). The target stays his real reply.
   Distractor rows add memory blocks that have nothing to do with the
   conversation; there the real reply ignores them. Memories used by the
   evaluation (民謡, TextBlur, ナナチキ) and their source messages are never used.
2. It overused 「うんこ」 (14% of replies, against 0.78% of real replies). So
   targets containing it are downsampled.
3. It leaked anonymisation labels such as 【参加者】. So targets naming another
   participant are dropped, and 【URL】 tokens are removed from targets.
4. It only ever saw the short training prompt, never the production prompt
   alone. So a share of ordinary rows also gets the production prompt.

Every target is still a real Discord reply; no synthetic replies are added.
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
DATASET = PROJECT / "dataset"

SELF_STATEMENT = re.compile(
    r"(僕|ぼく|俺|おれ|私|わたし|自分)(は|も|が|の)?.{0,20}"
    r"(好き|嫌い|苦手|昔|習って|作った|作って|行った|住ん|食べ|飼って|持って|使って|やって|通って|生まれ|歳|学校|趣味)"
)
EVAL_MEMORY_TERMS = ("民謡", "textblur", "ナナチキ")
URL_TOKEN = re.compile(r"\s*【URL】\s*")
PARTICIPANT = "【参加者】"

MEMORY_HEADER = "今回の質問に関係する本人の確認済み記憶候補:"
MEMORY_FOOTER = "記憶候補と質問の話題が一致するときだけ使う。"
STALE_NOTE = "（当時の情報で、現在は変わっている可能性がある）"


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def eval_memory_source_ids() -> set[str]:
    path = PROJECT / "memory" / "personal_memory.jsonl"
    return {json.loads(line)["source_message_id"] for line in path.open(encoding="utf-8") if line.strip()}


def clean_target(text: str) -> str:
    lines = [URL_TOKEN.sub(" ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def memory_line(quote: str, row: dict, rng: random.Random) -> str:
    meta = row["metadata"]
    date = str(meta.get("last_target_time", ""))[:10]
    source = (meta.get("message_ids") or ["unknown"])[0]
    stale = STALE_NOTE if rng.random() < 0.5 else ""
    confidence = rng.choice(("0.9", "0.92", "0.94", "0.97"))
    return f"- {date}、本人が「{quote}」と発言。{stale} [source_message_id: {source}; confidence: {confidence}]"


def memory_block(lines: list[str]) -> str:
    return "\n".join([MEMORY_HEADER, *lines, MEMORY_FOOTER])


def with_system(row: dict, system: str, tag: str) -> dict:
    row = json.loads(json.dumps(row))
    row["messages"][0]["content"] = system
    row["metadata"]["kobajin_clean_v2"] = tag
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-prefix", default="kobajin_clean_v1")
    parser.add_argument("--output-prefix", default="kobajin_clean_v2")
    parser.add_argument("--unko-keep-rate", type=float, default=0.33)
    parser.add_argument("--memory-copies", type=int, default=2, help="memory rows per self-statement")
    parser.add_argument("--distractor-rows", type=int, default=400)
    parser.add_argument("--production-prompt-rate", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    production = (PROJECT / "inference" / "system_prompt.txt").read_text(encoding="utf-8").strip()
    excluded_sources = eval_memory_source_ids()
    manifest = {"source_prefix": args.source_prefix, "args": vars(args), "splits": {}}

    for split in ("train", "validation"):
        rows = load(DATASET / f"{args.source_prefix}_{split}.jsonl")
        stats = collections.Counter(source=len(rows))
        cleaned = []
        for row in rows:
            target = row["messages"][-1]["content"]
            if PARTICIPANT in target:
                stats["dropped_participant_label"] += 1
                continue
            target = clean_target(target)
            if not target:
                stats["dropped_empty_after_url_strip"] += 1
                continue
            if split == "train" and "うんこ" in target and rng.random() >= args.unko_keep_rate:
                stats["dropped_unko_downsample"] += 1
                continue
            row["messages"][-1]["content"] = target
            row["metadata"]["kobajin_clean_v2"] = "plain"
            cleaned.append(row)

        if split == "train":
            candidates = []
            for index, row in enumerate(cleaned):
                target = row["messages"][-1]["content"]
                ids = set(row["metadata"].get("message_ids") or [])
                if ids & excluded_sources or any(term in target.lower() for term in EVAL_MEMORY_TERMS):
                    continue
                if len(target) > 120 or "【" in target:
                    continue
                # A memory is a statement, not a question, and long enough to carry a fact.
                line = next(
                    (
                        line
                        for line in target.split("\n")
                        if SELF_STATEMENT.search(line) and len(line.strip()) >= 6 and not re.search(r"[？?]\s*$", line)
                    ),
                    None,
                )
                if line:
                    candidates.append((index, line.strip()))
            stats["memory_self_statements"] = len(candidates)
            candidate_indices = {index for index, _ in candidates}
            memory_rows = []
            for index, quote in candidates:
                row = cleaned[index]
                others = [item for item in candidates if item[0] != index]
                for _ in range(args.memory_copies):
                    lines = [memory_line(quote, row, rng)]
                    for other_index, other_quote in rng.sample(others, rng.randint(0, 2)):
                        lines.append(memory_line(other_quote, cleaned[other_index], rng))
                    rng.shuffle(lines)
                    memory_rows.append(with_system(row, f"{production}\n\n{memory_block(lines)}", "memory_relevant"))
            plain_indices = [i for i in range(len(cleaned)) if i not in candidate_indices]
            distractor_indices = set(rng.sample(plain_indices, min(args.distractor_rows, len(plain_indices))))
            for index in sorted(distractor_indices):
                lines = [memory_line(q, cleaned[i], rng) for i, q in rng.sample(candidates, rng.randint(1, 2))]
                cleaned[index] = with_system(cleaned[index], f"{production}\n\n{memory_block(lines)}", "memory_distractor")
            production_count = 0
            for index in plain_indices:
                if index not in distractor_indices and rng.random() < args.production_prompt_rate:
                    cleaned[index] = with_system(cleaned[index], production, "production_prompt")
                    production_count += 1
            stats["memory_relevant_rows"] = len(memory_rows)
            stats["memory_distractor_rows"] = len(distractor_indices)
            stats["production_prompt_rows"] = production_count
            # Keep chronological order for the real rows; the trainer shuffles anyway.
            cleaned.extend(memory_rows)

        stats["kept"] = len(cleaned)
        path = DATASET / f"{args.output_prefix}_{split}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for row in cleaned:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        tags = collections.Counter(row["metadata"].get("kobajin_clean_v2") for row in cleaned)
        manifest["splits"][split] = {
            "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            **stats,
            "row_types": dict(tags),
        }
    (DATASET / f"{args.output_prefix}_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
