#!/usr/bin/env python3
"""Select a small, deterministic, human-review set from held-out test sessions."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=Path, default=Path(__file__).resolve().parents[1] / "dataset" / "test.jsonl")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "evaluation_prompts.jsonl")
    parser.add_argument("--per-category", type=int, default=5)
    args = parser.parse_args()

    by_category: dict[str, list[dict]] = {}
    with args.test.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            by_category.setdefault(row["metadata"]["category"], []).append(row)

    selected = []
    selected_ids: set[str] = set()
    for category, rows in sorted(by_category.items()):
        # Evenly sample across the chronological category slice.
        count = min(args.per_category, len(rows))
        positions = sorted({round(i * (len(rows) - 1) / max(1, count - 1)) for i in range(count)})
        for position in positions:
            row = rows[position]
            selected.append(
                {
                    "category": category,
                    "prompt_messages": row["messages"][:-1],
                    "reference_response": row["messages"][-1]["content"],
                    "target_message_ids": row["metadata"]["message_ids"],
                    "context_message_ids": row["metadata"]["context_message_ids"],
                    "human_scores": {
                        "voice": None,
                        "reaction_fit": None,
                        "conversation_fit": None,
                        "naturalness": None,
                    },
                }
            )
            selected_ids.update(row["metadata"]["message_ids"])

    # Reserve one real held-out example for low-stakes correction/disagreement.
    disagreement_id = "1566319870352564409"
    for row in (row for rows in by_category.values() for row in rows):
        if disagreement_id in row["metadata"]["message_ids"] and not selected_ids.intersection(row["metadata"]["message_ids"]):
            selected.append(
                {
                    "category": "disagreement_or_correction",
                    "prompt_messages": row["messages"][:-1],
                    "reference_response": row["messages"][-1]["content"],
                    "reference_kind": "held_out_reply",
                    "target_message_ids": row["metadata"]["message_ids"],
                    "context_message_ids": row["metadata"]["context_message_ids"],
                    "human_scores": {"voice": None, "reaction_fit": None, "conversation_fit": None, "naturalness": None},
                }
            )
            selected_ids.update(row["metadata"]["message_ids"])
            break

    memory_system = (
        "自然な日本語で返事してください。過去の事実について、記録にない場合は推測で埋めないでください。"
    )
    memory_cases = [
        {
            "query": "前に歌の話で、昔習ってたって言ってたの何だっけ？",
            "reference": "昔、民謡を習っていたと発言している。",
            "source_id": "1310535154892279848",
            "expected_terms": ["民謡"],
        },
        {
            "query": "TextBlurって前に自分で作ったって言ってたやつだよね？",
            "reference": "TextBlurを作ったと発言している。今も保守・利用しているかは不明。",
            "source_id": "1235828364368154646",
            "expected_terms": ["TextBlur"],
        },
        {
            "query": "前に好きな食べ物何て言ってたっけ？",
            "reference": "2024年時点でナナチキが大好きだと発言している。",
            "source_id": "1278367462609653761",
            "expected_terms": ["ナナチキ"],
        },
        {
            "query": "前に好きなスポーツ何て言ってたっけ？",
            "reference": None,
            "source_id": None,
            "expected_terms": [],
        },
    ]
    for case in memory_cases:
        selected.append(
            {
                "category": "memory_recall" if case["source_id"] else "memory_unknown",
                "prompt_messages": [
                    {"role": "system", "content": memory_system},
                    {"role": "user", "content": case["query"]},
                ],
                "reference_response": case["reference"],
                "reference_kind": "memory_fact" if case["source_id"] else "unverified_memory",
                "expected_fact_terms": case["expected_terms"],
                "target_message_ids": [case["source_id"]] if case["source_id"] else [],
                "context_message_ids": [],
                "human_scores": {"voice": None, "reaction_fit": None, "conversation_fit": None, "naturalness": None},
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(selected)} prompts: held-out replies plus memory recall/abstention cases")


if __name__ == "__main__":
    main()
