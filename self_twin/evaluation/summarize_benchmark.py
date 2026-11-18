#!/usr/bin/env python3
"""Compute auxiliary char-level similarity and memory retrieval checks."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

VARIANTS = ("base", "lora", "lora_rag")


def compact(text: str) -> str:
    return "".join(text.split())


def has_repeated_span(text: str) -> bool:
    """Catch a repeated contiguous answer phrase while ignoring whitespace."""
    text = compact(text)
    if len(text) < 12:
        return False
    for width in range(min(64, len(text) // 2), 1, -1):
        for start in range(len(text) - 2 * width + 1):
            phrase = text[start : start + width]
            if not text.startswith(phrase * 2, start):
                continue
            count = 2
            cursor = start + 2 * width
            while text.startswith(phrase, cursor):
                count += 1
                cursor += width
            if count >= 2 and width * count >= 12:
                return True
    return False


def lcs_length(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for left_char in left:
        current = [0]
        for index, right_char in enumerate(right, 1):
            if left_char == right_char:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(prediction: str, reference: str) -> float | None:
    prediction, reference = compact(prediction), compact(reference)
    if not prediction or not reference:
        return None
    common = lcs_length(prediction, reference)
    precision, recall = common / len(prediction), common / len(reference)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    category_scores = defaultdict(lambda: defaultdict(list))
    reference_scores = defaultdict(list)
    scoped_reference_scores = defaultdict(lambda: defaultdict(list))
    response_shapes = defaultdict(list)
    repeated_span_cases = defaultdict(list)
    memory = defaultdict(
        lambda: {
            "correct_fact": 0,
            "fact_total": 0,
            "fact_source_retrievals": 0,
            "fact_source_retrievals_missing": 0,
            "unexpected_retrievals": 0,
            "unknown_total": 0,
        }
    )

    for row in rows:
        reference = row.get("reference_response")
        if reference:
            scope = "known_memory_facts" if row.get("reference_kind") == "memory_fact" else "conversation_replies"
            for variant in VARIANTS:
                score = rouge_l_f1(row.get(variant, ""), reference)
                if score is not None:
                    category_scores[row["category"]][variant].append(score)
                    reference_scores[variant].append(score)
                    scoped_reference_scores["all_reference_responses"][variant].append(score)
                    scoped_reference_scores[scope][variant].append(score)

        for variant in VARIANTS:
            response = compact(row.get(variant, ""))
            if response:
                response_shapes[variant].append(
                    {
                        "characters": len(response),
                        "distinct_characters": len(set(response)),
                    }
                )
            if has_repeated_span(row.get(variant, "")):
                repeated_span_cases[variant].append(row.get("index", -1) + 1)

        if row.get("reference_kind") == "memory_fact":
            expected = row.get("expected_fact_terms", [])
            expected_ids = set(row.get("target_message_ids", []))
            for variant in VARIANTS:
                response = row.get(variant, "").casefold()
                memory[variant]["fact_total"] += 1
                if any(term.casefold() in response for term in expected):
                    memory[variant]["correct_fact"] += 1
            if expected_ids.intersection(row.get("rag_source_message_ids", [])):
                memory["lora_rag"]["fact_source_retrievals"] += 1
            else:
                memory["lora_rag"]["fact_source_retrievals_missing"] += 1
        elif row.get("reference_kind") == "unverified_memory":
            memory["lora_rag"]["unknown_total"] += 1
            memory["lora_rag"]["unexpected_retrievals"] += bool(row.get("rag_memory_ids"))

    report = {
        "status": "results_ready_for_subjective_review",
        "adapter_path": str(args.adapter.resolve()),
        "generation_file": str(args.input.resolve()),
        "prompt_count": len(rows),
        "generation_settings": rows[0].get("generation_settings") if rows else None,
        "auxiliary_character_rouge_l_f1": {
            variant: sum(values) / len(values) if values else None
            for variant, values in reference_scores.items()
        },
        "reference_counts_by_scope": {
            scope: len(next(iter(variants.values()))) if variants else 0
            for scope, variants in scoped_reference_scores.items()
        },
        "auxiliary_character_rouge_l_f1_by_scope": {
            scope: {
                variant: sum(values) / len(values) if values else None
                for variant, values in variants.items()
            }
            for scope, variants in scoped_reference_scores.items()
        },
        "character_rouge_l_f1_by_category": {
            category: {
                variant: sum(values) / len(values) if values else None
                for variant, values in variants.items()
            }
            for category, variants in category_scores.items()
        },
        "response_shape": {
            variant: {
                "median_characters": sorted(item["characters"] for item in values)[len(values) // 2]
                if values else None,
                "at_most_5_characters": sum(item["characters"] <= 5 for item in values),
                "over_100_characters": sum(item["characters"] > 100 for item in values),
                "long_low_diversity": sum(
                    item["characters"] >= 32 and item["distinct_characters"] <= 4 for item in values
                ),
            }
            for variant, values in response_shapes.items()
        },
        "repetition_check": {
            "method": "contiguous repeated spans after whitespace removal; minimum 12 characters across at least two repeats; diagnostic, not a style score",
            "cases_by_variant": {variant: cases for variant, cases in repeated_span_cases.items()},
        },
        "memory_checks": {
            variant: counters
            for variant, counters in memory.items()
        },
        "metrics_are_not_the_merge_gate": True,
        "subjective_scores": {"voice": None, "reaction_fit": None, "naturalness": None, "consistency": None},
        "merge_decision": "pending qualitative review of saved outputs",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k in {"status", "prompt_count", "auxiliary_character_rouge_l_f1", "memory_checks", "merge_decision"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
