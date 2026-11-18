#!/usr/bin/env python3
"""Build a blinded, local-only review sheet for one benchmark run."""

from __future__ import annotations

import argparse
import html
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def block(label: str, text: str | None) -> str:
    safe = html.escape(text or "（応答なし）")
    return f"### {label}\n\n<pre>{safe}</pre>\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, default=HERE / "evaluation_prompts.jsonl")
    parser.add_argument("--output", type=Path, default=HERE / "subjective_review.md")
    parser.add_argument("--key", type=Path, default=HERE / "subjective_review_key.json")
    parser.add_argument("--seed", type=int, default=20261118)
    args = parser.parse_args()

    prompts = load_jsonl(args.prompts)
    generations = load_jsonl(args.generations)
    if len(prompts) != len(generations):
        raise SystemExit("Prompt and generation counts differ")

    sections = [
        "# 盲検モデル評価シート",
        "",
        "各ケースでA/B/Cを読んでから、本人らしさ・会話への反応・自然さ・一貫性を1〜5で採点してください。",
        "記憶の質問では、根拠のない事実を作らないかも採点してください。実際の返答は採点後に開いてください。",
        "A/B/Cのモデル対応は別ファイルのanswer keyにあります。採点前に開くと盲検になりません。",
        "",
    ]
    answer_key: dict[str, dict[str, str]] = {}

    for index, (prompt, result) in enumerate(zip(prompts, generations)):
        if result.get("index") != index or result.get("category") != prompt.get("category"):
            raise SystemExit(f"Prompt/generation mismatch at row {index}")
        if result.get("target_message_ids", []) != prompt.get("target_message_ids", []):
            raise SystemExit(f"Target message IDs differ at row {index}")

        options = [("base", result.get("base")), ("lora", result.get("lora")), ("lora_rag", result.get("lora_rag"))]
        rng = random.Random(args.seed + index)
        rng.shuffle(options)
        labels = "ABC"
        answer_key[str(index + 1)] = {label: model for label, (model, _) in zip(labels, options)}

        sections.extend([f"## ケース {index + 1}: {prompt['category']}", ""])
        for message in prompt.get("prompt_messages", []):
            role = message.get("role")
            if role not in {"user", "assistant"}:
                continue
            speaker = "会話相手" if role == "user" else "本人の過去発言"
            sections.extend([f"### {speaker}", "", f"<pre>{html.escape(str(message.get('content', '')))}</pre>", ""])
        for label, (_, response) in zip(labels, options):
            sections.append(block(f"候補 {label}", response))
        sections.extend([
            "### 採点",
            "",
            "- 本人らしさ (1–5):",
            "- 会話への反応 (1–5):",
            "- 自然さ (1–5):",
            "- 一貫性 (1–5):",
            "- 記憶の正確さ・不明時の慎重さ (1–5 / 該当時):",
            "- コメント:",
            "",
            "<details><summary>採点後に実際の返答を確認</summary>",
            "",
            f"<pre>{html.escape(str(prompt.get('reference_response') or '記録上の正解なし'))}</pre>",
            "",
            "</details>",
            "",
        ])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.key.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(sections), encoding="utf-8")
    args.key.write_text(json.dumps(answer_key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote blinded review for {len(prompts)} cases to {args.output}")


if __name__ == "__main__":
    main()
