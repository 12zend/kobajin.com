#!/usr/bin/env python3
"""Feed-post held-out evaluation: 20 fixed Wikipedia articles, same pipeline as the site.

For the pinned base model and each named adapter, every article goes through the
browser pipeline in src/agents.js: seed role → tweet role → form / fact-rule /
identity / novelty gates → fact_check role, with up to two regenerations.
Prompts and sampling come from src/prompts (role_prompts.py); gate, token-budget,
and public-memory logic run through gate_bridge.mjs, i.e. the site's own JS.

  python self_twin/evaluation/run_tweet_heldout.py \\
      --adapter hf=self_twin/adapter/experiment-r8-luna-synthetic-1000-v3/adapter \\
      --adapter candidate=path/to/candidate/adapter \\
      --output self_twin/evaluation/tweet_heldout-<name>.jsonl

Outputs: generations (.jsonl), a per-variant summary (.summary.json), and a
blinded review sheet (.review.md + .review-key.json) for scoring kobajin-likeness
by hand. Transformers bf16 differs from the browser's MLC q4f16_1 weights, so
pass rates here are comparable across variants, not identical to the site.
The site also constrains JSON roles with WebLLM response_format; Transformers
generation here is unconstrained, so invalid JSON is likelier and is counted
as a failed check, exactly as the site does when parsing fails.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoProcessor

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / "inference"))
from generation import generate_assistant_response  # noqa: E402
from role_prompts import compose_system_prompt, load_role_config, role_prompt_sha256, role_settings  # noqa: E402

MODEL_ID = "Qwen/Qwen3.5-4B"
DEFAULT_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
DEFAULT_CACHE = PROJECT / "hf_home" / "hub"
BRIDGE = HERE / "gate_bridge.mjs"
MAX_ATTEMPTS = 3  # first try + two regenerations, as in composeTweet()


class GateBridge:
    """Runs src/gates.js and src/memory.js through a persistent Node process."""

    def __init__(self) -> None:
        self.process = subprocess.Popen(
            ["node", str(BRIDGE)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )

    def __call__(self, op: str, **kwargs):
        assert self.process.stdin and self.process.stdout
        self.process.stdin.write(json.dumps({"op": op, **kwargs}, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        response = json.loads(self.process.stdout.readline())
        if not response["ok"]:
            raise RuntimeError(f"gate bridge {op}: {response['error']}")
        return response["result"]

    def close(self) -> None:
        if self.process.stdin:
            self.process.stdin.close()
        self.process.wait(timeout=10)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


class RoleRunner:
    def __init__(self, model, processor, device: str, bridge: GateBridge, seed: int) -> None:
        self.model = model
        self.processor = processor
        self.device = device
        self.bridge = bridge
        self.seed = seed
        config = load_role_config()
        self.context_window = config["context_window"]
        self.excerpt_budget = config["excerpt_token_budget"]
        self.margin = config["template_margin_tokens"]

    def _fit(self, role: str, excerpt: str, apply) -> tuple[str, str]:
        """Mirror runRole(): trim the excerpt first so the request fits the window."""
        settings = role_settings(role)
        empty_user, empty_context = apply("")
        fixed = self.bridge("estimate_tokens", text=compose_system_prompt(role, empty_context))
        fixed += self.bridge("estimate_tokens", text=empty_user)
        room = self.context_window - fixed - settings["max_tokens"] - self.margin
        budget = max(0, min(self.excerpt_budget, room))
        return apply(self.bridge("trim_to_budget", text=excerpt, max_tokens=budget))

    def run(self, role: str, *, user: str = "", context: str = "", fit=None, seed_offset: int = 0) -> dict:
        settings = role_settings(role)
        if fit is not None:
            user, context = self._fit(role, *fit)
        messages = [
            {"role": "system", "content": compose_system_prompt(role, context)},
            {"role": "user", "content": user},
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        do_sample = settings["temperature"] > 0
        torch.manual_seed(self.seed + seed_offset)
        raw = generate_assistant_response(
            self.model,
            self.processor,
            inputs,
            settings["max_tokens"],
            do_sample=do_sample,
            temperature=settings["temperature"],
            top_p=settings["top_p"],
            repetition_penalty=settings["repetition_penalty"] if do_sample else 1.0,
        )
        token_count = len(self.processor.tokenizer(raw, add_special_tokens=False)["input_ids"])
        text = self.bridge("clean", text=raw)
        result = {"text": text, "finish_reason": "length" if token_count >= settings["max_tokens"] else "stop"}
        if "json_schema" in settings:
            result["json"] = self.bridge("parse_json", text=text)
        return result


def valid_seed(value) -> dict | None:
    if not isinstance(value, dict) or not isinstance(value.get("angle"), str) or not value["angle"].strip():
        return None
    if not isinstance(value.get("facts"), list):
        return None
    facts = [fact.strip() for fact in value["facts"] if isinstance(fact, str) and fact.strip()]
    return {"angle": value["angle"].strip()[:40], "facts": facts[:2]} if facts else None


def compose_tweet(runner: RoleRunner, bridge: GateBridge, article: dict, recent: list[str], index: int) -> dict:
    """Python mirror of composeTweet() in src/agents.js."""
    log = {"seed": None, "verdicts": [], "retries": 0}
    seed = None
    for attempt in range(MAX_ATTEMPTS):
        log["retries"] = attempt
        verdict: dict = {"attempt": attempt}
        log["verdicts"].append(verdict)
        offset = index * 100 + attempt * 10
        if seed is None:
            result = runner.run(
                "seed",
                fit=(article["excerpt"], lambda excerpt: (f"記事: {article['title']}\n抜粋:\n{excerpt}", "")),
                seed_offset=offset,
            )
            parsed = valid_seed(result.get("json"))
            facts = bridge("filter_facts", facts=parsed["facts"], excerpt=article["excerpt"], title=article["title"]) if parsed else []
            verdict["seed"] = ("ok" if facts else "facts_unsupported") if parsed else "invalid_json"
            if not parsed or not facts:
                continue
            seed = {"angle": parsed["angle"], "facts": facts}
            log["seed"] = seed

        memory_context = bridge("memory_context", query=f"{article['title']} {' '.join(seed['facts'])}")
        user = "\n".join([f"題材: {article['title']}", f"angle: {seed['angle']}", "facts:", *[f"- {fact}" for fact in seed["facts"]]])
        tweet = runner.run("tweet", user=user, context=memory_context, seed_offset=offset + 1)
        text = tweet["text"]
        verdict["draft"] = text

        form = bridge("form", text=text, finish_reason=tweet["finish_reason"])
        verdict["form"] = "pass" if form["pass"] else form["reason"]
        if not form["pass"]:
            continue
        fact_rule = bridge("fact_rule", text=text, excerpt=article["excerpt"], title=article["title"])
        verdict["fact_rule"] = "pass" if fact_rule["pass"] else {"missing": fact_rule["missing"]}
        if not fact_rule["pass"]:
            continue
        identity = bridge("identity", text=text)
        verdict["identity"] = ("backed_by_memory" if identity["backed"] else "pass") if identity["pass"] else {"hits": identity["hits"]}
        if not identity["pass"]:
            continue
        novelty = bridge("novelty", text=text, recent=recent, excerpt=article["excerpt"])
        verdict["novelty"] = "pass" if novelty["pass"] else {"max_jaccard": novelty["maxJaccard"], "copy_chars": novelty["copyLength"]}
        if not novelty["pass"]:
            continue
        check = runner.run(
            "fact_check",
            fit=(article["excerpt"], lambda excerpt: (f"抜粋:\n{excerpt}\n\n投稿:\n{text}", "")),
            seed_offset=offset + 2,
        )
        verdict_value = check.get("json", {}).get("verdict") if isinstance(check.get("json"), dict) else None
        verdict["fact_model"] = verdict_value if verdict_value in {"supported", "unsupported", "unknown"} else "invalid_json"
        if verdict["fact_model"] != "supported":
            continue
        return {"accepted": True, "text": text, **log}
    return {"accepted": False, "text": None, **log}


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for variant in sorted({row["variant"] for row in rows}):
        items = [row for row in rows if row["variant"] == variant]
        drafts = [verdict for row in items for verdict in row["verdicts"] if "draft" in verdict]
        attempts = [verdict for row in items for verdict in row["verdicts"]]

        def share(predicate, population) -> float:
            return round(sum(1 for item in population if predicate(item)) / len(population), 3) if population else 0.0

        summary[variant] = {
            "articles": len(items),
            "accepted": sum(1 for row in items if row["accepted"]),
            "gate_pass_rate": share(lambda row: row["accepted"], items),
            "first_try_pass_rate": share(lambda row: row["accepted"] and row["retries"] == 0, items),
            "attempts": len(attempts),
            "drafts": len(drafts),
            "seed_invalid_rate": share(lambda v: v.get("seed") in {"invalid_json", "facts_unsupported"}, attempts),
            "form_fail_rate": share(lambda v: v.get("form") not in {None, "pass"}, drafts),
            "fact_rule_fail_rate": share(lambda v: isinstance(v.get("fact_rule"), dict), drafts),
            # 捏造率: drafts that presented Wikipedia content as kobajin's own experience.
            "identity_violation_rate": share(lambda v: isinstance(v.get("identity"), dict), drafts),
            # 重複率: drafts too close to earlier posts or copied from the excerpt.
            "novelty_fail_rate": share(lambda v: isinstance(v.get("novelty"), dict), drafts),
            "fact_model_rejection_rate": share(lambda v: v.get("fact_model") in {"unsupported", "unknown", "invalid_json"}, drafts),
            "fact_model_invalid_json_rate": share(lambda v: v.get("fact_model") == "invalid_json", drafts),
        }
    return summary


def write_review(rows: list[dict], articles: list[dict], path: Path, key_path: Path, seed: int) -> None:
    """Blinded sheet: per article, accepted posts from every variant in random order."""
    rng = random.Random(seed)
    lines = [
        "# Feed post review (blinded)",
        "",
        "Score each post for kobajin-likeness (1–5) and note invented personal facts or factual errors.",
        "Wikipedia excerpts are CC BY-SA 4.0; follow each source link.",
        "",
    ]
    key = {}
    for article in articles:
        options = [row for row in rows if row["index"] == article["index"]]
        rng.shuffle(options)
        lines += [f"## {article['index'] + 1}. {article['title']}", "", f"Source: {article['url']}", ""]
        for letter, row in zip("ABCDEFGH", options):
            key[f"{article['index']}{letter}"] = row["variant"]
            text = row["text"] if row["accepted"] else "(dropped: no draft passed the gates)"
            lines += [f"- **{letter}** {text}", "  - score: ", "  - notes: "]
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    key_path.write_text(json.dumps(key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_adapter(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or label == "base":
        raise argparse.ArgumentTypeError("use LABEL=PATH with a label other than 'base'")
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--articles", type=Path, default=HERE / "tweet_heldout_articles.jsonl")
    parser.add_argument("--adapter", type=parse_adapter, action="append", default=[], help="LABEL=PATH, repeatable")
    parser.add_argument("--output", type=Path, default=HERE / "tweet_heldout-generations.jsonl")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, help="evaluate only the first N articles (smoke runs)")
    args = parser.parse_args()

    articles = load_jsonl(args.articles)[: args.limit]
    if not articles:
        raise SystemExit(f"No articles in {args.articles}; run build_tweet_heldout.mjs first")
    for label, path in args.adapter:
        if not path.is_dir():
            raise SystemExit(f"Adapter directory for {label} does not exist: {path}")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("No CUDA or MPS device is available; refusing 4B evaluation on CPU")
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=args.revision, cache_dir=DEFAULT_CACHE, local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID,
        revision=args.revision,
        cache_dir=DEFAULT_CACHE,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device)
    model.eval()

    settings = {
        "model_id": MODEL_ID,
        "revision": args.revision,
        "seed": args.seed,
        "enable_thinking": False,
        "roles": {role: {**role_settings(role), "prompt_sha256": role_prompt_sha256(role)} for role in ("seed", "tweet", "fact_check")},
        "articles": str(args.articles),
    }
    bridge = GateBridge()
    rows: list[dict] = []
    variants = [("base", None), *args.adapter]
    try:
        for variant_index, (label, adapter_path) in enumerate(variants):
            if adapter_path is not None:
                if isinstance(model, PeftModel):
                    model.load_adapter(adapter_path, adapter_name=label, is_trainable=False)
                else:
                    model = PeftModel.from_pretrained(model, adapter_path, adapter_name=label, is_trainable=False)
                model.set_adapter(label)
                model.eval()
            runner = RoleRunner(model, processor, device, bridge, args.seed)
            recent: list[str] = []
            for article in articles:
                result = compose_tweet(runner, bridge, article, recent, article["index"])
                if result["accepted"]:
                    recent.insert(0, result["text"])
                rows.append({
                    "variant": label,
                    "adapter": str(adapter_path) if adapter_path else None,
                    "index": article["index"],
                    "title": article["title"],
                    "url": article["url"],
                    "revid": article.get("revid"),
                    **result,
                    "generation_settings": settings,
                })
                write_jsonl(args.output, rows)
                print(f"{label} {article['index'] + 1}/{len(articles)} accepted={result['accepted']}", flush=True)
    finally:
        bridge.close()

    summary = summarize(rows)
    args.output.with_suffix(".summary.json").write_text(
        json.dumps({"settings": settings, "variants": summary}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_review(rows, articles, args.output.with_suffix(".review.md"), args.output.with_suffix(".review-key.json"), args.seed)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
