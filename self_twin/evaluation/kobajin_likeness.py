#!/usr/bin/env python3
"""Measure how kobajin-like base and LoRA candidates are on held-out real chat.

Two complementary checks, both on the chronological test split (never trained on):

1. Held-out NLL: the per-token negative log-likelihood of kobajin's actual
   replies given the same conversation context. Lower means the model puts more
   probability on what kobajin really wrote. It is computed under the production
   system prompt (what the site sends) and optionally under the short prompt the
   older runs trained with, so a train/inference prompt mismatch is visible.
2. Generation style profile: sampled replies with the site's decoding settings,
   compared to the real reply distribution (length, register, assistant-isms,
   synthetic template phrases, placeholders, repetition, char-bigram distance).

Neither number is a substitute for reading the outputs; the generations file
keeps every prompt/reply pair for review.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import re
import statistics
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
from generation import generate_assistant_response  # noqa: E402
from summarize_benchmark import has_repeated_span  # noqa: E402
from train import fit_context_to_length  # noqa: E402

MODEL_ID = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
CACHE = PROJECT / "hf_home" / "hub"
PROD_PROMPT = (PROJECT / "inference" / "system_prompt.txt").read_text(encoding="utf-8").strip()
SHORT_PROMPT = "次のグループ会話の流れに合わせて、自然に返事してください。"
PROMPTS = {"prod": PROD_PROMPT, "short": SHORT_PROMPT}

KEIGO = re.compile(r"(です|ます|ました|ません|ましょう|でしょう|ください)")
ASSISTANTISM = re.compile(r"(お手伝い|いかがでしょう|ございます|でしょうか|教えてください|詳しく|具体的に|ですね)")
SYNTHETIC_TEMPLATE = re.compile(r"(小さく試|全部を変えずに|差を見よ|一回だけ|一回やる|試せそう|手間をかけすぎ)")
PLACEHOLDER_ONLY = re.compile(r"(\s*【[^】]+】\s*)+")


def clean_target(text: str) -> str:
    return re.sub(r"\s*\(edited\)", "", text).strip()


def load_rows(path: Path, count: int, seed: int) -> list[dict]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    rows = [row for row in rows if clean_target(row["messages"][-1]["content"])]
    random.Random(seed).shuffle(rows)
    return rows[:count]


def prompt_ids(processor, system: str, context: str) -> list[int]:
    ids = processor.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": context}],
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_dict=True,
        return_tensors=None,
    )["input_ids"]
    return ids[0] if ids and isinstance(ids[0], list) else ids


def fitted_context(processor, row: dict, system: str, max_length: int) -> str | None:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": row["messages"][1]["content"]},
        {"role": "assistant", "content": clean_target(row["messages"][-1]["content"])},
    ]
    fitted, _ = fit_context_to_length(processor, messages, max_length)
    return None if fitted is None else fitted[1]["content"]


@torch.inference_mode()
def target_nll(model, processor, system: str, context: str, target: str, device: str) -> tuple[float, int]:
    prefix = prompt_ids(processor, system, context)
    target_ids = processor.tokenizer.encode(target + "<|im_end|>", add_special_tokens=False)
    input_ids = torch.tensor([prefix + target_ids], device=device)
    logits = model(input_ids=input_ids).logits[0, len(prefix) - 1 : -1].float()
    labels = input_ids[0, len(prefix) :]
    loss = torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
    return float(loss.item()), len(target_ids)


def bigram_counts(texts: list[str]) -> collections.Counter:
    counts: collections.Counter = collections.Counter()
    for text in texts:
        compact = "".join(text.split())
        counts.update(compact[i : i + 2] for i in range(len(compact) - 1))
    return counts


def jensen_shannon(left: collections.Counter, right: collections.Counter) -> float:
    left_total, right_total = sum(left.values()) or 1, sum(right.values()) or 1
    divergence = 0.0
    for key in set(left) | set(right):
        p, q = left[key] / left_total, right[key] / right_total
        m = (p + q) / 2
        if p:
            divergence += 0.5 * p * math.log2(p / m)
        if q:
            divergence += 0.5 * q * math.log2(q / m)
    return divergence


def style_profile(texts: list[str], reference_bigrams: collections.Counter) -> dict:
    lengths = [len("".join(text.split())) for text in texts]
    count = len(texts) or 1
    return {
        "n": len(texts),
        "median_chars": statistics.median(lengths) if lengths else 0,
        "share_le5_chars": round(sum(length <= 5 for length in lengths) / count, 3),
        "share_gt100_chars": round(sum(length > 100 for length in lengths) / count, 3),
        "mean_lines": round(statistics.mean(text.count("\n") + 1 for text in texts), 2) if texts else 0,
        "keigo_rate": round(sum(bool(KEIGO.search(text)) for text in texts) / count, 3),
        "assistantism_rate": round(sum(bool(ASSISTANTISM.search(text)) for text in texts) / count, 3),
        "synthetic_template_rate": round(sum(bool(SYNTHETIC_TEMPLATE.search(text)) for text in texts) / count, 3),
        "placeholder_only_rate": round(sum(bool(PLACEHOLDER_ONLY.fullmatch(text)) for text in texts) / count, 3),
        "repeated_span_rate": round(sum(has_repeated_span(text) for text in texts) / count, 3),
        "boku_rate": round(sum("僕" in text for text in texts) / count, 3),
        "laugh_w_rate": round(sum(bool(re.search(r"(w{1,}|ｗ{1,}|草)$", text.strip())) for text in texts) / count, 3),
        "char_bigram_jsd_vs_real": round(jensen_shannon(bigram_counts(texts), reference_bigrams), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", action="append", default=[], help="name=path; base is always included")
    parser.add_argument("--test", type=Path, default=PROJECT / "dataset" / "test.jsonl")
    parser.add_argument("--nll-samples", type=int, default=240)
    parser.add_argument("--gen-samples", type=int, default=60)
    parser.add_argument("--prompt-variants", default="prod,short")
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--temperature", type=float, default=0.72, help="site default 0.72")
    parser.add_argument("--repetition-penalty", type=float, default=1.12, help="site default 1.12")
    parser.add_argument("--skip-nll", action="store_true", help="Only sample generations (for decoding checks)")
    parser.add_argument("--skip-base", action="store_true", help="Reuse base numbers from an earlier run")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generations", type=Path, required=True)
    args = parser.parse_args()

    adapters = []
    for item in args.adapter:
        name, _, path = item.partition("=")
        if not path or not Path(path).is_dir():
            raise SystemExit(f"Adapter must be name=existing_dir: {item}")
        adapters.append((name, Path(path)))
    variants = [variant for variant in args.prompt_variants.split(",") if variant]
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("No GPU device")

    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION, cache_dir=CACHE, local_files_only=True)
    rows = load_rows(args.test, args.nll_samples, args.seed)
    all_test = [json.loads(line) for line in args.test.open(encoding="utf-8") if line.strip()]
    reference_bigrams = bigram_counts([clean_target(row["messages"][-1]["content"]) for row in all_test])
    contexts = {variant: [fitted_context(processor, row, PROMPTS[variant], args.max_length) for row in rows] for variant in variants}
    keep = [i for i in range(len(rows)) if all(contexts[v][i] is not None for v in variants)]
    rows = [rows[i] for i in keep]
    contexts = {v: [contexts[v][i] for i in keep] for v in variants}
    gen_rows = list(range(min(args.gen_samples, len(rows))))

    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, revision=REVISION, cache_dir=CACHE, dtype=torch.bfloat16, low_cpu_mem_usage=True, local_files_only=True
    ).to(device)
    model.eval()
    if adapters:
        model = PeftModel.from_pretrained(model, adapters[0][1], adapter_name=adapters[0][0], is_trainable=False)
        for name, path in adapters[1:]:
            model.load_adapter(path, adapter_name=name, is_trainable=False)
        model.eval()

    candidates = ([] if args.skip_base else ["base"]) + [name for name, _ in adapters]
    previous = json.loads(args.output.read_text(encoding="utf-8")) if args.output.exists() else {}
    report = {
        "seed": args.seed,
        "test_rows_scored": len(rows),
        "generation_rows": len(gen_rows),
        "max_length": args.max_length,
        "decoding": {"temperature": args.temperature, "top_p": 1.0, "repetition_penalty": args.repetition_penalty},
        "system_prompts": {v: PROMPTS[v][:40] + "…" for v in variants},
        "adapters": {name: str(path) for name, path in adapters},
        "real_reference_style": style_profile(
            [clean_target(rows[i]["messages"][-1]["content"]) for i in gen_rows], reference_bigrams
        ),
        "candidates": previous.get("candidates", {}),
    }
    generations = {}
    if args.generations.exists():
        for line in args.generations.open(encoding="utf-8"):
            if line.strip():
                record = json.loads(line)
                generations[record["index"]] = record

    for name in candidates:
        print(f"== {name}", flush=True)
        if name == "base":
            context_manager = model.disable_adapter() if adapters else torch.no_grad()
        else:
            model.set_adapter(name)
            context_manager = torch.no_grad()
        with context_manager:
            result: dict = {"nll": {}}
            for variant in ([] if args.skip_nll else variants):
                per_category: dict = collections.defaultdict(lambda: [0.0, 0])
                total_loss, total_tokens = 0.0, 0
                for index, row in enumerate(rows):
                    loss, tokens = target_nll(
                        model, processor, PROMPTS[variant], contexts[variant][index],
                        clean_target(row["messages"][-1]["content"]), device,
                    )
                    total_loss += loss
                    total_tokens += tokens
                    bucket = per_category[row["metadata"]["category"]]
                    bucket[0] += loss
                    bucket[1] += tokens
                result["nll"][variant] = {
                    "token_nll": round(total_loss / total_tokens, 4),
                    "perplexity": round(math.exp(total_loss / total_tokens), 2),
                    "by_category": {
                        category: round(loss / tokens, 4) for category, (loss, tokens) in sorted(per_category.items())
                    },
                }
                print(f"  nll[{variant}] {result['nll'][variant]['token_nll']}", flush=True)
            texts = []
            for index in gen_rows:
                torch.manual_seed(args.seed + index)
                inputs = processor.apply_chat_template(
                    [
                        {"role": "system", "content": PROD_PROMPT},
                        {"role": "user", "content": contexts["prod"][index] if "prod" in contexts else rows[index]["messages"][1]["content"]},
                    ],
                    add_generation_prompt=True, enable_thinking=False, tokenize=True,
                    return_dict=True, return_tensors="pt",
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}
                text = generate_assistant_response(
                    model, processor, inputs, 160, do_sample=True, temperature=args.temperature, top_p=1.0,
                    repetition_penalty=args.repetition_penalty,
                )
                texts.append(text)
                record = generations.setdefault(index, {
                    "index": index,
                    "category": rows[index]["metadata"]["category"],
                    "target_message_ids": rows[index]["metadata"]["message_ids"],
                    "context_tail": contexts["prod"][index][-400:] if "prod" in contexts else "",
                    "reference": clean_target(rows[index]["messages"][-1]["content"]),
                })
                record[name] = text
            result["style"] = style_profile(texts, reference_bigrams)
            print(f"  style {json.dumps(result['style'], ensure_ascii=False)}", flush=True)
        report["candidates"][name] = result
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with args.generations.open("w", encoding="utf-8") as stream:
            for index in sorted(generations):
                stream.write(json.dumps(generations[index], ensure_ascii=False) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
