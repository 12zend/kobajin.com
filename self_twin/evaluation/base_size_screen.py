#!/usr/bin/env python3
"""Screen small base models for a low-memory kobajinAI before training one.

For each unadapted base model, on the same held-out real replies (chronological
test split, the 240 rows kobajin_likeness.py uses, production system prompt):

- NLL per character of kobajin's actual reply given the conversation. Tokenizers
  differ between model families, so per-token NLL is not comparable; per
  character is. Only the reply text is scored (not the end-of-turn token).
- Tokens per character of the replies and of the whole prompt. Fewer tokens for
  the same Japanese text means less prompt processing and fewer decoding steps.
- A few sampled replies with the site's decoding, to read for language and sense.

A base model's fit to kobajin's real Japanese chat is a proxy for how much a
LoRA on the same data can reach; it is not a substitute for training and
evaluating the adapted model.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoModelForMultimodalLM, AutoProcessor, AutoTokenizer

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT / "training"))
from kobajin_likeness import CACHE, MODEL_ID, PROD_PROMPT, REVISION, clean_target, fitted_context, load_rows  # noqa: E402

CANDIDATES = {
    "qwen35-0.8b": ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17", "multimodal"),
    "qwen35-2b": ("Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc", "multimodal"),
    "qwen35-4b": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", "multimodal"),
    "minicpm5-1b": ("openbmb/MiniCPM5-1B", "ac83f813ba7da263c51a9f3ec37c2072597e54af", "causal"),
    "minicpm5-2b": ("openbmb/MiniCPM5-2B", "42433e90dad95123ab87729fd7011e7110629556", "causal"),
    "lfm25-1.2b-jp": ("LiquidAI/LFM2.5-1.2B-JP-202606", "52b8b4475311a63bf839c6494f78f8ad59d13515", "causal"),
}


def template_ids(tokenizer, messages, add_generation_prompt):
    kwargs = dict(tokenize=True, add_generation_prompt=add_generation_prompt, return_dict=True, return_tensors=None)
    try:
        ids = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)["input_ids"]
    except TypeError:
        ids = tokenizer.apply_chat_template(messages, **kwargs)["input_ids"]
    return ids[0] if ids and isinstance(ids[0], list) else ids


def prompt_ids(tokenizer, context):
    return template_ids(
        tokenizer, [{"role": "system", "content": PROD_PROMPT}, {"role": "user", "content": context}], True
    )


@torch.inference_mode()
def reply_nll(model, tokenizer, context, target, device):
    prefix = prompt_ids(tokenizer, context)
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    input_ids = torch.tensor([prefix + target_ids], device=device)
    logits = model(input_ids=input_ids).logits[0, len(prefix) - 1 : -1].float()
    loss = torch.nn.functional.cross_entropy(logits, input_ids[0, len(prefix) :], reduction="sum")
    return float(loss.item()), len(target_ids), len(prefix)


@torch.inference_mode()
def sample_reply(model, tokenizer, context, device, seed):
    prefix = prompt_ids(tokenizer, context)
    torch.manual_seed(seed)
    output = model.generate(
        input_ids=torch.tensor([prefix], device=device),
        do_sample=True,
        temperature=0.72,
        top_p=1.0,
        top_k=0,
        repetition_penalty=1.12,
        max_new_tokens=80,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    )
    return tokenizer.decode(output[0, len(prefix) :], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(CANDIDATES))
    parser.add_argument("--adapter", action="append", default=[], help="label=model_key=adapter_dir; scored like a model")
    parser.add_argument("--test", type=Path, default=PROJECT / "dataset" / "test.jsonl")
    parser.add_argument("--nll-samples", type=int, default=240)
    parser.add_argument("--gen-samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--output", type=Path, required=True, help="Aggregate numbers only (no conversation text)")
    parser.add_argument("--generations", type=Path, required=True, help="Contains private conversation text; keep local")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    # Contexts are fitted once with the 4B processor so every model sees identical text.
    reference = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION, cache_dir=CACHE, local_files_only=True)
    rows = load_rows(args.test, args.nll_samples, args.seed)
    contexts = [fitted_context(reference, row, PROD_PROMPT, 768) for row in rows]
    pairs = [(c, clean_target(r["messages"][-1]["content"])) for c, r in zip(contexts, rows) if c is not None]
    target_chars = sum(len(t) for _, t in pairs)

    report = json.loads(args.output.read_text(encoding="utf-8")) if args.output.exists() else {}
    report.update({"rows_scored": len(pairs), "target_chars": target_chars, "seed": args.seed, "system_prompt": "prod"})
    report.setdefault("models", {})
    generations = {}
    if args.generations.exists():
        generations = json.loads(args.generations.read_text(encoding="utf-8"))

    jobs = [(m, m, None) for m in args.models.split(",") if m]
    for item in args.adapter:
        label, key, path = item.split("=", 2)
        jobs.append((label, key, Path(path)))
    for name, key, adapter in jobs:
        repo, revision, kind = CANDIDATES[key]
        print(f"== {name} {repo}@{revision[:7]}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision, cache_dir=CACHE, local_files_only=True)
        loader = AutoModelForMultimodalLM if kind == "multimodal" else AutoModelForCausalLM
        model = loader.from_pretrained(
            repo, revision=revision, cache_dir=CACHE, dtype=torch.bfloat16, low_cpu_mem_usage=True, local_files_only=True
        ).to(device)
        if adapter is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
        model.eval()
        total_loss = total_tokens = total_prompt = 0
        per_row = []
        for index, (context, target) in enumerate(pairs):
            loss, tokens, prompt_tokens = reply_nll(model, tokenizer, context, target, device)
            total_loss += loss
            total_tokens += tokens
            total_prompt += prompt_tokens
            per_row.append(loss / max(1, len(target)))
            if (index + 1) % 60 == 0:
                print(f"  {index + 1}/{len(pairs)} nll/char {total_loss / sum(len(t) for _, t in pairs[: index + 1]):.4f}", flush=True)
        prompt_chars = sum(len(PROD_PROMPT) + len(c) for c, _ in pairs)
        samples = [
            {"context_tail": context[-200:], "real": target, "sampled": sample_reply(model, tokenizer, context, device, 42 + i)}
            for i, (context, target) in enumerate(pairs[: args.gen_samples])
        ]
        report["models"][name] = {
            "repo": repo,
            "revision": revision,
            "adapter": str(adapter) if adapter else None,
            "parameters": sum(p.numel() for p in model.parameters()),
            "nll_per_char": round(total_loss / target_chars, 4),
            "bits_per_char": round(total_loss / target_chars / math.log(2), 4),
            "median_row_nll_per_char": round(statistics.median(per_row), 4),
            "nll_per_token": round(total_loss / total_tokens, 4),
            "reply_tokens_per_char": round(total_tokens / target_chars, 4),
            "prompt_tokens_mean": round(total_prompt / len(pairs), 1),
            "prompt_tokens_per_char": round(total_prompt / prompt_chars, 4),
        }
        generations[name] = samples
        print(json.dumps(report["models"][name], ensure_ascii=False), flush=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        args.generations.write_text(json.dumps(generations, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        del model
        if device == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
