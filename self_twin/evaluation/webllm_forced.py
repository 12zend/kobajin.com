#!/usr/bin/env python3
"""Teacher-forced WebLLM vs Transformers comparison at fixed token positions.

Greedy parity runs compare each engine on its own reply, so two bundles are
measured at different positions. This tool fixes the positions:

  reference  Transformers (BF16) samples replies to the parity prompts with the
             site's decoding. Every reply token whose prefix survives a
             decode/encode round trip becomes a position: the prompt plus the
             reply so far, as text, and Transformers' top-20 next-token log
             probabilities there.
  compare    Reads a WebLLM run made with the parity page in `mode=forced`
             (WebLLM's text completion scores each prefix and keeps its most
             likely next token) and reports, at the same positions, how often
             WebLLM's top token is Transformers' top token and the mean
             absolute log-probability difference for WebLLM's token.

Usage:
  webllm_forced.py reference --run evaluation/webllm_parity/v2-site-greedy.json --name v2-forced-ref
  (open /self_twin/webllm/parity/?mode=forced&ref=v2-forced-ref&… in the dev server)
  webllm_forced.py compare --ref evaluation/webllm_parity/v2-forced-ref.json evaluation/webllm_parity/<run>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / "inference"))
from generation import generate_assistant_response  # noqa: E402
from webllm_parity import MERGED  # noqa: E402

TOP = 20


PRODUCTION_PROMPT = (PROJECT / "inference" / "system_prompt.txt").read_text(encoding="utf-8").strip()


def heldout_reference(args, model, processor, device) -> list[dict]:
    """Real held-out replies with the site's system prompt, teacher-forced.

    The rows hold private conversation text; keep the output local
    (self_twin/.gitignore ignores *heldout* files here).
    """
    tokenizer = processor.tokenizer
    rows = [json.loads(line) for line in args.heldout.open(encoding="utf-8") if line.strip()]
    step = max(1, len(rows) // args.count)
    positions = []
    for number, row in enumerate(rows[::step][: args.count]):
        messages = [{"role": "system", "content": PRODUCTION_PROMPT}, *row["messages"][1:-1]]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, tokenize=False)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(prompt_ids) > 1536:
            continue
        reply_ids = tokenizer.encode(row["messages"][-1]["content"] + "<|im_end|>", add_special_tokens=False)
        positions.extend(score_positions(model, tokenizer, device, f"heldout-{number}", 0, prompt, prompt_ids, reply_ids))
    return positions


def score_positions(model, tokenizer, device, label, sample, prompt, prompt_ids, reply_ids) -> list[dict]:
    ids = prompt_ids + reply_ids
    with torch.inference_mode():
        logits = model(input_ids=torch.tensor([ids], device=device)).logits[0, len(prompt_ids) - 1 : -1].float()
    logprobs = torch.log_softmax(logits, -1)
    positions = []
    for k, next_id in enumerate(reply_ids):
        prefix = prompt + tokenizer.decode(reply_ids[:k], skip_special_tokens=False)
        if tokenizer.encode(prefix, add_special_tokens=False) != ids[: len(prompt_ids) + k]:
            continue  # WebLLM would see different tokens for this text
        top = torch.topk(logprobs[k], TOP)
        positions.append({
            "prompt": label, "sample": sample, "k": k, "prefix": prefix, "next_id": next_id,
            "transformers_top": [[int(i), round(float(v), 5)] for v, i in zip(top.values, top.indices)],
        })
    return positions


def build_reference(args) -> None:
    run = json.loads(args.run.read_text(encoding="utf-8"))
    model_dir = MERGED[run["version"]]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(model_dir, dtype=torch.bfloat16, low_cpu_mem_usage=True, local_files_only=True).to(device).eval()
    tokenizer = processor.tokenizer
    if args.heldout:
        positions = heldout_reference(args, model, processor, device)
        out = args.run.with_name(f"{args.name}.json")
        out.write_text(json.dumps({"version": run["version"], "model": str(model_dir), "heldout": str(args.heldout), "positions": positions}, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{len(positions)} held-out positions; saved {out}")
        return
    positions = []
    for item in run["results"]:
        prompt = processor.apply_chat_template(item["messages"], add_generation_prompt=True, enable_thinking=False, tokenize=False)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        for sample in range(args.samples):
            torch.manual_seed(7000 + sample)
            inputs = {"input_ids": torch.tensor([prompt_ids], device=device), "attention_mask": torch.ones(1, len(prompt_ids), dtype=torch.long, device=device)}
            reply = generate_assistant_response(model, processor, inputs, args.max_tokens, do_sample=True, temperature=0.72, top_p=1.0, repetition_penalty=1.12)
            reply_ids = tokenizer.encode(reply + "<|im_end|>", add_special_tokens=False)
            positions.extend(score_positions(model, tokenizer, device, item["prompt"], sample, prompt, prompt_ids, reply_ids))
    out = args.run.with_name(f"{args.name}.json")
    out.write_text(json.dumps({"version": run["version"], "model": str(model_dir), "samples": args.samples, "positions": positions}, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(positions)} positions from {len(run['results'])} prompts x {args.samples} replies; saved {out}")


def compare(args) -> None:
    ref = json.loads(args.ref.read_text(encoding="utf-8"))
    summary = {}
    for path in args.runs:
        run = json.loads(path.read_text(encoding="utf-8"))
        agree, deltas, missing = 0, [], 0
        for position, result in zip(ref["positions"], run["results"]):
            top = dict((i, lp) for i, lp in position["transformers_top"])
            token = result["token_id"]
            agree += int(token == position["transformers_top"][0][0])
            if token in top:
                deltas.append(abs(top[token] - result["logprob"]))
            else:
                missing += 1
        n = len(run["results"])
        summary[path.name] = {
            "positions": n,
            "top1_agreement": agree / n,
            "mean_abs_logprob_diff": sum(deltas) / len(deltas),
            "webllm_top_outside_transformers_top20": missing,
            "bundle": run.get("model_url"),
        }
    print(json.dumps(summary, indent=1))
    if args.output:
        args.output.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    reference = sub.add_parser("reference")
    reference.add_argument("--run", type=Path, required=True, help="a parity run (for its messages and version)")
    reference.add_argument("--name", required=True)
    reference.add_argument("--samples", type=int, default=3)
    reference.add_argument("--max-tokens", type=int, default=40)
    reference.add_argument("--heldout", type=Path, help="score real replies from this held-out chat JSONL instead")
    reference.add_argument("--count", type=int, default=60)
    cmp = sub.add_parser("compare")
    cmp.add_argument("--ref", type=Path, required=True)
    cmp.add_argument("--output", type=Path)
    cmp.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    build_reference(args) if args.command == "reference" else compare(args)


if __name__ == "__main__":
    main()
