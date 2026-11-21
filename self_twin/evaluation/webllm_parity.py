#!/usr/bin/env python3
"""Compare a WebLLM parity run with Transformers on the same prompts.

A run comes from the dev-only page self_twin/webllm/parity/, which builds
prompts with the site's own code (src/chat_request.js) and decodes with the
real WebLLM engine in the browser. This script loads the version's merged
BF16 checkpoint, builds the Transformers prompt from the same messages with the
training chat template (enable_thinking=False), and reports:

greedy runs
  - whether WebLLM's prompt is the Transformers prompt, character for character
  - WebLLM's and Transformers' greedy replies, and the first token where they differ
  - teacher-forced on WebLLM's tokens: how often Transformers' top token is the
    token WebLLM chose, and the mean absolute difference of the chosen token's
    log probability (WebLLM q4f16_1 vs Transformers BF16)
sampled runs
  - the same number of Transformers samples with the site's decoding, and how
    many replies in each mention the prompt's memory keyword

Nothing here leaves the machine.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMultimodalLM, AutoProcessor, AutoTokenizer

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / "inference"))
from generation import generate_assistant_response  # noqa: E402

MERGED = {
    "v1": PROJECT / "merged" / "qwen35_4b_self_twin_kobajin_clean_v1_20261121",
    "v2": PROJECT / "merged" / "qwen35_4b_self_twin_kobajin_clean_v2_20261121",
    "v3": PROJECT / "merged" / "qwen35_4b_self_twin_experimental_luna_synthetic_v3_20261119",
    "v4": PROJECT / "merged" / "lfm25_1_2b_jp_self_twin_kobajin_clean_v2_20261122",
}
# Words that show a reply used the memory candidate the site offered.
MEMORY_KEYWORDS = {"好きな食べ物は何？": "ナナチキ", "民謡習ってたってほんと？": "民謡", "TextBlurって何作ったの？": "TextBlur"}
EMPTY_THINK = "<think>\n\n</think>\n\n"
KANA = re.compile(r"[\u3040-\u30ff]")
HAN = re.compile(r"[\u4e00-\u9fff]")


def han_without_kana(text: str) -> bool:
    """Kanji and no kana at all, like the reply 「藕」: reads as Chinese, not casual Japanese."""
    return bool(HAN.search(text)) and not KANA.search(text)


def hf_prompt(processor, messages: list[dict]) -> str:
    return processor.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, tokenize=False)


def first_difference(left: list[int], right: list[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def load(model_dir: Path, device: str):
    """Qwen3.5 checkpoints load as multimodal with a processor; LFM2.5 as a causal LM with its tokenizer."""
    if AutoConfig.from_pretrained(model_dir, local_files_only=True).model_type == "lfm2":
        processor = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, low_cpu_mem_usage=True, local_files_only=True)
    else:
        processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForMultimodalLM.from_pretrained(model_dir, dtype=torch.bfloat16, low_cpu_mem_usage=True, local_files_only=True)
    return model.to(device).eval(), processor


def compare_greedy(model, processor, device, item: dict) -> dict:
    tokenizer = getattr(processor, "tokenizer", processor)
    reply = item["replies"][0]
    prompt = hf_prompt(processor, item["messages"])
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

    # WebLLM's prompt array ends with the assistant header and the finished reply,
    # whose text starts with the empty thinking block; cut it after that block.
    joined = "".join(reply.get("prompt_array") or [])
    text = reply["text"]
    cut = joined.rfind(text)
    webllm_prompt = joined[:cut] + (EMPTY_THINK if text.startswith(EMPTY_THINK) else "")
    header = webllm_prompt[webllm_prompt.rfind("<|im_start|>assistant"):]
    # WebLLM adds LFM2.5's BOS <|startoftext|> as an id (system_prefix_token_ids),
    # not as text in the prompt array; the chat template writes it as text.
    bos = tokenizer.bos_token
    if bos and prompt.startswith(bos) and not webllm_prompt.startswith(bos):
        webllm_prompt = bos + webllm_prompt

    prefix_len = len(tokenizer.encode(EMPTY_THINK, add_special_tokens=False)) if text.startswith(EMPTY_THINK) else 0
    webllm_ids = reply.get("output_ids", [])[prefix_len:]
    stop_ids = {tokenizer.convert_tokens_to_ids(name) for name in ("<|im_end|>", "<|endoftext|>") if name in tokenizer.get_vocab()}
    webllm_ids = [token for token in webllm_ids if token not in stop_ids]

    max_new = max(len(reply.get("logprobs", [])), 1)
    inputs = {"input_ids": torch.tensor([prompt_ids], device=device), "attention_mask": torch.ones(1, len(prompt_ids), dtype=torch.long, device=device)}
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, use_cache=True, eos_token_id=list(stop_ids))
    hf_ids = [token for token in out[0, len(prompt_ids):].tolist() if token not in stop_ids]

    # Teacher-force WebLLM's tokens through Transformers.
    agreement, deltas = 0, []
    if webllm_ids:
        forced = torch.tensor([prompt_ids + webllm_ids], device=device)
        with torch.inference_mode():
            logits = model(input_ids=forced).logits[0, len(prompt_ids) - 1 : -1].float()
        logprobs = torch.log_softmax(logits, dim=-1)
        for position, token in enumerate(webllm_ids):
            agreement += int(logprobs[position].argmax().item() == token)
            if position < len(reply["logprobs"]):
                deltas.append(abs(logprobs[position, token].item() - reply["logprobs"][position]["logprob"]))
    return {
        "prompt": item["prompt"],
        "prompt_matches_transformers": webllm_prompt == prompt,
        "webllm_assistant_header": header,
        "webllm": text.removeprefix(EMPTY_THINK).strip(),
        "transformers": tokenizer.decode(hf_ids, skip_special_tokens=True).strip(),
        "greedy_identical": hf_ids == webllm_ids,
        "first_different_token": first_difference(hf_ids, webllm_ids),
        "tokens_compared": len(webllm_ids),
        "transformers_top1_agrees": agreement,
        "mean_abs_logprob_diff": sum(deltas) / len(deltas) if deltas else None,
        "max_abs_logprob_diff": max(deltas) if deltas else None,
    }


def compare_sampled(model, processor, device, item: dict, samples: int, decoding: dict) -> dict:
    keyword = MEMORY_KEYWORDS.get(item["prompt"])
    texts = []
    for sample in range(samples):
        torch.manual_seed(1000 + sample)
        inputs = processor.apply_chat_template(
            item["messages"], add_generation_prompt=True, enable_thinking=False, tokenize=True, return_dict=True, return_tensors="pt"
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        texts.append(generate_assistant_response(
            model, processor, inputs, decoding["max_tokens"], do_sample=True,
            temperature=decoding["temperature"], top_p=decoding["top_p"], repetition_penalty=decoding["repetition_penalty"],
        ))
    webllm = [reply["text"].removeprefix(EMPTY_THINK).strip() for reply in item["replies"]]
    count = (lambda replies: sum(keyword in text for text in replies)) if keyword else (lambda replies: None)
    return {
        "prompt": item["prompt"],
        "memory_keyword": keyword,
        "webllm_keyword_replies": count(webllm),
        "transformers_keyword_replies": count(texts),
        "webllm_han_without_kana": sum(map(han_without_kana, webllm)),
        "transformers_han_without_kana": sum(map(han_without_kana, texts)),
        "webllm": webllm,
        "transformers": texts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path, help="JSON saved by the parity page")
    parser.add_argument("--model", type=Path, help="Merged checkpoint (default: the run's version)")
    parser.add_argument("--output", type=Path, help="Where to write the comparison (default: <run>-compare.json)")
    args = parser.parse_args()

    run = json.loads(args.run.read_text(encoding="utf-8"))
    model_dir = args.model or MERGED[run["version"]]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, processor = load(model_dir, device)

    if run["mode"] == "greedy":
        rows = [compare_greedy(model, processor, device, item) for item in run["results"]]
        compared = sum(row["tokens_compared"] for row in rows)
        deltas = [row["mean_abs_logprob_diff"] for row in rows if row["mean_abs_logprob_diff"] is not None]
        summary = {
            "prompts": len(rows),
            "prompt_matches_transformers": sum(row["prompt_matches_transformers"] for row in rows),
            "greedy_identical": sum(row["greedy_identical"] for row in rows),
            "transformers_top1_agreement": sum(row["transformers_top1_agrees"] for row in rows) / compared if compared else math.nan,
            "mean_abs_logprob_diff": sum(deltas) / len(deltas) if deltas else math.nan,
        }
    else:
        samples = run["request"]["samples"]
        rows = [compare_sampled(model, processor, device, item, samples, run["request"]) for item in run["results"]]
        summary = {
            "prompts": len(rows),
            "samples_per_prompt": samples,
            "keyword_replies_webllm_vs_transformers": {row["prompt"]: [row["webllm_keyword_replies"], row["transformers_keyword_replies"]] for row in rows if row["memory_keyword"]},
            "han_without_kana_webllm_vs_transformers": [sum(row["webllm_han_without_kana"] for row in rows), sum(row["transformers_han_without_kana"] for row in rows)],
        }

    result = {"run": str(args.run), "version": run["version"], "template": run["template"], "mode": run["mode"], "model": str(model_dir), "summary": summary, "rows": rows}
    output = args.output or args.run.with_name(args.run.stem + "-compare.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    for row in rows:
        if run["mode"] == "greedy":
            print(f"- {row['prompt']}\n  WebLLM: {row['webllm']!r}\n  HF:     {row['transformers']!r}\n  first diff token: {row['first_different_token']}  top1 agree: {row['transformers_top1_agrees']}/{row['tokens_compared']}  |Δlogp|: {row['mean_abs_logprob_diff']}")
        else:
            print(f"- {row['prompt']}  keyword {row['memory_keyword']}: WebLLM {row['webllm_keyword_replies']} / HF {row['transformers_keyword_replies']}; kanji-only WebLLM {row['webllm_han_without_kana']} / HF {row['transformers_han_without_kana']}")
            print("  WebLLM:", [text[:30] for text in row["webllm"]])
            print("  HF:    ", [text[:30] for text in row["transformers"]])
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
