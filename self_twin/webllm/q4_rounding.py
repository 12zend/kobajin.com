#!/usr/bin/env python3
"""Ways to round weights into MLC's q4f16_1 format, and a check of each.

MLC's q4f16_1 stores every weight it quantizes (all linear layers and the tied
embedding / output head of Qwen3.5) as 4-bit levels -7..7 with one float16
scale per 32 consecutive input weights. `mlc_llm convert_weight` picks the
scale as max|w| / 7 and rounds to the nearest level ("rtn" here).

Any choice of levels and scales in that format runs on WebLLM's prebuilt
Qwen3.5-4B q4f16_1 library. This module offers better choices:

  rtn   MLC's own rounding (what convert_weight does)
  mse   per group, the scale (max|w| / 7 shrunk by up to 30 %) that minimizes
        the squared rounding error; outliers are clipped to ±7 levels
  gptq  GPTQ (Frantar et al., 2022) with the same group scales as `mse`: it
        rounds input columns in order and moves each column's error onto the
        columns not yet rounded, weighted by calibration activations; the
        tied embedding / output head uses `mse`
  gptq_head  `gptq`, plus GPTQ for the tied output head

`export` writes a checkpoint whose quantized weights already sit on the chosen
q4 grid (as float16), so running `mlc_llm convert_weight --quantization
q4f16_1` on it reproduces those levels exactly: MLC's max|w| / 7 then recovers
the chosen scale, and rounding recovers the chosen levels. The bundle layout is
therefore unchanged.

`check` measures, on fixed prompts, how far each rounding moves the model from
its BF16 checkpoint: KL divergence of the next-token distribution and top-1
agreement along the BF16 greedy reply.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
GROUP = 32
MAX_LEVEL = 7
SHRINK = [1.0 - 0.02 * step for step in range(16)]  # 1.00 … 0.70
LINEAR_NAMES = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
)


def quantized_modules(model) -> list[tuple[str, torch.nn.Module]]:
    """Every module whose weight MLC stores as q4f16_1, in forward order."""
    language = model.model.language_model
    modules = [("embed_tokens", language.embed_tokens)]
    for name, module in language.layers.named_modules():
        if isinstance(module, torch.nn.Linear) and name.split(".")[-1] in LINEAR_NAMES:
            modules.append((f"layers.{name}", module))
    return modules


def group_view(weight: torch.Tensor) -> torch.Tensor:
    rows, cols = weight.shape
    assert cols % GROUP == 0, (rows, cols)
    return weight.reshape(rows, cols // GROUP, GROUP)


def rtn_scale(weight: torch.Tensor) -> torch.Tensor:
    """MLC's scale: max|w| / 7 per group, in float16."""
    groups = group_view(weight.float())
    return (groups.abs().amax(-1) .half().float() / MAX_LEVEL).half().float()


def mse_scale(weight: torch.Tensor) -> torch.Tensor:
    groups = group_view(weight.float())
    peak = groups.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    best_err = torch.full(peak.shape[:-1], math.inf, device=weight.device)
    best = torch.zeros_like(best_err)
    for shrink in SHRINK:
        scale = (peak * shrink / MAX_LEVEL).half().float()
        levels = torch.clamp(torch.round(groups / scale), -MAX_LEVEL, MAX_LEVEL)
        err = ((levels * scale - groups) ** 2).sum(-1)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best = torch.where(better, scale.squeeze(-1), best)
    return best


def dequantize_with(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    groups = group_view(weight.float())
    levels = torch.clamp(torch.round(groups / scale.unsqueeze(-1)), -MAX_LEVEL, MAX_LEVEL)
    return (levels * scale.unsqueeze(-1)).reshape(weight.shape)


def gptq(weight: torch.Tensor, hessian: torch.Tensor, damp: float = 0.01, block: int = 128, rows: int = 16384) -> torch.Tensor:
    """GPTQ with fixed `mse` group scales; output rows are independent, so large
    matrices (the 248k-row output head) are done a slice of rows at a time."""
    if weight.shape[0] > rows:
        return torch.cat([gptq(weight[start:start + rows], hessian, damp, block, rows) for start in range(0, weight.shape[0], rows)])
    return _gptq_rows(weight, hessian, damp, block)


def _gptq_rows(weight: torch.Tensor, hessian: torch.Tensor, damp: float, block: int) -> torch.Tensor:
    W = weight.float().clone()
    rows, cols = W.shape
    scale = mse_scale(weight)  # [rows, cols / GROUP]
    H = hessian.float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H += torch.eye(cols, device=H.device) * damp * torch.mean(torch.diag(H))
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W)
    for start in range(0, cols, block):
        end = min(start + block, cols)
        W1 = W[:, start:end].clone()
        Q1 = torch.zeros_like(W1)
        E1 = torch.zeros_like(W1)
        H1 = Hinv[start:end, start:end]
        for offset in range(end - start):
            column = start + offset
            w = W1[:, offset]
            s = scale[:, column // GROUP]
            q = torch.clamp(torch.round(w / s), -MAX_LEVEL, MAX_LEVEL) * s
            Q1[:, offset] = q
            err = (w - q) / H1[offset, offset]
            W1[:, offset:] -= err.unsqueeze(1) * H1[offset, offset:].unsqueeze(0)
            E1[:, offset] = err
        Q[:, start:end] = Q1
        W[:, end:] -= E1 @ Hinv[start:end, end:]
    return Q


class _StopForward(Exception):
    pass


def collect_hessians(model, processor, texts: list[str], device: str, names: list[str], stop_after, modules: dict | None = None) -> dict:
    """X^T X of the inputs of `names` over the calibration texts.

    Each forward pass stops after module `stop_after`, so only the layers
    being rounded in this pass are run.
    """
    modules = modules or dict(quantized_modules(model))
    hessians, counts, hooks = {}, {}, []
    for name in names:
        module = modules[name]
        hessians[name] = torch.zeros(module.in_features, module.in_features, device=device)
        counts[name] = 0

        def hook(mod, args, _name=name):
            x = args[0].detach().reshape(-1, args[0].shape[-1]).float()
            total = counts[_name] + x.shape[0]
            hessians[_name] *= counts[_name] / total
            x = x * math.sqrt(2 / total)
            hessians[_name] += x.T @ x
            counts[_name] = total

        hooks.append(module.register_forward_pre_hook(hook))

    def stop(*_args):
        raise _StopForward

    hooks.append(stop_after.register_forward_hook(stop))
    tokenizer = processor.tokenizer
    try:
        with torch.inference_mode():
            for text in texts:
                ids = tokenizer(text, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=1024).to(device)
                try:
                    model(**ids)
                except _StopForward:
                    pass
    finally:
        for handle in hooks:
            handle.remove()
    return hessians


def apply_rounding(model, method: str, device: str = "mps", processor=None, texts: list[str] | None = None, layers_per_pass: int = 4) -> tuple[dict, dict]:
    """Round every q4f16_1 weight in place.

    Returns each module's weight MSE and its rounded float16 weight. The model
    keeps its own dtype for the forward passes GPTQ calibration runs; export
    writes the float16 weights.

    GPTQ rounds the decoder layers a few at a time, in order, so each pass's
    calibration inputs already come through the layers rounded before it.
    With `gptq` the tied embedding / output head uses the `mse` rounding;
    `gptq_head` also rounds it with GPTQ, calibrated as the output head.
    """
    stats, rounded = {}, {}

    def store(name, module, q):
        w = module.weight.data
        stats[name] = float(((q - w.to(device).float()) ** 2).mean().item())
        rounded[name] = q.half().cpu()
        module.weight.data = q.half().to(device=w.device, dtype=w.dtype)

    modules = quantized_modules(model)
    use_gptq = method in ("gptq", "gptq_head")
    if method in ("rtn", "mse") or use_gptq:
        for name, module in modules:
            if use_gptq and name != "embed_tokens":
                continue
            w = module.weight.data.to(device)
            scale = rtn_scale(w) if method == "rtn" else mse_scale(w)
            store(name, module, dequantize_with(w, scale))
    else:
        raise ValueError(method)
    if use_gptq:
        if not texts:
            raise ValueError("gptq needs calibration texts")
        layers = model.model.language_model.layers
        by_layer: dict[int, list[tuple[str, torch.nn.Module]]] = {}
        for name, module in modules:
            if name != "embed_tokens":
                by_layer.setdefault(int(name.split(".")[1]), []).append((name, module))
        for first in range(0, len(layers), layers_per_pass):
            chunk = [index for index in range(first, min(first + layers_per_pass, len(layers)))]
            names = [name for index in chunk for name, _ in by_layer[index]]
            started = time.time()
            hessians = collect_hessians(model, processor, texts, device, names, layers[chunk[-1]])
            for index in chunk:
                for name, module in by_layer[index]:
                    store(name, module, gptq(module.weight.data.to(device), hessians.pop(name)))
            print(f"  gptq layers {chunk[0]}-{chunk[-1]}: {time.time() - started:.0f}s", flush=True)
            if device == "mps":
                torch.mps.empty_cache()
    if method == "gptq_head":
        # The tied embedding is also the output head; round it for its use as
        # the head, calibrated on the final hidden states.
        started = time.time()
        head = model.lm_head
        embed = dict(modules)["embed_tokens"]
        original = embed.weight.data.detach().clone()
        hessian = collect_hessians(model, processor, texts, device, ["embed_tokens"], head, {"embed_tokens": head})["embed_tokens"]
        store("embed_tokens", embed, gptq(original.to(device), hessian))
        print(f"  gptq output head: {time.time() - started:.0f}s", flush=True)
    return stats, rounded


def load_model(path: Path, dtype, device: str):
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(path, dtype=dtype, low_cpu_mem_usage=True, local_files_only=True).to(device)
    return processor, model.eval()


def parity_prompts(processor, run: Path) -> list[str]:
    data = json.loads(run.read_text(encoding="utf-8"))
    return [
        processor.apply_chat_template(item["messages"], add_generation_prompt=True, enable_thinking=False, tokenize=False)
        for item in data["results"]
    ]


def calibration_texts(processor, path: Path, count: int) -> list[str]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    step = max(1, len(rows) // count)
    texts = []
    for row in rows[::step][:count]:
        texts.append(processor.apply_chat_template(row["messages"], tokenize=False, enable_thinking=False))
    return texts


def reference(model, processor, prompts: list[str], device: str, new_tokens: int):
    """BF16 greedy replies to the parity prompts, as (token ids, prompt length)."""
    tokenizer = processor.tokenizer
    stop = [tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.convert_tokens_to_ids("<|endoftext|>")]
    sequences = []
    with torch.inference_mode():
        for prompt in prompts:
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            x = torch.tensor([ids], device=device)
            out = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=new_tokens, do_sample=False, eos_token_id=stop)
            sequences.append((out[0].tolist(), len(ids)))
    return sequences


def heldout_sequences(processor, path: Path, count: int) -> list[tuple[list[int], int]]:
    """Real replies from a held-out split, in the site's prompt format, as (ids, prompt length)."""
    tokenizer = processor.tokenizer
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    step = max(1, len(rows) // count)
    sequences = []
    for row in rows[::step][:count]:
        prompt = processor.apply_chat_template(row["messages"][:-1], add_generation_prompt=True, enable_thinking=False, tokenize=False)
        reply = row["messages"][-1]["content"] + "<|im_end|>"
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)[-1536:]
        sequences.append((prompt_ids + tokenizer.encode(reply, add_special_tokens=False), len(prompt_ids)))
    return sequences


def reply_logprobs(model, sequences, device: str) -> list[torch.Tensor]:
    out = []
    with torch.inference_mode():
        for seq, prompt_len in sequences:
            logits = model(input_ids=torch.tensor([seq], device=device)).logits[0, prompt_len - 1 : -1].float()
            out.append(torch.log_softmax(logits, -1).half().cpu())
    return out


def kl_stats(model, sequences, ref: list[torch.Tensor], device: str) -> dict:
    kls, agree, total = [], 0, 0
    for lp, r in zip(reply_logprobs(model, sequences, device), ref):
        lp, r = lp.float(), r.float()
        kls.extend((r.exp() * (r - lp)).sum(-1).tolist())
        agree += int((lp.argmax(-1) == r.argmax(-1)).sum())
        total += lp.shape[0]
    return {"mean_kl": sum(kls) / len(kls), "max_kl": max(kls), "top1_agreement": agree / total, "positions": total}


def greedy_identical(model, sequences, device: str, tokenizer) -> int:
    stop = [tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.convert_tokens_to_ids("<|endoftext|>")]
    same = 0
    with torch.inference_mode():
        for seq, prompt_len in sequences:
            prompt = torch.tensor([seq[:prompt_len]], device=device)
            out = model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt), max_new_tokens=len(seq) - prompt_len, do_sample=False, eos_token_id=stop)
            same += int(out[0, prompt_len:].tolist() == seq[prompt_len:])
    return same


def read_bundle_q4(bundle: Path, name: str) -> torch.Tensor:
    """Dequantize one q4f16_1 weight from a converted bundle: (level - 7) * scale."""
    import numpy as np

    cache = json.loads((bundle / "tensor-cache.json").read_text())
    found = {}
    for shard in cache["records"]:
        for record in shard["records"]:
            if record["name"] in (f"{name}.q_weight", f"{name}.q_scale"):
                raw = (bundle / shard["dataPath"]).read_bytes()[record["byteOffset"]: record["byteOffset"] + record["nbytes"]]
                dtype = np.uint32 if record["dtype"] == "uint32" else np.float16
                found[record["name"].rsplit(".", 1)[1]] = np.frombuffer(raw, dtype=dtype).reshape(record["shape"])
    packed, scale = found["q_weight"], found["q_scale"].astype(np.float32)
    levels = np.stack([(packed >> (4 * i)) & 0xF for i in range(8)], axis=-1).reshape(packed.shape[0], -1).astype(np.float32)
    return torch.from_numpy((levels - MAX_LEVEL) * np.repeat(scale, GROUP, axis=1))


def verify(bundle: Path, checkpoint: Path, layers: list[int]) -> dict:
    """Check a bundle converted from an `export` checkpoint holds the exported weights."""
    from safetensors import safe_open

    index_file = checkpoint / "model.safetensors.index.json"
    if index_file.exists():
        index = json.loads(index_file.read_text())["weight_map"]
    else:
        with safe_open(checkpoint / "model.safetensors", "pt") as handle:
            index = {key: "model.safetensors" for key in handle.keys()}

    def hf(name):
        with safe_open(checkpoint / index[name], "pt") as handle:
            return handle.get_tensor(name).float()

    prefix = "model.language_model"
    pairs = [("model.embed_tokens", lambda: hf(f"{prefix}.embed_tokens.weight"))]
    for n in layers:
        pairs.append((f"model.layers.{n}.mlp.down_proj", lambda n=n: hf(f"{prefix}.layers.{n}.mlp.down_proj.weight")))
        pairs.append((f"model.layers.{n}.mlp.gate_up_proj", lambda n=n: torch.cat([hf(f"{prefix}.layers.{n}.mlp.gate_proj.weight"), hf(f"{prefix}.layers.{n}.mlp.up_proj.weight")])))
        attn = f"{prefix}.layers.{n}.self_attn.o_proj.weight"
        if attn in index:
            pairs.append((f"model.layers.{n}.self_attn.o_proj", lambda attn=attn: hf(attn)))
        else:
            pairs.append((f"model.layers.{n}.linear_attn.out_proj", lambda n=n: hf(f"{prefix}.layers.{n}.linear_attn.out_proj.weight")))
    report = {}
    for name, load in pairs:
        expected, actual = load(), read_bundle_q4(bundle, name)
        scale = expected.reshape(expected.shape[0], -1, GROUP).abs().amax(-1).repeat_interleave(GROUP, 1) / MAX_LEVEL
        report[name] = {"max_error_in_levels": float(((actual - expected).abs() / scale.clamp_min(1e-12)).max())}
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="compare roundings against BF16 on the parity prompts")
    check.add_argument("--model", type=Path, required=True)
    check.add_argument("--run", type=Path, required=True, help="a parity run (for its prompts)")
    check.add_argument("--methods", default="fp16,rtn,mse,gptq")
    check.add_argument("--calibration", type=Path, help="chat JSONL for GPTQ calibration")
    check.add_argument("--calibration-count", type=int, default=128)
    check.add_argument("--new-tokens", type=int, default=64)
    check.add_argument("--heldout", type=Path, help="chat JSONL whose real replies are also scored")
    check.add_argument("--heldout-count", type=int, default=40)
    check.add_argument("--output", type=Path)
    export = sub.add_parser("export", help="write a checkpoint already on the chosen q4 grid")
    export.add_argument("--model", type=Path, required=True)
    export.add_argument("--method", choices=["rtn", "mse", "gptq", "gptq_head"], required=True)
    export.add_argument("--calibration", type=Path)
    export.add_argument("--calibration-count", type=int, default=128)
    export.add_argument("--output", type=Path, required=True)
    check_bundle = sub.add_parser("verify", help="check a converted bundle holds an exported checkpoint's weights")
    check_bundle.add_argument("--bundle", type=Path, required=True)
    check_bundle.add_argument("--checkpoint", type=Path, required=True)
    check_bundle.add_argument("--layers", default="0,3,17,31")
    args = parser.parse_args()
    if args.command == "verify":
        print(json.dumps(verify(args.bundle, args.checkpoint, [int(n) for n in args.layers.split(",")]), indent=1))
        return
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    if args.command == "check":
        processor, model = load_model(args.model, torch.bfloat16, device)
        assert model.lm_head.weight is model.model.language_model.embed_tokens.weight, "output head must be tied"
        prompts = parity_prompts(processor, args.run)
        sequences = reference(model, processor, prompts, device, args.new_tokens)
        ref = reply_logprobs(model, sequences, device)
        heldout = heldout_sequences(processor, args.heldout, args.heldout_count) if args.heldout else []
        heldout_ref = reply_logprobs(model, heldout, device) if heldout else []
        original = {name: module.weight.data.detach().cpu().clone() for name, module in quantized_modules(model)}
        model.to(torch.float16)
        methods = args.methods.split(",")
        texts = None
        if any(method.startswith("gptq") for method in methods):
            if not args.calibration:
                raise SystemExit("--calibration is required for gptq")
            texts = calibration_texts(processor, args.calibration, args.calibration_count)
        results = {}
        for method in methods:
            for name, module in quantized_modules(model):
                module.weight.data = original[name].to(device=device, dtype=torch.float16)
            started = time.time()
            weight_mse = apply_rounding(model, method, device, processor, texts)[0] if method != "fp16" else {}
            results[method] = {"parity_prompts": kl_stats(model, sequences, ref, device)}
            results[method]["parity_prompts"]["greedy_identical"] = greedy_identical(model, sequences, device, processor.tokenizer)
            if heldout:
                results[method]["heldout_replies"] = kl_stats(model, heldout, heldout_ref, device)
            if weight_mse:
                results[method]["mean_weight_mse"] = sum(weight_mse.values()) / len(weight_mse)
            results[method]["seconds"] = round(time.time() - started)
            print(method, json.dumps(results[method]), flush=True)
        if args.output:
            args.output.write_text(json.dumps({"model": str(args.model), "run": str(args.run), "results": results}, indent=1) + "\n")
        return

    # Unquantized parameters (norms, A_log, conv) keep their checkpoint dtype;
    # only the rounded weights are written as float16.
    processor, model = load_model(args.model, torch.bfloat16, device)
    texts = None
    if args.method.startswith("gptq"):
        if not args.calibration:
            raise SystemExit("--calibration is required for gptq")
        texts = calibration_texts(processor, args.calibration, args.calibration_count)
    _, rounded = apply_rounding(model, args.method, device, processor, texts)
    model.to("cpu")
    for name, module in quantized_modules(model):
        module.weight.data = rounded[name]
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    processor.save_pretrained(args.output)
    (args.output / "q4_rounding.json").write_text(json.dumps({
        "source": str(args.model), "method": args.method, "group_size": GROUP,
        "calibration": str(args.calibration) if args.calibration else None,
        "calibration_count": args.calibration_count if args.calibration else 0,
    }, indent=1) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
