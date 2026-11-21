#!/usr/bin/env python3
"""Guarded BF16 LoRA training entry point for a pinned Qwen3.5 checkpoint (4B by default).

This script intentionally aborts before training when the checkpoint, library
versions, text-layer targets, or assistant-only loss mask are not verified.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
from base_models import DEFAULT_MODEL_ID, RECEIPT_NAMES, load_base, receipt_path, spec, text_tokenizer, training_chat_template

CACHE_DIR = PROJECT / "hf_home" / "hub"


def token_ids(processor, messages, add_generation_prompt=False, chat_template=None, enable_thinking=False):
    result = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_tensors=None,
        chat_template=chat_template,
        enable_thinking=enable_thinking,
    )
    # A processor returns a dict, a tokenizer a BatchEncoding (a Mapping, not a dict).
    if hasattr(result, "keys") and "input_ids" in result:
        result = result["input_ids"]
    if hasattr(result, "tolist"):
        result = result.tolist()
    if result and isinstance(result[0], list):
        result = result[0]
    return result


CONTEXT_HEADER_RE = re.compile(r"^(?:【返信先 \d{2}-\d{2} \d{2}:\d{2}】|\d{2}-\d{2} \d{2}:\d{2} )")


def context_segments(text: str) -> list[tuple[str, bool]]:
    """Keep embedded newlines within a message while finding conversation entries."""
    segments: list[tuple[str, bool]] = []
    current: list[str] = []
    current_is_reply = False
    for line in text.splitlines():
        if CONTEXT_HEADER_RE.match(line):
            if current:
                segments.append(("\n".join(current), current_is_reply))
            current = [line]
            current_is_reply = line.startswith("【返信先 ")
        elif current:
            current.append(line)
        else:
            current = [line]
    if current:
        segments.append(("\n".join(current), current_is_reply))
    return segments


def fit_context_to_length(processor, messages, max_length):
    fitted = [dict(message) for message in messages]
    target_tokens = len(text_tokenizer(processor).encode(fitted[-1]["content"]))
    if target_tokens > max_length - 64:
        return None, {"dropped_long_target": True, "dropped_context_entries": 0, "dropped_reply": False}
    original_context = fitted[1]["content"]
    segments = context_segments(original_context)
    dropped = 0
    dropped_reply = False
    while len(token_ids(processor, fitted)) > max_length:
        non_reply_index = next((i for i, (_, is_reply) in enumerate(segments) if not is_reply), None)
        if non_reply_index is not None:
            segments.pop(non_reply_index)
            dropped += 1
        else:
            reply_index = next((i for i, (_, is_reply) in enumerate(segments) if is_reply), None)
            if reply_index is None:
                return None, {"dropped_long_target": False, "dropped_context_entries": dropped, "dropped_reply": dropped_reply}
            segments.pop(reply_index)
            dropped_reply = True
        fitted[1]["content"] = "\n".join(text for text, _ in segments) or "（長さ制限により会話文脈なし）"
    return fitted, {"dropped_long_target": False, "dropped_context_entries": dropped, "dropped_reply": dropped_reply}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True, help="Pinned official checkpoint revision")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, choices=sorted(RECEIPT_NAMES))
    parser.add_argument("--rank", type=int, choices=(8, 16, 32), required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Separate experiment adapter/checkpoint path")
    parser.add_argument("--pilot-examples", type=int, default=1000, help="Use 0 for the full training split")
    parser.add_argument("--synthetic-train", type=Path, help="Optional curated synthetic conversation JSONL")
    parser.add_argument("--synthetic-validation", type=Path, help="Optional held-out synthetic JSONL for validation")
    parser.add_argument(
        "--synthetic-share",
        type=float,
        default=0.15,
        help="Target fraction of synthetic examples in a pilot when --synthetic-train is supplied",
    )
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument(
        "--low-weight-sample-rate",
        type=float,
        default=0.25,
        help="Keep this fraction of metadata.low_weight_review targets in training; 1.0 keeps all.",
    )
    parser.add_argument("--max-seq-length", type=int, default=0, help="0 derives the cap from the training length distribution")
    parser.add_argument("--preflight-only", action="store_true", help="Load model, inspect targets and mask, then exit before training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-file", type=Path, default=PROJECT / "dataset" / "train.jsonl")
    parser.add_argument("--validation-file", type=Path, default=PROJECT / "dataset" / "validation.jsonl")
    parser.add_argument(
        "--validation-examples",
        type=int,
        default=0,
        help="Fixed-seed validation subset size; 0 keeps the old behavior (200 for pilots, all for full runs)",
    )
    parser.add_argument("--eval-steps", type=int, default=0, help="Evaluate and save every N steps; 0 uses once per epoch")
    parser.add_argument("--lr-scheduler", default="linear", choices=("linear", "cosine", "constant_with_warmup"))
    parser.add_argument("--resume", action="store_true", help="Resume from the newest checkpoint in --output-dir")
    args = parser.parse_args()
    MODEL_ID = args.model_id

    if args.revision in {"main", "latest", "PIN_EXACT_OFFICIAL_REVISION_BEFORE_LOAD"}:
        raise SystemExit(f"Pass an immutable official {args.model_id} revision; moving branches are not accepted")
    if args.pilot_examples < 0:
        raise SystemExit("--pilot-examples must be non-negative")
    if args.epochs <= 0 or args.learning_rate <= 0:
        raise SystemExit("--epochs and --learning-rate must be positive")
    if not 0.0 <= args.low_weight_sample_rate <= 1.0:
        raise SystemExit("--low-weight-sample-rate must be between 0 and 1")
    if not 0.0 <= args.synthetic_share <= 1.0:
        raise SystemExit("--synthetic-share must be between 0 and 1")
    receipt_file = receipt_path(MODEL_ID)
    if not receipt_file.exists():
        raise SystemExit(f"Missing training/{receipt_file.name}; pin and audit the official checkpoint first")
    receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    if receipt.get("repo_id") != MODEL_ID or receipt.get("revision") != args.revision:
        raise SystemExit("Checkpoint receipt does not match the requested official repo and revision")
    if receipt.get("quantization") is not None or not receipt.get("post_trained_model"):
        raise SystemExit("Checkpoint receipt is not an unquantized post-trained model")

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoConfig, set_seed
        from trl import SFTConfig, SFTTrainer
    except ImportError as error:
        raise SystemExit(f"Training dependencies are missing: {error}") from error

    if int(torch.__version__.split(".")[0]) < 2:
        raise SystemExit(f"PyTorch >=2 is required, found {torch.__version__}")
    set_seed(args.seed)
    config = AutoConfig.from_pretrained(
        MODEL_ID, revision=args.revision, cache_dir=CACHE_DIR, local_files_only=True
    )
    model_type = getattr(config, "model_type", None)
    expected_type = spec(MODEL_ID)["model_type"]
    if model_type != expected_type:
        raise SystemExit(f"Expected model_type={expected_type}, got {model_type!r}")
    quantization = getattr(config, "quantization_config", None)
    if quantization:
        raise SystemExit("Quantized checkpoints are not accepted; use the pinned official BF16 Safetensors model")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("No CUDA or MPS device is available; refusing fine-tuning on CPU")
    if not torch.cuda.is_available() and device == "mps":
        # MPS is a local fallback; verify BF16 and all model kernels before pilot training.
        if not hasattr(torch.backends, "mps"):
            raise SystemExit("This PyTorch build has no MPS backend")

    dtype = torch.bfloat16
    model, processor = load_base(MODEL_ID, args.revision, CACHE_DIR, dtype)
    marked_template = training_chat_template(MODEL_ID)
    model.to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    excluded_fragments = ("visual", "vision", "image", "video", "lm_head", "embed")
    # A multimodal checkpoint also holds vision Linear layers; train only the language side.
    language_fragments = ("language_model", "text_model") if spec(MODEL_ID)["kind"] == "multimodal" else ("",)
    target_modules = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and any(fragment in name.lower() for fragment in language_fragments)
        and not any(fragment in name.lower() for fragment in excluded_fragments)
    ]
    if not target_modules:
        visible = [
            name for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear) and not any(x in name.lower() for x in excluded_fragments)
        ][:50]
        raise SystemExit("No text-side Linear modules matched; inspect model structure before choosing targets: " + repr(visible))
    if len(target_modules) < 32:
        raise SystemExit(f"Only {len(target_modules)} text Linear modules found; target selection needs review")

    lora = LoraConfig(
        r=args.rank,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    trainable, total = model.get_nb_trainable_parameters()
    if trainable <= 0 or trainable >= total:
        raise SystemExit(f"Unexpected trainable parameter count: {trainable}/{total}")
    applied_targets = [
        name for name, module in model.named_modules()
        if hasattr(module, "lora_A") and len(module.lora_A) > 0
    ]
    if len(applied_targets) != len(target_modules):
        raise SystemExit(
            f"LoRA target coverage mismatch: found {len(applied_targets)} adapter layers for "
            f"{len(target_modules)} inspected Linear layers"
        )
    if any(any(part in name.lower() for part in excluded_fragments) for name in applied_targets):
        raise SystemExit("A LoRA adapter was attached to a vision, embedding, or LM-head module")

    train_path = args.train_file
    valid_path = args.validation_file
    with train_path.open(encoding="utf-8") as stream:
        train_rows = [json.loads(line) for line in stream if line.strip()]
    with valid_path.open(encoding="utf-8") as stream:
        valid_rows = [json.loads(line) for line in stream if line.strip()]
    synthetic_rows = []
    if args.synthetic_train:
        with args.synthetic_train.open(encoding="utf-8") as stream:
            synthetic_rows = [json.loads(line) for line in stream if line.strip()]
    synthetic_valid_rows = []
    if args.synthetic_validation:
        with args.synthetic_validation.open(encoding="utf-8") as stream:
            synthetic_valid_rows = [json.loads(line) for line in stream if line.strip()]
    low_weight_rows = [row for row in train_rows if row.get("metadata", {}).get("low_weight_review")]
    regular_rows = [row for row in train_rows if not row.get("metadata", {}).get("low_weight_review")]
    selection_rng = random.Random(args.seed + 1)
    selection_rng.shuffle(low_weight_rows)
    selection_rng.shuffle(regular_rows)
    selection_rng.shuffle(synthetic_rows)
    if args.pilot_examples:
        desired_total = min(args.pilot_examples, len(train_rows) + len(synthetic_rows))
        desired_synthetic = min(round(desired_total * args.synthetic_share), len(synthetic_rows))
        desired_original = desired_total - desired_synthetic
        desired_low = min(round(desired_original * args.low_weight_sample_rate), len(low_weight_rows))
        desired_regular = min(desired_original - desired_low, len(regular_rows))
        remaining = desired_original - desired_low - desired_regular
        if remaining:
            add_low = min(remaining, len(low_weight_rows) - desired_low)
            desired_low += add_low
            remaining -= add_low
        if remaining:
            desired_regular += min(remaining, len(regular_rows) - desired_regular)
        selected_train_rows = (
            regular_rows[:desired_regular]
            + low_weight_rows[:desired_low]
            + synthetic_rows[:desired_synthetic]
        )
    else:
        keep_low = round(len(low_weight_rows) * args.low_weight_sample_rate)
        desired_synthetic = len(synthetic_rows)
        selected_train_rows = regular_rows + low_weight_rows[:keep_low] + synthetic_rows
    random.Random(args.seed).shuffle(selected_train_rows)
    selected_low_weight_count = sum(
        bool(row.get("metadata", {}).get("low_weight_review"))
        and not row.get("metadata", {}).get("synthetic")
        for row in selected_train_rows
    )
    selected_synthetic_count = sum(
        bool(row.get("metadata", {}).get("synthetic")) for row in selected_train_rows
    )
    selected_regular_count = sum(
        not row.get("metadata", {}).get("low_weight_review")
        and not row.get("metadata", {}).get("synthetic")
        for row in selected_train_rows
    )
    all_lengths = [len(token_ids(processor, row["messages"])) for row in train_rows]
    if not all_lengths:
        raise SystemExit("Training split is empty")
    percentile_95 = sorted(all_lengths)[math.ceil(0.95 * len(all_lengths)) - 1]
    max_length = args.max_seq_length or max(512, math.ceil(percentile_95 / 256) * 256)

    fitted_train_rows = []
    dropped_long_targets = []
    trimmed_context_entries = 0
    dropped_reply_contexts = 0
    for row in selected_train_rows:
        fitted, fit_report = fit_context_to_length(processor, row["messages"], max_length)
        # fitted is also None when the system prompt and reply alone exceed the
        # budget with every context entry removed; skip those rows the same way.
        if fit_report["dropped_long_target"] or fitted is None:
            dropped_long_targets.extend(row["metadata"]["message_ids"])
            continue
        trimmed_context_entries += fit_report["dropped_context_entries"]
        dropped_reply_contexts += bool(fit_report["dropped_reply"])
        fitted_train_rows.append(
            {"messages": fitted, "chat_template_kwargs": {"enable_thinking": False}}
        )
    if not fitted_train_rows:
        raise SystemExit("No training rows fit the sequence budget without truncating assistant targets")
    train_count_after_filter = len(fitted_train_rows)

    if args.validation_examples:
        random.Random(args.seed + 1).shuffle(valid_rows)
        valid_rows = valid_rows[: min(args.validation_examples, len(valid_rows))]
    elif args.pilot_examples:
        random.Random(args.seed + 1).shuffle(valid_rows)
        valid_rows = valid_rows[: min(200, len(valid_rows))]
    valid_rows.extend(synthetic_valid_rows)
    fitted_valid_rows = []
    for row in valid_rows:
        fitted, fit_report = fit_context_to_length(processor, row["messages"], max_length)
        if fit_report["dropped_long_target"] or fitted is None:
            raise SystemExit("A validation answer exceeds the chosen sequence budget; review max length")
        fitted_valid_rows.append(
            {"messages": fitted, "chat_template_kwargs": {"enable_thinking": False}}
        )
    train_rows = fitted_train_rows
    train_data = Dataset.from_list(train_rows)
    valid_data = Dataset.from_list(fitted_valid_rows)

    run_report = {
        "model_id": MODEL_ID,
        "base_repo": MODEL_ID,
        "revision": args.revision,
        "model_type": model_type,
        "dtype": "bfloat16",
        "device": device,
        "rank": args.rank,
        "seed": args.seed,
        "target_module_count": len(target_modules),
        "target_modules": target_modules,
        "applied_adapter_module_count": len(applied_targets),
        "applied_adapter_modules": applied_targets,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "pilot_examples": len(train_rows),
        "train_rows_available_before_low_weight_sampling": len(low_weight_rows) + len(regular_rows),
        "synthetic_train_rows_available": len(synthetic_rows),
        "synthetic_train_rows_selected": selected_synthetic_count,
        "synthetic_share_requested": args.synthetic_share if args.synthetic_train else 0.0,
        "synthetic_validation_rows": len(synthetic_valid_rows),
        "low_weight_rows_available": len(low_weight_rows),
        "regular_rows_available": len(regular_rows),
        "low_weight_sample_rate": args.low_weight_sample_rate,
        "low_weight_rows_selected": selected_low_weight_count,
        "regular_rows_selected": selected_regular_count,
        "validation_examples_used": len(fitted_valid_rows),
        "train_examples_after_length_filter": train_count_after_filter,
        "training_targets_skipped_over_budget": dropped_long_targets,
        "context_entries_trimmed_to_preserve_assistant_target": trimmed_context_entries,
        "reply_contexts_trimmed_after_recent_context": dropped_reply_contexts,
        "train_sequence_length_p50": sorted(all_lengths)[len(all_lengths) // 2],
        "train_sequence_length_p95": percentile_95,
        "max_sequence_length": max_length,
        "source_sequences_over_max_length": sum(length > max_length for length in all_lengths),
        "assistant_target_truncation": "prevented: over-budget assistant targets are skipped; context is trimmed oldest-first",
        "assistant_mask_verified": False,
        "training_enable_thinking": False,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.epochs,
        "train_file": str(train_path),
        "train_file_sha256": __import__("hashlib").sha256(train_path.read_bytes()).hexdigest(),
        "validation_file": str(valid_path),
        "validation_file_sha256": __import__("hashlib").sha256(valid_path.read_bytes()).hexdigest(),
        "system_prompt_sha256_set": sorted({
            __import__("hashlib").sha256(row["messages"][0]["content"].encode("utf-8")).hexdigest()
            for row in train_rows
        }),
        "lr_scheduler": args.lr_scheduler,
        "eval_steps": args.eval_steps,
        "status": "preflight_only",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "preflight.json").write_text(
        json.dumps(run_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    num_train_epochs = args.epochs
    warmup_ratio = 0.03
    effective_batch_size = 1
    estimated_steps = math.ceil(len(train_rows) / effective_batch_size) * num_train_epochs
    warmup_steps = max(1, math.ceil(estimated_steps * warmup_ratio))
    training_args = SFTConfig(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        optim="adamw_torch",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_checkpointing=True,
        bf16=True,
        max_length=max_length,
        assistant_only_loss=True,
        eval_strategy="steps" if args.eval_steps else "epoch",
        save_strategy="steps" if args.eval_steps else "epoch",
        eval_steps=args.eval_steps or None,
        save_steps=args.eval_steps or None,
        # Step checkpoints are kept so the held-out likeness review can pick one;
        # validation loss alone is not the selection criterion.
        save_total_limit=None if args.eval_steps else 2,
        load_best_model_at_end=not args.eval_steps,
        lr_scheduler_type=args.lr_scheduler,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        prediction_loss_only=True,
        logging_steps=10,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        packing=False,
    )
    if marked_template is not None:
        # The model's own template has no {% generation %} markers and TRL cannot
        # patch it, so use the marked copy, but only if it renders every checked
        # row exactly as the original does (full conversation and reply prompt).
        original_template = processor.chat_template
        for row in train_rows[:300] + valid_rows[:100]:
            messages = row["messages"]
            for rendered_messages, generation in ((messages, False), (messages[:-1], True)):
                original = processor.apply_chat_template(
                    rendered_messages, tokenize=False, add_generation_prompt=generation, enable_thinking=False
                )
                marked = processor.apply_chat_template(
                    rendered_messages,
                    tokenize=False,
                    add_generation_prompt=generation,
                    enable_thinking=False,
                    chat_template=marked_template,
                )
                if marked != original:
                    raise SystemExit("The marked training chat template renders differently from the model's own")
        processor.chat_template = marked_template
        run_report["original_chat_template_sha256"] = __import__("hashlib").sha256(
            original_template.encode("utf-8")
        ).hexdigest()
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        processing_class=processor,
    )

    tokenized_lengths = [len(row["input_ids"]) for row in trainer.train_dataset]
    if any(length > max_length for length in tokenized_lengths):
        raise SystemExit("TRL tokenization exceeded max_length; refusing to truncate assistant targets")
    if not all(
        "labels" in row and any(label != -100 for label in row["labels"])
        for row in trainer.train_dataset
    ):
        raise SystemExit("At least one tokenized training sample has no supervised assistant tokens")

    # Inspect the actual collator labels before any optimizer step.
    mask_check_index = next(
        (i for i, row in enumerate(train_rows) if len(token_ids(processor, row["messages"])) <= max_length),
        None,
    )
    if mask_check_index is None:
        raise SystemExit("No untruncated sample is available to verify the assistant-only loss mask")
    first_row = trainer.train_dataset[mask_check_index]
    batch = trainer.data_collator([first_row])
    labels = batch.get("labels")
    input_ids = batch.get("input_ids")
    if labels is None or input_ids is None:
        raise SystemExit("TRL did not return labels/input_ids; assistant-only mask could not be verified")
    label_row = labels[0]
    input_row = input_ids[0]
    active_positions = torch.where(label_row != -100)[0]
    if active_positions.numel() == 0 or active_positions.numel() >= input_row.numel():
        raise SystemExit("Assistant-only mask is empty or covers the full prompt; refusing to train")
    first_active = int(active_positions[0].item())
    from trl.data_utils import _tokenize

    messages = train_rows[mask_check_index]["messages"]
    rendered = _tokenize(
        processor,
        messages,
        chat_template=trainer.chat_template or processor.chat_template,
        return_assistant_tokens_mask=True,
        enable_thinking=False,
    )
    expected_ids = torch.as_tensor(rendered["input_ids"], dtype=input_row.dtype, device=input_row.device)
    raw_assistant_mask = torch.as_tensor(
        rendered["assistant_masks"], dtype=torch.bool, device=label_row.device
    )
    if expected_ids.numel() > input_row.numel() or raw_assistant_mask.numel() != expected_ids.numel():
        raise SystemExit("Rendered mask and collated sequence lengths are inconsistent")
    if not torch.equal(input_row[: expected_ids.numel()], expected_ids):
        raise SystemExit("TRL's stored training tokens differ from the template rendered for mask verification")
    sample_assistant_mask = torch.zeros_like(input_row, dtype=torch.bool)
    sample_assistant_mask[: raw_assistant_mask.numel()] = raw_assistant_mask
    observed_mask = label_row != -100
    if not torch.equal(observed_mask, sample_assistant_mask):
        raise SystemExit("Collator labels do not exactly match TRL's assistant generation mask; refusing to train")

    prompt_ids = token_ids(
        getattr(trainer, "processing_class", processor),
        messages[:-1],
        add_generation_prompt=True,
        chat_template=trainer.chat_template or processor.chat_template,
        enable_thinking=False,
    )
    if first_active != len(prompt_ids):
        raise SystemExit(
            f"Loss mask starts at token {first_active}, but rendered system/user prompt has "
            f"{len(prompt_ids)} tokens; prompt boundary is not exact"
        )
    run_report.update(
        {
            "assistant_mask_verified": True,
            "assistant_mask_source": "TRL training chat template (or the checked marked copy) and collator labels compared token-for-token",
            "assistant_supervised_tokens": int(active_positions.numel()),
            "first_supervised_token_index": first_active,
            "rendered_prompt_token_count": len(prompt_ids),
            "training_chat_template_sha256": __import__("hashlib").sha256(
                (trainer.chat_template or processor.chat_template).encode("utf-8")
            ).hexdigest(),
            "training_tokenized_sequence_length_p95": sorted(tokenized_lengths)[
                math.ceil(0.95 * len(tokenized_lengths)) - 1
            ],
            "warmup_ratio": warmup_ratio,
            "warmup_steps": warmup_steps,
            "estimated_total_steps": estimated_steps,
            "status": "mask_verified_ready_for_manual_start",
        }
    )
    (args.output_dir / "preflight.json").write_text(
        json.dumps(run_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    concise_report = {
        key: value for key, value in run_report.items()
        if key not in {"target_modules", "applied_adapter_modules", "training_targets_skipped_over_budget"}
    }
    print(json.dumps(concise_report, ensure_ascii=False, indent=2))
    if args.preflight_only:
        return
    resume = None
    if args.resume:
        checkpoints = sorted(
            (args.output_dir / "checkpoints").glob("checkpoint-*"),
            key=lambda path: int(path.name.rsplit("-", 1)[1]),
        )
        if not checkpoints:
            raise SystemExit("--resume was given but no checkpoint exists yet")
        resume = str(checkpoints[-1])
        print(f"Resuming from {resume}", flush=True)
    train_result = trainer.train(resume_from_checkpoint=resume)
    model.config.use_cache = True
    trainer.save_model(str(args.output_dir / "adapter"))
    processor.save_pretrained(str(args.output_dir / "adapter"))
    trainer.save_state()
    metrics = {
        "train": train_result.metrics,
        "final_validation": trainer.evaluate(),
        "log_history": trainer.state.log_history,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
    }
    (args.output_dir / "training_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
