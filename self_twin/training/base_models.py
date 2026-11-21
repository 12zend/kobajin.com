"""Official post-trained checkpoints this project may train on.

Each base model has its own pin receipt next to this file. The 4B receipt keeps
its original name, base_checkpoint.json, so earlier records and commands stay
valid. The smaller models are candidates for low-memory devices (8 GB Macs,
4 GB tablets, phones); see self_twin/webllm/README.md and
evaluation/base_size_screen.json for how they were chosen.

- kind: "multimodal" loads with AutoModelForMultimodalLM and trains only the
  language-model Linear layers; "causal" loads with AutoModelForCausalLM.
- training_chat_template: a copy of the model's own chat template with
  {% generation %} markers around assistant replies, for models whose template
  TRL does not patch itself. train.py checks that it renders exactly like the
  original template before using it.
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-4B"

BASE_MODELS = {
    "Qwen/Qwen3.5-4B": {"receipt": "base_checkpoint.json", "model_type": "qwen3_5", "kind": "multimodal"},
    "Qwen/Qwen3.5-2B": {"receipt": "base_checkpoint_qwen35_2b.json", "model_type": "qwen3_5", "kind": "multimodal"},
    "Qwen/Qwen3.5-0.8B": {"receipt": "base_checkpoint_qwen35_0_8b.json", "model_type": "qwen3_5", "kind": "multimodal"},
    "LiquidAI/LFM2.5-1.2B-JP-202606": {
        "receipt": "base_checkpoint_lfm25_1_2b_jp.json",
        "model_type": "lfm2",
        "kind": "causal",
        "training_chat_template": "chat_templates/lfm2_5_jp_training.jinja",
    },
}
RECEIPT_NAMES = {model_id: spec["receipt"] for model_id, spec in BASE_MODELS.items()}


def spec(model_id: str) -> dict:
    if model_id not in BASE_MODELS:
        raise SystemExit(f"Unsupported base model {model_id!r}; choose one of {sorted(BASE_MODELS)}")
    return BASE_MODELS[model_id]


def receipt_path(model_id: str) -> Path:
    return HERE / spec(model_id)["receipt"]


def pinned_revision(model_id: str) -> str:
    """The immutable revision recorded in the model's pin receipt."""
    import json

    receipt = json.loads(receipt_path(model_id).read_text(encoding="utf-8"))
    if receipt.get("repo_id") != model_id:
        raise SystemExit(f"Pin receipt {receipt_path(model_id).name} is not for {model_id}")
    return receipt["revision"]


def training_chat_template(model_id: str) -> str | None:
    name = spec(model_id).get("training_chat_template")
    return (HERE / name).read_text(encoding="utf-8") if name else None


def load_base(model_id: str, revision: str, cache_dir: Path, dtype):
    """Load (model, processor) the way each kind needs; processor may be a tokenizer."""
    from transformers import AutoModelForCausalLM, AutoModelForMultimodalLM, AutoProcessor, AutoTokenizer

    kind = spec(model_id)["kind"]
    loader = AutoModelForMultimodalLM if kind == "multimodal" else AutoModelForCausalLM
    processor_loader = AutoProcessor if kind == "multimodal" else AutoTokenizer
    processor = processor_loader.from_pretrained(model_id, revision=revision, cache_dir=cache_dir, local_files_only=True)
    model = loader.from_pretrained(
        model_id, revision=revision, cache_dir=cache_dir, dtype=dtype, low_cpu_mem_usage=True, local_files_only=True
    )
    return model, processor


def text_tokenizer(processor):
    return getattr(processor, "tokenizer", processor)
