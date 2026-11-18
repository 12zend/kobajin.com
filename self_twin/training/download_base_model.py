#!/usr/bin/env python3
"""Download and pin only the official post-trained Qwen3.5-4B checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

PROJECT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3.5-4B"
CACHE_DIR = PROJECT / "hf_home" / "hub"
RECEIPT = Path(__file__).resolve().parent / "base_checkpoint.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Allow resolving a newer main revision")
    args = parser.parse_args()

    old = None
    if RECEIPT.exists():
        old = json.loads(RECEIPT.read_text(encoding="utf-8"))
    info = HfApi().model_info(MODEL_ID, revision="main")
    revision = info.sha
    if not revision:
        raise SystemExit("Hugging Face did not return an immutable commit SHA")
    if old and old.get("revision") != revision and not args.refresh:
        raise SystemExit(
            f"Official main moved from {old.get('revision')} to {revision}; "
            "existing checkpoint is preserved. Review and rerun with --refresh explicitly."
        )

    snapshot = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=revision,
            cache_dir=str(CACHE_DIR),
            max_workers=4,
        )
    )
    config_path = snapshot / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_5":
        raise SystemExit(f"Unexpected model_type in official checkpoint: {config.get('model_type')!r}")
    if config.get("quantization") or config.get("quantization_config"):
        raise SystemExit("Official checkpoint unexpectedly reports quantization; refusing to train")
    if not any(snapshot.glob("*.safetensors")):
        raise SystemExit("No Safetensors weights were downloaded")

    weight_files = sorted(snapshot.glob("*.safetensors"))
    tokenizer_path = snapshot / "tokenizer.json"
    template_path = snapshot / "chat_template.jinja"
    receipt = {
        "repo_id": MODEL_ID,
        "revision": revision,
        "resolved_from": "official main at download time, then pinned to immutable commit",
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "post_trained_model": True,
        "quantization": None,
        "snapshot_path": str(snapshot),
        "config_sha256": sha256_file(config_path),
        "tokenizer_sha256": sha256_file(tokenizer_path) if tokenizer_path.exists() else None,
        "chat_template_sha256": sha256_file(template_path) if template_path.exists() else None,
        "chat_template_has_generation_markers": (
            "{% generation %}" in template_path.read_text(encoding="utf-8")
            and "{% endgeneration %}" in template_path.read_text(encoding="utf-8")
        ) if template_path.exists() else False,
        "safetensors_files": [
            {"name": path.name, "bytes": path.stat().st_size} for path in weight_files
        ],
        "safetensors_total_bytes": sum(path.stat().st_size for path in weight_files),
        "downloaded_locally": True,
        "chat_history_uploaded": False,
    }
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
