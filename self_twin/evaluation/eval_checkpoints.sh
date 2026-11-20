#!/bin/bash
# Wait for a training run to exit, then score every saved checkpoint and the
# final adapter with the held-out kobajin-likeness evaluation (same seed/rows).
set -u
RUN="$1"   # e.g. adapter/experiment-r8-kobajin-clean-v1-lr5e5
TAG="$2"   # output name tag
cd "$(dirname "$0")/.."
while pgrep -f "training/train.py.*$RUN" >/dev/null; do sleep 60; done
ARGS=()
for ckpt in $(ls -d "$RUN"/checkpoints/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
  ARGS+=(--adapter "ck$(basename "$ckpt" | cut -d- -f2)=$ckpt")
done
[ -d "$RUN/adapter" ] && ARGS+=(--adapter "final=$RUN/adapter")
.venv/bin/python evaluation/kobajin_likeness.py --skip-base --prompt-variants prod "${ARGS[@]}" \
  --output "evaluation/kobajin_likeness-$TAG.json" \
  --generations "evaluation/generations-kobajin-likeness-$TAG.jsonl" 2>&1 | grep -E "==|nll|style|Saved|Error|Traceback"
