#!/bin/sh
# Full v2-recipe LoRA runs for the low-memory site versions, one after another
# (they would compete for the same GPU). Arguments match the published v2 run
# (adapter/experiment-r8-kobajin-clean-v2-memory/preflight.json): full
# kobajin_clean_v2 split, rank 8, LR 5e-5, cosine, 1 epoch, max length 768,
# low-weight rows kept, 300 validation rows, checkpoint every 2,000 steps, seed 42.
# Pass --resume to continue interrupted runs from their newest checkpoint.
set -u
cd "$(dirname "$0")/.."

train() {
  model="$1"; revision="$2"; name="$3"; shift 3
  out="adapter/experiment-r8-kobajin-clean-v2-memory-$name"
  .venv/bin/python training/train.py --model-id "$model" --revision "$revision" --output-dir "$out" \
    --rank 8 --pilot-examples 0 --epochs 1 --learning-rate 5e-5 --low-weight-sample-rate 1.0 \
    --max-seq-length 768 --train-file dataset/kobajin_clean_v2_train.jsonl \
    --validation-file dataset/kobajin_clean_v2_validation.jsonl --validation-examples 300 \
    --eval-steps 2000 --lr-scheduler cosine --seed 42 "$@" >> "$out.log" 2>&1
  echo "$name exit $? $(date '+%F %T')"
}

echo "start $(date '+%F %T')"
train LiquidAI/LFM2.5-1.2B-JP-202606 52b8b4475311a63bf839c6494f78f8ad59d13515 lfm25-1.2b-jp ${1:+"$1"}
train Qwen/Qwen3.5-2B 15852e8c16360a2fea060d615a32b45270f8a8fc qwen35-2b ${1:+"$1"}
