#!/bin/sh
# 1,000-example pilots of the v2 recipe on the low-memory base candidates, then
# per-character NLL on the held-out test split (evaluation/base_size_screen.py).
# Everything else matches v2: rank 8, LR 5e-5, cosine, 1 epoch, max length 768,
# low-weight rows kept, 300 validation rows, seed 42.
set -u
cd "$(dirname "$0")/.."

pilot() {
  out="adapter/pilot-r8-v2recipe-1000-$3"
  .venv/bin/python training/train.py --model-id "$1" --revision "$2" --output-dir "$out" \
    --rank 8 --pilot-examples 1000 --epochs 1 --learning-rate 5e-5 --low-weight-sample-rate 1.0 \
    --max-seq-length 768 --train-file dataset/kobajin_clean_v2_train.jsonl \
    --validation-file dataset/kobajin_clean_v2_validation.jsonl --validation-examples 300 \
    --lr-scheduler cosine --seed 42 > "$out.log" 2>&1
  echo "$3 exit $? $(date +%T)"
}

date +%T
pilot Qwen/Qwen3.5-0.8B 2fc06364715b967f1860aea9cf38778875588b17 qwen35-0.8b
pilot LiquidAI/LFM2.5-1.2B-JP-202606 52b8b4475311a63bf839c6494f78f8ad59d13515 lfm25-jp
.venv/bin/python evaluation/base_size_screen.py --models "" \
  --adapter pilot1000-qwen35-0.8b=qwen35-0.8b=adapter/pilot-r8-v2recipe-1000-qwen35-0.8b/adapter \
  --adapter pilot1000-lfm25-1.2b-jp=lfm25-1.2b-jp=adapter/pilot-r8-v2recipe-1000-lfm25-jp/adapter \
  --output evaluation/base_size_screen.json --generations evaluation/generations-base-size-screen.jsonl 2>&1 \
  | grep -E "^==|^\{|Error|Traceback"
date +%T
