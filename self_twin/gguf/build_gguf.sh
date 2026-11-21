#!/bin/zsh
# Convert the merged BF16 checkpoints of every published kobajinAI version to
# GGUF Q4_K_M for llama.cpp. Usage: self_twin/gguf/build_gguf.sh [v1 v2 v3]
# LLAMA_CPP must point at a llama.cpp checkout whose commit matches the
# installed llama.cpp tools (checked below).
# --no-mtp: the merged checkpoints have no MTP tensors (Transformers drops them
# on merge) although config.json still sets mtp_num_hidden_layers = 1.
set -euo pipefail
cd "$(dirname "$0")/../.."

LLAMA_CPP=${LLAMA_CPP:-/Users/kobajin/repo/qwen3.5-2b/tools/llama.cpp}
PY=self_twin/mlc-venv/bin/python
OUT=self_twin/gguf/build

typeset -A SRC
SRC[v1]=self_twin/merged/qwen35_4b_self_twin_kobajin_clean_v1_20261121
SRC[v2]=self_twin/merged/qwen35_4b_self_twin_kobajin_clean_v2_20261121
SRC[v3]=self_twin/merged/qwen35_4b_self_twin_experimental_luna_synthetic_v3_20261119

want=$(git -C "$LLAMA_CPP" rev-parse --short=9 HEAD)
have=$(llama-cli --version 2>&1 | sed -n 's/.*commit \([0-9a-f]*\).*/\1/p')
[[ "$want" == "$have"* || "$have" == "$want"* ]] || { echo "llama.cpp checkout $want != installed llama.cpp $have" >&2; exit 1; }

mkdir -p "$OUT"
for v in ${@:-v1 v2 v3}; do
  bf16="$OUT/kobajinAI-$v-BF16.gguf"
  q4="$OUT/kobajinAI-$v-Q4_K_M.gguf"
  PYTHONPATH="$LLAMA_CPP/gguf-py" "$PY" "$LLAMA_CPP/convert_hf_to_gguf.py" "${SRC[$v]}" \
    --outtype bf16 --no-mtp --model-name "kobajinAI $v" --outfile "$bf16"
  llama-quantize "$bf16" "$q4" Q4_K_M
  rm "$bf16"
  shasum -a 256 "$q4"
done
