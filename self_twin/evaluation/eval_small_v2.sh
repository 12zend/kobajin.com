#!/bin/sh
# Evaluate a low-memory v2-recipe run (training/run_small_v2.sh) the way v2 was
# evaluated, against its own unadapted base and against v2 (the site default):
#
# 1. base_size_screen.py: NLL per character of kobajin's real held-out replies
#    (comparable across tokenizers), with 12 sampled replies.
# 2. kobajin_likeness.py: per-token NLL within the family and the style profile
#    of 60 replies with the site's decoding (v2: kobajin_likeness-clean-v2-all.json).
# 3. memory_use_eval.py: relevant 29 / unrelated 40 memories (v2: memory_use-clean-v2.json).
# 4. run_benchmark.py + summarize_benchmark.py: the fixed 35 prompts with the
#    site's sampling, seed 42 (v2: benchmark-r8-kobajin-clean-v2-final-sampled.json).
#
# Every test uses the same rows, prompts (fitted with the 4B processor), system
# prompt and decoding as v2's. Usage: sh evaluation/eval_small_v2.sh <name> <model id> <screen key>
# A fourth argument (screen, likeness, memory or benchmark) starts at that step.
set -eu
cd "$(dirname "$0")/.."
name="$1"; model="$2"; key="$3"; from="${4:-screen}"
run="adapter/experiment-r8-kobajin-clean-v2-memory-$name"
adapter="$run/adapter"
py=.venv/bin/python
[ -d "$adapter" ] || { echo "missing $adapter"; exit 1; }
order="screen likeness memory benchmark"
step() { case " ${order#*$from} " in *" $1 "*) return 0;; esac; [ "$1" = "$from" ]; }

step screen && {
echo "screen $(date '+%F %T')"
$py evaluation/base_size_screen.py --models "" --adapter "v2recipe-$name=$key=$adapter" \
  --output evaluation/base_size_screen.json --generations evaluation/generations-base-size-screen.jsonl
}
step likeness && {
echo "likeness $(date '+%F %T')"
$py evaluation/kobajin_likeness.py --model-id "$model" --adapter "final=$adapter" --prompt-variants prod \
  --output "evaluation/kobajin_likeness-clean-v2-$name.json" \
  --generations "evaluation/generations-kobajin-likeness-clean-v2-$name.jsonl"
}
step memory && {
echo "memory $(date '+%F %T')"
# PEFT adapter names cannot contain ".".
$py evaluation/memory_use_eval.py --model-id "$model" --adapter "$(echo "$name" | tr . _)=$adapter" \
  --output "evaluation/memory_use-clean-v2-$name.json" \
  --generations "evaluation/generations-memory-use-clean-v2-$name.jsonl"
}
step benchmark && {
echo "benchmark $(date '+%F %T')"
$py evaluation/run_benchmark.py --model-id "$model" --adapter "$adapter" --do-sample \
  --output "evaluation/generations-r8-kobajin-clean-v2-$name-final-sampled.jsonl"
$py evaluation/summarize_benchmark.py --adapter "$adapter" \
  --input "evaluation/generations-r8-kobajin-clean-v2-$name-final-sampled.jsonl" \
  --output "evaluation/benchmark-r8-kobajin-clean-v2-$name-final-sampled.json"
}
echo "done $(date '+%F %T')"
