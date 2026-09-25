#!/bin/bash
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENC=${1:-infused_mlp1}
for cfg in "4 0" "4 1" "4 2" "16 0"; do
  set -- $cfg
  name=learned_prompt_k$1_s$2
  [ -f results/runs/$ENC/$name/summary.json ] && continue
  echo "$(date +%H:%M) start $ENC/$name"
  .venv/bin/python -u prompter_bench/train_learned_prompt.py --encoder $ENC --k $1 --seed $2 > results/logs/extra_${ENC}_${name}.log 2>&1 || echo "FAILED $ENC/$name"
  echo "$(date +%H:%M) done $ENC/$name"
done
