#!/bin/bash
# Evaluates every finished run that has no test results yet, whenever the GPU has room. Usage: bash run_eval_lane.sh
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
E=".venv/bin/python -u prompter_bench/evaluate.py"
while true; do
  for enc in infused_mlp1 infused_mlp05; do
    [ -f exp/feature_cache/$enc/done.json ] || continue
    for m in oracle_gt cgnet_official cgnet_finetuned; do
      [ -f results/eval/$enc/$m.json ] && continue
      free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
      [ "$free" -lt 15000 ] && continue
      echo "$(date +%H:%M) eval $enc/$m"; $E --encoder $enc --methods ${m%_gt} >> results/logs/eval_lane.log 2>&1
    done
    for d in results/runs/$enc/*/; do
      n=$(basename $d)
      case $n in decoder_adapt_*|*fold*) continue;; esac
      [ -f $d/summary.json ] || continue
      [ -f results/eval/$enc/$n.json ] && continue
      free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
      [ "$free" -lt 15000 ] && continue
      echo "$(date +%H:%M) eval $enc/$n"; $E --encoder $enc --methods runs --runs $n >> results/logs/eval_lane.log 2>&1 || echo "FAILED eval $enc/$n"
    done
  done
  sleep 60
done
