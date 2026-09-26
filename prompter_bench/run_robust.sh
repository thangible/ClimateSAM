#!/bin/bash
# Prompt-robust decoder runs (PLAN_20h B3), after the light queue of run_queue2.sh
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENC=infused_mlp1
run() {  # run <name> <args>
  name=$1; shift
  if [ -f results/runs/$ENC/$name/summary.json ]; then echo "skip $name"; return; fi
  echo "$(date +%H:%M) start $ENC/$name"
  .venv/bin/python -u prompter_bench/train_robust_decoder.py --encoder $ENC --wandb "$@" > results/logs/robust_${name}.log 2>&1 || echo "FAILED $ENC/$name"
  echo "$(date +%H:%M) done $ENC/$name"
}
run robust_decoder_hybrid_s0 --mode hybrid --seed 0
run robust_decoder_box_s0 --mode box --seed 0
run robust_decoder_hybrid_gtonly_s0 --mode hybrid --seed 0 --no_real
run robust_decoder_hybrid_s1 --mode hybrid --seed 1
run robust_decoder_box_s1 --mode box --seed 1
