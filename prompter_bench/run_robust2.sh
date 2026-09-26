#!/bin/bash
# Box-mode prompt-robust decoders (re-run after OOM; at most 12 training boxes per class and image)
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENC=infused_mlp1
for args in "robust_decoder_box_s0 --seed 0" "robust_decoder_box_s1 --seed 1"; do
  set -- $args; name=$1; shift
  [ -f results/runs/$ENC/$name/summary.json ] && continue
  bash prompter_bench/wait_gpu.sh 26
  echo "$(date +%H:%M) start $name"
  .venv/bin/python -u prompter_bench/train_robust_decoder.py --encoder $ENC --mode box "$@" > results/logs/robust_${name}.log 2>&1 || echo "FAILED $name"
  echo "$(date +%H:%M) done $name"
done
# control without real prompts, re-run to also keep its last-epoch decoder
name=robust_decoder_hybrid_gtonly_s0
if [ ! -f results/runs/$ENC/$name/summary.json ]; then
  bash prompter_bench/wait_gpu.sh 16
  echo "$(date +%H:%M) start $name"
  .venv/bin/python -u prompter_bench/train_robust_decoder.py --encoder $ENC --mode hybrid --seed 0 --no_real > results/logs/robust_${name}.log 2>&1 || echo "FAILED $name"
  echo "$(date +%H:%M) done $name"
fi
