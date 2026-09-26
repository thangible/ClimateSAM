#!/bin/bash
# MPG + raw CG-Net fields (3 seeds, primary checkpoint)
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENC=infused_mlp1
for s in 0 1 2; do
  name=mpg_fields_seg_s$s
  [ -f results/runs/$ENC/$name/summary.json ] && continue
  bash prompter_bench/wait_gpu.sh 10
  echo "$(date +%H:%M) start $name"
  .venv/bin/python -u prompter_bench/train.py --encoder $ENC --arch mpg_fields --seed $s --name $name --wandb > results/logs/train_${ENC}_${name}.log 2>&1 || echo "FAILED $name"
  echo "$(date +%H:%M) done $name"
done
