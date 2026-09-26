#!/bin/bash
# Re-planned lanes (PLAN_20h): waits for a running python pid, then runs the rest. Usage: run_queue3.sh <heavy|light> <pid>
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENC=infused_mlp1
while kill -0 $2 2>/dev/null; do sleep 60; done
run() {
  name=$1; shift
  if [ -f results/runs/$ENC/$name/summary.json ]; then echo "skip $name"; return; fi
  echo "$(date +%H:%M) start $ENC/$name"
  .venv/bin/python -u prompter_bench/train.py --encoder $ENC --name $name --wandb "$@" > results/logs/train_${ENC}_${name}.log 2>&1 || echo "FAILED $ENC/$name"
  echo "$(date +%H:%M) done $ENC/$name"
}
if [ "$1" = heavy ]; then
  for s in 0 1 2; do run msf_sp_token_seg_s$s --arch msf_sp_token --seed $s --micro_bs 2; done
  run msf_token_cg_seg_s1 --arch msf_token_cg --seed 1 --micro_bs 2
else
  for s in 0 1 2; do run cgnet_scratch_seg_s$s --arch cgnet_scratch --seed $s --lr 1e-3 --micro_bs 4; done
  run cgnet_train_seg_s1 --arch cgnet_train --seed 1 --lr 1e-4 --micro_bs 4
fi
