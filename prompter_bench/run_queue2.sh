#!/bin/bash
# Second round (PLAN_20h.md). Usage: bash prompter_bench/run_queue2.sh <heavy|light>
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENC=infused_mlp1
run() {
  name=$1; shift
  if [ -f results/runs/$ENC/$name/summary.json ]; then echo "skip $name"; return; fi
  echo "$(date +%H:%M) start $ENC/$name"
  .venv/bin/python -u prompter_bench/train.py --encoder $ENC --name $name "$@" > results/logs/train_${ENC}_${name}.log 2>&1 || echo "FAILED $ENC/$name"
  echo "$(date +%H:%M) done $ENC/$name"
}
if [ "$1" = heavy ]; then
  for s in 0 1 2; do run msf_token_cg_seg_s$s --arch msf_token_cg --seed $s --micro_bs 2; done
  for s in 0 1 2; do run msf_sp_token_seg_s$s --arch msf_sp_token --seed $s --micro_bs 2; done
else
  for s in 0 1 2; do run cgnet_train_seg_s$s --arch cgnet_train --seed $s --lr 1e-4 --micro_bs 4; done
  for s in 0 1 2; do run cgnet_scratch_seg_s$s --arch cgnet_scratch --seed $s --lr 1e-3 --micro_bs 4; done
fi
