#!/bin/bash
# A2 now (in parallel to msf_token_cg seed 0), then msf_token_cg seed 1 once seed 0 has finished
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENC=infused_mlp1
run() {
  name=$1; shift
  if [ -f results/runs/$ENC/$name/summary.json ]; then echo "skip $name"; return; fi
  bash prompter_bench/wait_gpu.sh $GPU_GB
  echo "$(date +%H:%M) start $ENC/$name"
  .venv/bin/python -u prompter_bench/train.py --encoder $ENC --name $name --wandb "$@" > results/logs/train_${ENC}_${name}.log 2>&1 || echo "FAILED $ENC/$name"
  echo "$(date +%H:%M) done $ENC/$name"
}
GPU_GB=24
for s in 0 1 2; do run msf_sp_token_seg_s$s --arch msf_sp_token --seed $s --micro_bs 2; done
while [ ! -f results/runs/$ENC/msf_token_cg_seg_s0/summary.json ]; do sleep 60; done
GPU_GB=36
run msf_token_cg_seg_s1 --arch msf_token_cg --seed 1 --micro_bs 2
