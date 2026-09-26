#!/bin/bash
# second seed of the SAM-feature box head, after seed 0 (2 seeds: the head is far below the other prompters)
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while [ ! -f results/runs/infused_mlp1/det_head_s0/summary.json ]; do sleep 60; done
[ -f results/runs/infused_mlp1/det_head_s1/summary.json ] && exit 0
bash prompter_bench/wait_gpu.sh 10
echo "$(date +%H:%M) start det_head_s1"
.venv/bin/python -u prompter_bench/train_det_head_bench.py --seed 1 > results/logs/det_head_s1.log 2>&1 || echo "FAILED det_head_s1"
echo "$(date +%H:%M) done det_head_s1"
