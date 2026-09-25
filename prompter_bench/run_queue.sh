#!/bin/bash
# Two lanes of training jobs on one GPU. Usage: bash prompter_bench/run_queue.sh <encoder> <lane: heavy|light>
cd "$(dirname "$0")/.."
ENC=$1; LANE=$2
PY=".venv/bin/python -u prompter_bench/train.py --encoder $ENC"
run() {  # run <name> <args...>
  name=$1; shift
  if [ -f results/runs/$ENC/$name/summary.json ]; then echo "skip $name"; return; fi
  echo "$(date +%H:%M) start $ENC/$name"
  $PY --name $name "$@" > results/logs/train_${ENC}_${name}.log 2>&1 || echo "FAILED $ENC/$name"
  echo "$(date +%H:%M) done $ENC/$name"
}
if [ "$LANE" = heavy ]; then
  for s in 0 1 2; do run msf_seg_s$s --arch msf --seed $s --micro_bs 2; done
  for s in 0 1 2; do run msf_token_seg_s$s --arch msf_token --seed $s --micro_bs 2; done
else
  for s in 0 1 2; do run logreg_l0_seg_s$s --arch logreg_l0 --seed $s; done
  for s in 0 1 2; do run logreg_last_seg_s$s --arch logreg_last --seed $s; done
  for s in 0 1 2; do run mpg_seg_s$s --arch mpg --seed $s; done
  for s in 0 1 2; do run mpg_sam_e2e_s$s --arch mpg --mode sam --seed $s --aux_weight 0.5; done
  for s in 0 1 2; do run mpg_twostage_s$s --arch mpg --mode sam --seed $s --init results/runs/$ENC/mpg_seg_s$s/best.pth --lr 1e-4 --epochs 30; done
fi
