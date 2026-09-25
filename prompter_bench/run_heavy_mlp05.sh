#!/bin/bash
# Second encoder, heavy prompters: one seed each (the primary encoder has three). Waits for the primary heavy lane.
cd "$(dirname "$0")/.."
while kill -0 1101947 2>/dev/null; do sleep 60; done
ENC=infused_mlp05
run() {
  name=$1; shift
  if [ -f results/runs/$ENC/$name/summary.json ]; then echo "skip $name"; return; fi
  echo "$(date +%H:%M) start $ENC/$name"
  .venv/bin/python -u prompter_bench/train.py --encoder $ENC --name $name "$@" > results/logs/train_${ENC}_${name}.log 2>&1 || echo "FAILED $ENC/$name"
  echo "$(date +%H:%M) done $ENC/$name"
}
run msf_seg_s0 --arch msf --seed 0 --micro_bs 2
run msf_token_seg_s0 --arch msf_token --seed 0 --micro_bs 2
