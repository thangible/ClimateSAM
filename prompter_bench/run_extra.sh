#!/bin/bash
# Follow-up experiments: out-of-fold prompters, label smoothing, decoder adaptation. Usage: bash run_extra.sh <encoder>
cd "$(dirname "$0")/.."
ENC=$1
T=".venv/bin/python -u prompter_bench/train.py --encoder $ENC"
D=".venv/bin/python -u prompter_bench/train_decoder.py --encoder $ENC"
run() {  # run <run name> <command...>
  name=$1; shift
  if [ -f results/runs/$ENC/$name/summary.json ]; then echo "skip $name"; return; fi
  echo "$(date +%H:%M) start $ENC/$name"
  "$@" > results/logs/extra_${ENC}_${name}.log 2>&1 || echo "FAILED $ENC/$name"
  echo "$(date +%H:%M) done $ENC/$name"
}
until [ -f results/runs/$ENC/mpg_seg_s0/summary.json ]; do sleep 30; done
for k in 0 1 2 3 4; do run mpg_seg_fold$k $T --arch mpg --seed 0 --exclude_fold $k --name mpg_seg_fold$k; done
for s in 0 1 2; do run mpg_seg_smooth_s$s $T --arch mpg --seed $s --smooth --name mpg_seg_smooth_s$s; done
run decoder_adapt_mpg_seg_s0_bbox_s0 $D --prompter mpg_seg_s0 --kind bbox
run decoder_adapt_mpg_seg_s0_bbox_oof_s0 $D --prompter mpg_seg_s0 --kind bbox --oof_prefix mpg_seg_fold
run decoder_adapt_mpg_seg_s0_hybrid_oof_s0 $D --prompter mpg_seg_s0 --kind hybrid --oof_prefix mpg_seg_fold
