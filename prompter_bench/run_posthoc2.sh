#!/bin/bash
# Post-hoc study for the prompters that finish later + the cross-family ensemble (after the CG-Net queue)
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENC=infused_mlp1
until [ -f results/runs/$ENC/cgnet_scratch_seg_s2/summary.json ] && [ -f results/runs/$ENC/mpg_fields_seg_s2/summary.json ] \
      && ! pgrep -f "^.venv/bin/python -u prompter_bench/posthoc.py" > /dev/null; do sleep 60; done
bash prompter_bench/wait_gpu.sh 16
.venv/bin/python -u prompter_bench/posthoc.py --encoder $ENC --methods cgnet_scratch_seg mpg_fields_seg 'mpg_seg+cgnet_scratch_seg' \
  > results/logs/posthoc_mlp1c.log 2>&1 || echo FAILED
echo "$(date +%H:%M) done"
