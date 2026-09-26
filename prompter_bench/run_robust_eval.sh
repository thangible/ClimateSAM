#!/bin/bash
# Test evaluation of the prompt-robust decoders (PLAN_20h B3): oracle study + real prompters with the replaced decoder
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENC=infused_mlp1
KINDS="bbox mask bbox+mask hybrid"
for dec in robust_decoder_hybrid_s0 robust_decoder_hybrid_s1 decoder_adapt_mpg_seg_s0_hybrid_oof_s0 decoder_adapt_mpg_seg_s0_bbox_oof_s0 \
           robust_decoder_box_s0 robust_decoder_box_s1 robust_decoder_hybrid_gtonly_s0 robust_decoder_hybrid_gtonly_s0_last; do
  run=$dec; [ -d results/runs/$ENC/$dec ] || run=${dec%_last}
  while [ ! -f results/runs/$ENC/$run/summary.json ] || { [ "$run" != "$dec" ] && [ ! -f results/runs/$ENC/$run/last_decoder.pth ]; }; do
    pgrep -f "^bash prompter_bench/run_robust2?\.sh" > /dev/null || { echo "robust queue gone, $dec missing"; continue 2; }
    sleep 60
  done
  echo "$(date +%H:%M) eval $dec"
  if [ ! -f "results/01_oracle_prompts/oracle_sweep_${ENC}@${dec}.csv" ]; then
    bash prompter_bench/wait_gpu.sh 12
    .venv/bin/python -u prompter_bench/oracle_sweep.py --encoder $ENC --decoder $dec > results/logs/robust_eval_oracle_${dec}.log 2>&1 || echo "FAILED oracle $dec"
  fi
  out=results/eval/$ENC/decoder/$dec
  [ -f $out/cgnet_finetuned.json ] || bash prompter_bench/wait_gpu.sh 12
  [ -f $out/cgnet_finetuned.json ] || .venv/bin/python -u prompter_bench/evaluate.py --encoder $ENC --decoder $dec --kinds $KINDS \
      --methods oracle cgnet_finetuned > results/logs/robust_eval_${dec}_a.log 2>&1 || echo "FAILED a $dec"
  for pat in 'mpg_seg_s[0-9]' 'msf_seg_s0' 'logreg_last_seg_s0' 'cgnet_scratch_seg_s0' 'msf_sp_token_seg_s0'; do
    ls results/runs/$ENC/$pat/summary.json > /dev/null 2>&1 || continue  # finished runs only
    n=$(ls results/runs/$ENC/$pat/summary.json | wc -l); have=$(ls $out/${pat}.json 2>/dev/null | wc -l)
    [ "$have" -ge "$n" ] && continue
    bash prompter_bench/wait_gpu.sh 12
    .venv/bin/python -u prompter_bench/evaluate.py --encoder $ENC --decoder $dec --kinds $KINDS --methods runs --runs "$pat" \
      > results/logs/robust_eval_${dec}_runs.log 2>&1 || echo "FAILED runs $pat $dec"
  done
  echo "$(date +%H:%M) done $dec"
done
