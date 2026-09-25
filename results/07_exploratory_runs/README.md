# 07 — Exploratory runs (before the benchmark)

These were the first experiments with the mask-prompt generator. They used a **different frozen checkpoint**
(`best_weights/infused_token_vit_b_0.5_retrain_infused_05_00-005`, the one used by the last runs of
`train_generator.py`), a Tversky + BCE loss, and selected the best epoch **on the test set**. The numbers are therefore
optimistic and not comparable 1:1 with the benchmark; they are kept because they motivated its design, and because
some of them are negative results worth reporting.

Files: `logs/` (full training logs), `images/<run>/` (validation images every 25 / 10 epochs; `gen` = generator mask,
`sam` = SAM output), `scripts/` (the one-off analysis scripts), `../tables/exploratory_runs.*`,
`../figures/exploratory_runs.*`.

## Baseline being improved on

`train_generator.py` (multi-scale fusion, 1.41 M parameters, box prompts from connected components) on the same
checkpoint: SAM TC 0.311 / AR 0.385, generator mask TC 0.313 / AR 0.398 (wandb run `generator_128_vit_b_bbox`).
With ground-truth prompts the same SAM reaches ~0.6–0.7 TC / ~0.55–0.6 AR, so the prompts looked like the bottleneck.

## Design 1 — end-to-end mask-prompt generator (`run_v1`)

Idea: make the prompt differentiable, so the generator can be trained on SAM's *output* instead of on a proxy
segmentation loss followed by non-differentiable connected-component prompt extraction.

- Small head on the frozen features (neck embedding + 2 ViT blocks, all at 64×64, 671 k parameters) that outputs one
  256×256 logit map per class (the resolution of SAM's dense mask prompt).
- The map goes through the frozen prompt encoder (mask downscaling convolutions) into the frozen HQ decoder; loss on
  SAM's output + 0.5 × loss on the generator map.

Two problems found on the way (oracle tests with ground-truth masks, `oracle_first_encoder.csv`):

1. **{0, 1} mask prompts do nothing on this checkpoint** (TC 0.03 / AR 0.01 with *perfect* masks). The decoder reads
   dense prompts as logits, like SAM's own low-resolution masks: ±5 … ±20 logit maps give AR 0.74–0.85.
   The benchmark checkpoint behaves differently (see `00_setup`), i.e. this is checkpoint-dependent.
2. **Tropical cyclones disappear in mask-only prompts** (TC ≤ 0.28 even with perfect logits): the prompt encoder
   downsamples 256 → 64, and a cyclone covers only a few cells. A box fixes it (box 0.77, box + mask 0.85).

→ Final prompt design ("hybrid"): AR = logit map as dense prompt; TC = one box per connected component + the TC logit
map as dense prompt (the box is not differentiable, the dense part is).

Result: SAM TC 0.280 / AR 0.391 (best epoch 65), i.e. **no better than the old pipeline**, and the generator's own
mask was clearly worse (TC 0.234 / AR 0.339) — the SAM loss fought the generator's own objective early in training
(TC loss stayed flat for ~10 epochs). A first attempt crashed with an out-of-memory error: a noisy early TC map
produced hundreds of connected components → hundreds of decoder passes; capped at the 16 largest blobs
(ground truth has ≤ 13 TCs per image).

## Design 2 — more features (`run_ml4`)

Blocks 3, 6, 9, 12 instead of 6, 12: SAM 0.288 / 0.379, generator 0.187 / 0.304. No gain; dropped.

## Design 3 — segmentation loss only (`run_segonly`)

Same 671 k head trained only on its own maps: **generator TC 0.330 / AR 0.410 after 10 epochs**, better than the
1.41 M-parameter multi-scale generator (0.313 / 0.398) with half of its parameters and a small fraction of its compute
(the old model runs 3×3 convolutions with up to 512 channels at 1024×1024).
SAM prompted by these (un-calibrated) logits was poor for AR (0.265): nothing taught the logits to be good prompts.

## Design 4 — two-stage (`run_finetune`)

Design 3, then 30 epochs through SAM at lr 1e-4: SAM 0.311 / 0.389, generator 0.303 / 0.390 — SAM catches up with the
old pipeline, but still does not beat the generator mask it was prompted with.

## Why SAM could not add anything (`old_generator_analysis.csv`)

1. The old generator's own logits fed to SAM as mask prompts: 0.295 / 0.370 — *below* its own mask (0.304 / 0.401).
2. Error decomposition of the old generator's mask:
   - TC: 65 % of the cyclones found, 52 % of the predicted blobs false. Removing false blobs 0.30 → 0.38, adding the
     missed ones → 0.45, both → 0.55, exact shapes for the detected ones → 0.61.
   - AR: 92 % found, 58 % of the predicted blobs false (mostly small fragments). Perfect detection only reaches 0.48,
     but exact shapes of the detected rivers reach **0.82**.
   So for ARs almost all the lost IoU is the *extent* of rivers that are already found; SAM only reproduces such
   shapes when the prompt already contains them (GT box / GT mask).

These observations led to the controlled benchmark in `04_sam_feature_prompters` (fixed validation split, three
seeds, the thesis loss, every prompt type) and to the decoder-adaptation experiment in `05_decoder_adaptation`.
