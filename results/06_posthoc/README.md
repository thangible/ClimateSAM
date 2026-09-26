# 06 — Evaluation-only improvements of the prompter masks

Script: `prompter_bench/posthoc.py` (`run_posthoc2.sh`) · data: `posthoc_<encoder>_<method>.json` (all variants,
selected thresholds / post-processing, paired bootstrap against the seed-0 mask, per-image counts) · table:
`../tables/posthoc_infused_mlp1.*`.

Every free parameter is chosen on the 40 validation images; the test set is only used for the final numbers.

| Variant | What it does |
|---|---|
| single seed | the benchmark prompter mask (logit > 0), mean over seeds |
| ensemble | mean of the logits of all seeds (`a+b`: all seeds of both prompters) |
| calibrated | per-class logit threshold from {−4, −3.75, …, +4} maximising the validation IoU of that class |
| post-processed | + minimum blob size per class {0 … 1600 px} and, for TC, maximum absolute latitude of the blob centroid {90, 45, 40, 35}° |
| SAM hybrid, round k | the post-processed mask as hybrid prompt (TC: boxes + logits, AR: logits); then SAM's own output as the next prompt |

## Results (mean FG IoU, test)

| Prompter | single seed | ensemble | + calibrated | + post-processed | SAM round 1 / 2 / 3 |
|---|---|---|---|---|---|
| MPG | 0.379 | **0.385** (+0.006 [+0.001, +0.010] vs. seed 0) | 0.383 | 0.384 | 0.369 / 0.366 / 0.359 |
| MSF | 0.368 | 0.376 | 0.375 | 0.375 | 0.361 / 0.359 / 0.355 |
| MSF + token gate | 0.362 | 0.368 | 0.367 | 0.362 | 0.354 / 0.351 / 0.345 |
| LogReg, block 12 | 0.343 | 0.344 | 0.344 | 0.341 | 0.330 / 0.326 / 0.322 |
| MPG end-to-end | 0.356 | 0.371 | 0.369 | 0.369 | 0.367 / 0.363 / 0.358 |
| CG-Net (from scratch) | 0.369 | 0.380 | 0.382 | 0.381 | 0.368 / 0.364 / 0.359 |
| MPG + raw fields | 0.375 | 0.382 | 0.380 | 0.382 | 0.369 / 0.365 / 0.358 |
| **MPG + CG-Net (3 + 3 models)** | — | **0.389** (TC 0.363 / AR 0.416) | 0.391 | 0.388 | 0.370 / 0.367 / 0.362 |

- Seed ensembles: +0.006 … +0.011 (end-to-end MPG +0.015); the linear probe's seeds are nearly identical.
- Calibration / post-processing: the selected TC thresholds scatter between −1.0 and +2.75 but change the test IoU by
  only −0.005 … +0.002 — 40 validation images are too few to tune them.
- Iterative SAM refinement loses in every round.
- MPG + CG-Net: best mask of the study; vs. the MPG ensemble +0.004 [−0.005, +0.012] (TC +0.010 [−0.005, +0.024]),
  vs. the CG-Net ensemble +0.010 [+0.004, +0.016]. Six models against three, so the gain over the MPG ensemble is not
  conclusive; the TC gain suggests the two input types make partly different TC errors.
