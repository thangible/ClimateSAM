<!-- generated from README.template.md by prompter_bench/make_report.py -->
# Section 4.3 — Automatic prompting of ClimateSAM: all experiments, data and conclusions

Everything needed for Section 4.3 (Automatic Prompt Generator for SAM) and the corresponding part of the Discussion:
the corrected Table 4.13, a controlled comparison of all prompters (the thesis prompters + a new mask-prompt
generator), oracle studies, significance tests, decoder adaptation, negative results, figures, LaTeX tables and all raw
data. Tables below are generated from the raw outputs (`prompter_bench/make_report.py`); LaTeX versions are in
`tables/*.tex` (need `\usepackage{booktabs}`), CSV versions in `tables/*.csv`.

## Summary

1. **The SAM rows of the original Table 4.13 do not reproduce.** With the thesis pipeline itself, SAM prompted by
   CG-Net reaches TC 0.341 / AR 0.369 (tight boxes), *below* CG-Net alone (0.349 / 0.382), not 0.430 / 0.482. The
   ranking of prompt types in the table is confirmed. → `02_table_4_13_recheck/`
2. **With automatically generated prompts, SAM does not improve on the prompter that produced them** — for every
   prompter (CG-Net, logistic regression, multi-scale fusion, token-gated fusion, mask-prompt generator), every prompt
   type (boxes, points, boxes + points, dense masks, hybrid) and both frozen checkpoints. SAM's output stays within
   about ±0.02 mean FG IoU of its prompter's own mask (gains up to +0.012 / +0.019 only for the prompter with the
   weakest mask, the end-to-end generator; points always lose 0.03–0.08), and the best SAM output overall equals the
   best prompter mask (primary checkpoint 0.376 vs. 0.379; second checkpoint 0.373 vs. 0.371, within noise). Paired
   bootstrap intervals: `tables/bootstrap_*`, `figures/sam_minus_prompter.png`. → `04_sam_feature_prompters/`
3. **Why:** the Phase-1 decoder was trained with perfect prompts and *reproduces* its prompts. Oracle experiments
   show every false box becomes a false object, every missed object stays missed and ±10–20 % box errors cost
   0.1–0.3 IoU (`01_oracle_prompts/`). The prompters' remaining errors are exactly those: for ARs almost all lost IoU
   is the extent/shape of rivers that were found; for TCs a third of the cyclones is missed and half the predicted
   blobs are false (`tables/error_decomposition_*`).
4. **Fine-tuning the decoder on generated prompts** (incl. out-of-fold prompts, so the decoder sees realistic prompt
   errors) teaches it to drop some false prompts (AR object precision 0.50 → 0.62) and closes most of the gap to the
   prompter, but still does not exceed it. → `05_decoder_adaptation/`
5. **The best automatic system is a small prompter on the frozen ClimateSAM features.** The new 0.67 M-parameter
   mask-prompt generator reaches TC 0.344 / AR 0.413 on its own (3 seeds): AR +0.031 over the fine-tuned CG-Net
   (95 % CI [+0.020, +0.043]) at statistically equal TC, and +0.011 mean FG IoU over the thesis' 1.41 M-parameter
   multi-scale fusion generator (CI [+0.003, +0.019]) with ~120× fewer FLOPs (5.7 vs. 702 GFLOPs per image,
   0.9 vs. 36 ms). SAM prompted by it reaches
   TC 0.342 / AR 0.398 (hybrid prompts). A 1.5 k-parameter linear probe on the last ViT block already reaches
   TC 0.301 / AR 0.385 — the adapted encoder's features carry most of the signal. On the second frozen checkpoint
   the mask-prompt generator ties multi-scale fusion (mean FG IoU 0.370 vs. 0.370, one MSF seed); both stay ahead
   of CG-Net on AR and of the linear probe.
6. **Prompt format matters, and in a checkpoint-dependent way.** Dense mask prompts must be logits for some
   checkpoints, and TCs vanish from mask-only prompts on others; a box plus a dense mask for TC and a dense mask for
   AR ("hybrid") is the robust choice (oracle: TC 0.872 / AR 0.920 vs. boxes 0.723 / 0.625).
7. **Also negative:** a YOLO box head on CG-Net (TC ≈ 0.21 / AR ≈ 0.28 via SAM); learned static prompts without any
   prompter (TC 0.24 / AR 0.33, thousands of fragments); the token gate on multi-scale fusion (−0.006 mean FG IoU);
   training the generator end-to-end through SAM (−0.011) or two-stage (−0.004); label smoothing for the prompter
   (−0.004); more ViT blocks as input (exploratory).

## Folder guide

| Folder | Content |
|---|---|
| `00_setup/` | protocol shared by every experiment (checkpoint choice, split, loss, metrics), `split.json` |
| `01_oracle_prompts/` | ground-truth prompts: prompt types, mask-prompt format, controlled prompt errors |
| `02_table_4_13_recheck/` | corrected Table 4.13 (CG-Net as prompter), incl. the original-script path |
| `03_cgnet_yolo_boxes/` | CG-Net with a YOLO box head as prompter |
| `04_sam_feature_prompters/` | main comparison of all prompters (designs, bugs found, results) |
| `05_decoder_adaptation/` | fine-tuning the decoder on generated prompts (in-sample vs. out-of-fold) |
| `07_exploratory_runs/` | first mask-prompt-generator runs incl. failed designs, analysis of the old generator |
| `tables/` | every table as LaTeX (`.tex`), CSV and Markdown |
| `figures/` | every figure as PNG and PDF (diagrams, bar charts, curves, map examples) |
| `runs/<encoder>/<run>/` | per run: `log.csv` (every epoch), `summary.json`, `best.pth` |
| `eval/<encoder>/` | test results per run (`*.json`, incl. per-image counts), example masks (`*_examples.npz`), `all_results.csv` |
| `logs/` | stdout of every job |

Encoders: `infused_mlp1` = primary (Infused Token, MLP 1.0, the model of Table 4.12 / Section 4.3),
`infused_mlp05` = robustness check (Infused Token, MLP 0.5).

## 1. Oracle prompts (upper bounds, `01_oracle_prompts/`)

| Prompt from the ground truth | TC | AR |
|---|---|---|
| tight box | 0.723 | 0.625 |
| points (5 pos. + centroid + 5 neg.) | 0.527 | 0.577 |
| dense mask (±10 logits) | 0.774 | 0.920 |
| hybrid (TC: box + mask, AR: mask) | 0.872 | 0.920 |
| box enlarged 20 % | 0.463 | 0.484 |
| 25 % of the objects not prompted | 0.548 | 0.483 |
| + 1 random false box per class | 0.574 | 0.578 |

![](figures/oracle_prompt_errors.png)

## 2. Corrected Table 4.13 (`02_table_4_13_recheck/`)

| Prompt type | Pos. | Neg. | Enlarge | IoU AR | IoU TC | IoU AR (orig. script) | IoU TC (orig. script) |
|---|---|---|---|---|---|---|---|
| CG-Net alone (baseline) |  |  |  | 0.3817 | 0.3494 |  |  |
| bbox | 0 | 0 | 0.0 | 0.3690 | 0.3414 | 0.3630 | 0.3390 |
| mask | 0 | 0 | 0.0 | 0.3758 | 0.3343 | 0.3568 | 0.3251 |
| bbox | 0 | 0 | -0.1 | 0.3267 | 0.3327 | 0.3294 | 0.3315 |
| point+bbox | 15 | 10 | 0.0 | 0.3305 | 0.3252 | 0.3115 | 0.3232 |
| point+bbox | 10 | 5 | 0.0 | 0.3427 | 0.3235 | 0.3254 | 0.3181 |
| point | 5 | 10 | 0.0 | 0.3373 | 0.3178 | 0.3236 | 0.3139 |
| point | 1 | 3 | 0.0 | 0.2713 | 0.3171 | 0.2774 | 0.3138 |
| point | 5 | 5 | 0.0 | 0.3375 | 0.3128 | 0.3259 | 0.3093 |
| bbox | 0 | 0 | 0.1 | 0.3582 | 0.3073 | 0.3436 | 0.3062 |
| point | 2 | 2 | 0.0 | 0.3180 | 0.3048 | 0.3148 | 0.2986 |
| point | 1 | 2 | 0.0 | 0.2866 | 0.3038 | 0.2878 | 0.3023 |
| point | 1 | 1 | 0.0 | 0.3086 | 0.2964 | 0.3049 | 0.2946 |
| point | 10 | 10 | 0.0 | 0.3262 | 0.2923 | 0.3031 | 0.2847 |
| bbox | 0 | 0 | 0.2 | 0.3334 | 0.2703 | 0.3110 | 0.2706 |
| bbox | 0 | 0 | -0.2 | 0.2288 | 0.2502 | 0.2357 | 0.2506 |
| point+bbox+mask | 20 | 10 | 0.0 | 0.3313 | 0.2396 | 0.3212 | 0.2184 |
| bbox | 0 | 0 | 0.3 | 0.2991 | 0.2358 | 0.2717 | 0.2376 |
| bbox | 0 | 0 | 0.5 | 0.2372 | 0.1822 | 0.2042 | 0.1837 |
| point+bbox+mask | 10 | 5 | 0.0 | 0.3471 | 0.1357 | 0.3390 | 0.1222 |


## 3. All prompters under one protocol (`04_sam_feature_prompters/`)

Frozen primary checkpoint, 358 train / 40 validation / 61 test images, loss of Table 4.4, 60 epochs, checkpoint
selected on validation, mean ± std over 3 seeds (CG-Net: fixed checkpoints).

| Prompter | Output | TC IoU | AR IoU | BG IoU | Mean IoU | Mean FG IoU |
|---|---|---|---|---|---|---|
| Ground truth (oracle) | Prompter mask (no SAM) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
|  | SAM + box | 0.723 | 0.625 | 0.971 | 0.773 | 0.674 |
|  | SAM + hybrid | 0.872 | 0.920 | 0.994 | 0.929 | 0.896 |
| CG-Net (official weights) | Prompter mask (no SAM) | 0.327 | 0.333 | 0.926 | 0.529 | 0.330 |
|  | SAM + box | 0.318 | 0.342 | 0.931 | 0.530 | 0.330 |
|  | SAM + hybrid | 0.327 | 0.339 | 0.929 | 0.532 | 0.333 |
| CG-Net (fine-tuned) | Prompter mask (no SAM) | 0.349 | 0.382 | 0.940 | 0.557 | 0.366 |
|  | SAM + box | 0.341 | 0.369 | 0.939 | 0.550 | 0.355 |
|  | SAM + hybrid | 0.349 | 0.384 | 0.941 | 0.558 | 0.367 |
| CG-Net (re-trained on 358 images, official init.) | Prompter mask (no SAM) | 0.350 ± 0.005 | 0.399 ± 0.005 | 0.945 ± 0.001 | 0.565 ± 0.003 | 0.375 ± 0.005 |
|  | SAM + box | 0.345 ± 0.003 | 0.387 ± 0.006 | 0.943 ± 0.001 | 0.558 ± 0.003 | 0.366 ± 0.004 |
|  | SAM + hybrid | 0.348 ± 0.007 | 0.400 ± 0.005 | 0.946 ± 0.001 | 0.564 ± 0.004 | 0.374 ± 0.006 |
| CG-Net (trained from scratch on 358 images) | Prompter mask (no SAM) | 0.343 ± 0.007 | 0.394 ± 0.008 | 0.946 ± 0.004 | 0.561 ± 0.003 | 0.369 ± 0.006 |
|  | SAM + box | 0.343 ± 0.006 | 0.380 ± 0.002 | 0.942 ± 0.004 | 0.555 ± 0.002 | 0.362 ± 0.003 |
|  | SAM + hybrid | 0.339 ± 0.008 | 0.395 ± 0.008 | 0.946 ± 0.003 | 0.560 ± 0.003 | 0.367 ± 0.006 |
| Logistic regression, ViT block 1 | Prompter mask (no SAM) | 0.195 ± 0.001 | 0.319 ± 0.001 | 0.927 ± 0.004 | 0.480 ± 0.002 | 0.257 ± 0.001 |
|  | SAM + box | 0.200 ± 0.002 | 0.321 ± 0.003 | 0.925 ± 0.004 | 0.482 ± 0.003 | 0.260 ± 0.002 |
|  | SAM + hybrid | 0.213 ± 0.000 | 0.316 ± 0.002 | 0.939 ± 0.001 | 0.489 ± 0.001 | 0.264 ± 0.001 |
| Logistic regression, ViT block 12 | Prompter mask (no SAM) | 0.301 ± 0.004 | 0.385 ± 0.001 | 0.938 ± 0.001 | 0.541 ± 0.002 | 0.343 ± 0.002 |
|  | SAM + box | 0.291 ± 0.001 | 0.362 ± 0.001 | 0.932 ± 0.001 | 0.528 ± 0.000 | 0.326 ± 0.000 |
|  | SAM + hybrid | 0.304 ± 0.004 | 0.373 ± 0.002 | 0.946 ± 0.000 | 0.541 ± 0.002 | 0.338 ± 0.002 |
| Multi-scale fusion | Prompter mask (no SAM) | 0.333 ± 0.002 | 0.403 ± 0.004 | 0.944 ± 0.003 | 0.560 ± 0.001 | 0.368 ± 0.003 |
|  | SAM + box | 0.339 ± 0.001 | 0.385 ± 0.002 | 0.941 ± 0.003 | 0.555 ± 0.001 | 0.362 ± 0.001 |
|  | SAM + hybrid | 0.330 ± 0.002 | 0.388 ± 0.010 | 0.949 ± 0.001 | 0.556 ± 0.003 | 0.359 ± 0.005 |
| Multi-scale fusion + token gate | Prompter mask (no SAM) | 0.324 ± 0.001 | 0.399 ± 0.003 | 0.944 ± 0.002 | 0.556 ± 0.001 | 0.362 ± 0.001 |
|  | SAM + box | 0.336 ± 0.002 | 0.382 ± 0.003 | 0.940 ± 0.002 | 0.553 ± 0.001 | 0.359 ± 0.002 |
|  | SAM + hybrid | 0.329 ± 0.004 | 0.387 ± 0.006 | 0.949 ± 0.001 | 0.555 ± 0.003 | 0.358 ± 0.005 |
| Mask-prompt generator (segmentation loss) | Prompter mask (no SAM) | 0.344 ± 0.003 | 0.413 ± 0.001 | 0.944 ± 0.000 | 0.567 ± 0.002 | 0.379 ± 0.002 |
|  | SAM + box | 0.343 ± 0.006 | 0.391 ± 0.001 | 0.940 ± 0.001 | 0.558 ± 0.002 | 0.367 ± 0.003 |
|  | SAM + hybrid | 0.342 ± 0.004 | 0.398 ± 0.002 | 0.949 ± 0.000 | 0.563 ± 0.002 | 0.370 ± 0.002 |
| Mask-prompt generator (end-to-end via SAM) | Prompter mask (no SAM) | 0.335 ± 0.010 | 0.378 ± 0.019 | 0.941 ± 0.006 | 0.551 ± 0.010 | 0.356 ± 0.014 |
|  | SAM + box | 0.336 ± 0.011 | 0.385 ± 0.005 | 0.940 ± 0.007 | 0.554 ± 0.003 | 0.360 ± 0.005 |
|  | SAM + hybrid | 0.334 ± 0.009 | 0.402 ± 0.007 | 0.946 ± 0.003 | 0.561 ± 0.004 | 0.368 ± 0.008 |
| Mask-prompt generator (two-stage) | Prompter mask (no SAM) | 0.338 ± 0.005 | 0.409 ± 0.002 | 0.938 ± 0.002 | 0.562 ± 0.000 | 0.374 ± 0.001 |
|  | SAM + box | 0.334 ± 0.003 | 0.386 ± 0.004 | 0.935 ± 0.002 | 0.552 ± 0.001 | 0.360 ± 0.001 |
|  | SAM + hybrid | 0.337 ± 0.007 | 0.411 ± 0.002 | 0.947 ± 0.001 | 0.565 ± 0.002 | 0.374 ± 0.004 |
| Mask-prompt generator (segmentation loss, label smoothing) | Prompter mask (no SAM) | 0.338 ± 0.009 | 0.412 ± 0.001 | 0.943 ± 0.001 | 0.564 ± 0.003 | 0.375 ± 0.005 |
|  | SAM + box | 0.340 ± 0.007 | 0.388 ± 0.003 | 0.939 ± 0.001 | 0.556 ± 0.002 | 0.364 ± 0.003 |
|  | SAM + hybrid | 0.335 ± 0.007 | 0.401 ± 0.002 | 0.949 ± 0.000 | 0.562 ± 0.003 | 0.368 ± 0.005 |
| Mask-prompt generator + raw CG-Net fields | Prompter mask (no SAM) | 0.340 ± 0.001 | 0.409 ± 0.006 | 0.942 ± 0.004 | 0.564 ± 0.004 | 0.375 ± 0.004 |
|  | SAM + box | 0.340 ± 0.001 | 0.387 ± 0.008 | 0.938 ± 0.004 | 0.555 ± 0.004 | 0.363 ± 0.004 |
|  | SAM + hybrid | 0.337 ± 0.003 | 0.399 ± 0.006 | 0.949 ± 0.001 | 0.561 ± 0.002 | 0.368 ± 0.003 |
| Box head on SAM features (YOLO-style) | Prompter mask (no SAM) | 0.084 | 0.168 | 0.718 | 0.323 | 0.126 |
|  | SAM + box | 0.095 | 0.321 | 0.887 | 0.434 | 0.208 |
| Learned static prompts (4 tokens / class) | SAM + learned static prompts | 0.237 ± 0.001 | 0.324 ± 0.008 | 0.926 ± 0.005 | 0.495 ± 0.004 | 0.280 ± 0.004 |
| Learned static prompts (16 tokens / class) | SAM + learned static prompts | 0.248 | 0.329 | 0.930 | 0.502 | 0.288 |


TC / AR IoU for every way of turning the prompter output into SAM prompts:

| Prompter | Prompter mask (no SAM) | SAM + box | SAM + box (+10%) | SAM + points | SAM + box + points | SAM + mask logits | SAM + box + mask | SAM + hybrid | mean(prompter, SAM hybrid) | SAM + learned static prompts |
|---|---|---|---|---|---|---|---|---|---|---|
| Ground truth (oracle) | 1.00 / 1.00 | 0.72 / 0.62 | 0.59 / 0.57 | 0.53 / 0.58 | 0.58 / 0.60 | 0.77 / 0.92 | 0.87 / 0.88 | 0.87 / 0.92 | 0.95 / 0.98 | -- |
| CG-Net (official weights) | 0.33 / 0.33 | 0.32 / 0.34 | 0.28 / 0.34 | 0.25 / 0.31 | 0.28 / 0.32 | 0.33 / 0.34 | 0.33 / 0.34 | 0.33 / 0.34 | -- | -- |
| CG-Net (fine-tuned) | 0.35 / 0.38 | 0.34 / 0.37 | 0.31 / 0.36 | 0.28 / 0.34 | 0.32 / 0.36 | 0.34 / 0.38 | 0.35 / 0.38 | 0.35 / 0.38 | -- | -- |
| CG-Net (re-trained on 358 images, official init.) | 0.35 / 0.40 | 0.35 / 0.39 | 0.33 / 0.38 | 0.29 / 0.36 | 0.32 / 0.37 | 0.34 / 0.40 | 0.35 / 0.40 | 0.35 / 0.40 | -- | -- |
| CG-Net (trained from scratch on 358 images) | 0.34 / 0.39 | 0.34 / 0.38 | 0.33 / 0.37 | 0.31 / 0.35 | 0.33 / 0.37 | 0.32 / 0.40 | 0.34 / 0.39 | 0.34 / 0.40 | -- | -- |
| Logistic regression, ViT block 1 | 0.20 / 0.32 | 0.20 / 0.32 | 0.19 / 0.32 | 0.14 / 0.27 | 0.18 / 0.30 | 0.04 / 0.32 | 0.20 / 0.31 | 0.21 / 0.32 | 0.21 / 0.32 | -- |
| Logistic regression, ViT block 12 | 0.30 / 0.39 | 0.29 / 0.36 | 0.28 / 0.35 | 0.25 / 0.31 | 0.28 / 0.35 | 0.04 / 0.37 | 0.29 / 0.36 | 0.30 / 0.37 | 0.30 / 0.37 | -- |
| Multi-scale fusion | 0.33 / 0.40 | 0.34 / 0.38 | 0.33 / 0.38 | 0.30 / 0.35 | 0.33 / 0.37 | 0.31 / 0.39 | 0.33 / 0.38 | 0.33 / 0.39 | 0.33 / 0.39 | -- |
| Multi-scale fusion + token gate | 0.32 / 0.40 | 0.34 / 0.38 | 0.32 / 0.38 | 0.28 / 0.34 | 0.32 / 0.37 | 0.26 / 0.39 | 0.33 / 0.38 | 0.33 / 0.39 | 0.33 / 0.39 | -- |
| Mask-prompt generator (segmentation loss) | 0.34 / 0.41 | 0.34 / 0.39 | 0.32 / 0.38 | 0.30 / 0.35 | 0.32 / 0.37 | 0.30 / 0.40 | 0.34 / 0.39 | 0.34 / 0.40 | 0.34 / 0.40 | -- |
| Mask-prompt generator (end-to-end via SAM) | 0.33 / 0.38 | 0.34 / 0.38 | 0.32 / 0.38 | 0.29 / 0.31 | 0.32 / 0.36 | 0.30 / 0.40 | 0.33 / 0.38 | 0.33 / 0.40 | 0.33 / 0.40 | -- |
| Mask-prompt generator (two-stage) | 0.34 / 0.41 | 0.33 / 0.39 | 0.31 / 0.36 | 0.29 / 0.33 | 0.32 / 0.36 | 0.29 / 0.41 | 0.33 / 0.40 | 0.34 / 0.41 | 0.34 / 0.41 | -- |
| Mask-prompt generator (segmentation loss, label smoothing) | 0.34 / 0.41 | 0.34 / 0.39 | 0.32 / 0.37 | 0.31 / 0.35 | 0.33 / 0.37 | 0.29 / 0.40 | 0.33 / 0.40 | 0.33 / 0.40 | 0.34 / 0.40 | -- |
| Mask-prompt generator + raw CG-Net fields | 0.34 / 0.41 | 0.34 / 0.39 | 0.32 / 0.37 | 0.30 / 0.35 | 0.32 / 0.37 | 0.30 / 0.40 | 0.34 / 0.39 | 0.34 / 0.40 | 0.34 / 0.40 | -- |
| Box head on SAM features (YOLO-style) | 0.08 / 0.17 | 0.10 / 0.32 | -- | -- | -- | -- | -- | -- | -- | -- |
| Learned static prompts (4 tokens / class) | -- | -- | -- | -- | -- | -- | -- | -- | -- | 0.24 / 0.32 |
| Learned static prompts (16 tokens / class) | -- | -- | -- | -- | -- | -- | -- | -- | -- | 0.25 / 0.33 |


Paired bootstrap over the test images (seeds pooled):

| Comparison (B minus A) | Δ TC IoU [95% CI] | Δ AR IoU [95% CI] | Δ mean FG IoU [95% CI] | P(ΔFG>0) |
|---|---|---|---|---|
| CG-Net (official weights): SAM + box vs. prompter mask | -0.009 [-0.017, -0.002] | +0.009 [+0.004, +0.014] | -0.000 [-0.005, +0.004] | 0.45 |
| CG-Net (official weights): SAM + hybrid vs. prompter mask | +0.000 [-0.003, +0.003] | +0.006 [+0.005, +0.007] | +0.003 [+0.002, +0.005] | 1.00 |
| CG-Net (fine-tuned): SAM + box vs. prompter mask | -0.008 [-0.017, +0.001] | -0.013 [-0.018, -0.006] | -0.010 [-0.016, -0.004] | 0.00 |
| CG-Net (fine-tuned): SAM + hybrid vs. prompter mask | -0.001 [-0.004, +0.003] | +0.003 [+0.002, +0.004] | +0.001 [-0.001, +0.003] | 0.89 |
| CG-Net (re-trained on 358 images, official init.): SAM + box vs. prompter mask | -0.005 [-0.014, +0.004] | -0.013 [-0.019, -0.007] | -0.009 [-0.014, -0.004] | 0.00 |
| CG-Net (re-trained on 358 images, official init.): SAM + hybrid vs. prompter mask | -0.002 [-0.005, +0.000] | +0.001 [-0.000, +0.001] | -0.001 [-0.002, +0.000] | 0.09 |
| CG-Net (trained from scratch on 358 images): SAM + box vs. prompter mask | +0.000 [-0.008, +0.007] | -0.015 [-0.021, -0.009] | -0.007 [-0.012, -0.002] | 0.00 |
| CG-Net (trained from scratch on 358 images): SAM + hybrid vs. prompter mask | -0.004 [-0.006, -0.001] | +0.001 [+0.000, +0.002] | -0.001 [-0.002, -0.000] | 0.02 |
| Logistic regression, ViT block 1: SAM + box vs. prompter mask | +0.005 [-0.003, +0.012] | +0.001 [-0.003, +0.006] | +0.003 [-0.002, +0.007] | 0.90 |
| Logistic regression, ViT block 1: SAM + hybrid vs. prompter mask | +0.017 [+0.008, +0.026] | -0.003 [-0.014, +0.007] | +0.007 [-0.000, +0.013] | 0.97 |
| Logistic regression, ViT block 1: mean(prompter, SAM hybrid) vs. prompter mask | +0.017 [+0.009, +0.025] | -0.003 [-0.013, +0.007] | +0.007 [+0.001, +0.013] | 0.98 |
| Logistic regression, ViT block 12: SAM + box vs. prompter mask | -0.010 [-0.029, +0.009] | -0.024 [-0.029, -0.019] | -0.017 [-0.027, -0.006] | 0.00 |
| Logistic regression, ViT block 12: SAM + hybrid vs. prompter mask | +0.003 [-0.005, +0.011] | -0.012 [-0.024, -0.001] | -0.005 [-0.012, +0.003] | 0.11 |
| Logistic regression, ViT block 12: mean(prompter, SAM hybrid) vs. prompter mask | +0.004 [-0.004, +0.011] | -0.011 [-0.023, +0.000] | -0.004 [-0.011, +0.003] | 0.15 |
| Multi-scale fusion: SAM + box vs. prompter mask | +0.006 [-0.004, +0.015] | -0.018 [-0.023, -0.013] | -0.006 [-0.012, -0.000] | 0.01 |
| Multi-scale fusion: SAM + hybrid vs. prompter mask | -0.003 [-0.007, +0.001] | -0.014 [-0.023, -0.005] | -0.009 [-0.013, -0.004] | 0.00 |
| Multi-scale fusion: mean(prompter, SAM hybrid) vs. prompter mask | +0.000 [-0.003, +0.003] | -0.012 [-0.020, -0.004] | -0.006 [-0.010, -0.002] | 0.00 |
| Multi-scale fusion + token gate: SAM + box vs. prompter mask | +0.011 [+0.000, +0.022] | -0.017 [-0.022, -0.011] | -0.003 [-0.009, +0.004] | 0.22 |
| Multi-scale fusion + token gate: SAM + hybrid vs. prompter mask | +0.005 [-0.000, +0.010] | -0.012 [-0.021, -0.003] | -0.004 [-0.009, +0.002] | 0.09 |
| Multi-scale fusion + token gate: mean(prompter, SAM hybrid) vs. prompter mask | +0.005 [+0.002, +0.009] | -0.010 [-0.018, -0.002] | -0.002 [-0.007, +0.002] | 0.14 |
| Mask-prompt generator (segmentation loss): SAM + box vs. prompter mask | -0.002 [-0.009, +0.006] | -0.022 [-0.028, -0.016] | -0.012 [-0.017, -0.007] | 0.00 |
| Mask-prompt generator (segmentation loss): SAM + hybrid vs. prompter mask | -0.002 [-0.007, +0.003] | -0.015 [-0.024, -0.006] | -0.008 [-0.014, -0.003] | 0.00 |
| Mask-prompt generator (segmentation loss): mean(prompter, SAM hybrid) vs. prompter mask | -0.000 [-0.005, +0.005] | -0.013 [-0.021, -0.004] | -0.006 [-0.011, -0.001] | 0.01 |
| Mask-prompt generator (end-to-end via SAM): SAM + box vs. prompter mask | +0.001 [-0.008, +0.009] | +0.006 [+0.001, +0.012] | +0.004 [-0.002, +0.009] | 0.91 |
| Mask-prompt generator (end-to-end via SAM): SAM + hybrid vs. prompter mask | -0.001 [-0.005, +0.003] | +0.024 [+0.021, +0.028] | +0.012 [+0.009, +0.015] | 1.00 |
| Mask-prompt generator (end-to-end via SAM): mean(prompter, SAM hybrid) vs. prompter mask | -0.000 [-0.004, +0.003] | +0.024 [+0.020, +0.028] | +0.012 [+0.009, +0.015] | 1.00 |
| Mask-prompt generator (two-stage): SAM + box vs. prompter mask | -0.004 [-0.013, +0.004] | -0.023 [-0.028, -0.017] | -0.014 [-0.019, -0.008] | 0.00 |
| Mask-prompt generator (two-stage): SAM + hybrid vs. prompter mask | -0.001 [-0.007, +0.005] | +0.002 [-0.006, +0.010] | +0.001 [-0.005, +0.006] | 0.58 |
| Mask-prompt generator (two-stage): mean(prompter, SAM hybrid) vs. prompter mask | +0.001 [-0.004, +0.006] | +0.003 [-0.004, +0.011] | +0.002 [-0.003, +0.007] | 0.80 |
| Mask-prompt generator (segmentation loss, label smoothing): SAM + box vs. prompter mask | +0.001 [-0.006, +0.009] | -0.024 [-0.029, -0.018] | -0.011 [-0.016, -0.006] | 0.00 |
| Mask-prompt generator (segmentation loss, label smoothing): SAM + hybrid vs. prompter mask | -0.003 [-0.009, +0.002] | -0.011 [-0.019, -0.002] | -0.007 [-0.013, -0.001] | 0.01 |
| Mask-prompt generator (segmentation loss, label smoothing): mean(prompter, SAM hybrid) vs. prompter mask | -0.001 [-0.005, +0.004] | -0.009 [-0.017, -0.001] | -0.005 [-0.010, -0.000] | 0.02 |
| Mask-prompt generator + raw CG-Net fields: SAM + box vs. prompter mask | +0.000 [-0.008, +0.008] | -0.023 [-0.028, -0.017] | -0.011 [-0.016, -0.007] | 0.00 |
| Mask-prompt generator + raw CG-Net fields: SAM + hybrid vs. prompter mask | -0.003 [-0.007, +0.001] | -0.010 [-0.020, -0.001] | -0.007 [-0.012, -0.002] | 0.00 |
| Mask-prompt generator + raw CG-Net fields: mean(prompter, SAM hybrid) vs. prompter mask | -0.001 [-0.004, +0.002] | -0.009 [-0.017, -0.000] | -0.005 [-0.009, -0.000] | 0.01 |
| Box head on SAM features (YOLO-style): SAM + box vs. prompter mask | +0.011 [+0.005, +0.017] | +0.153 [+0.141, +0.165] | +0.082 [+0.075, +0.089] | 1.00 |
| Mask-prompt generator (segmentation loss) vs. CG-Net (fine-tuned) (prompter masks) | -0.005 [-0.030, +0.020] | +0.031 [+0.020, +0.043] | +0.013 [-0.000, +0.027] | 0.97 |
| Mask-prompt generator (segmentation loss) vs. CG-Net (trained from scratch on 358 images) (prompter masks) | +0.001 [-0.018, +0.023] | +0.018 [+0.009, +0.027] | +0.010 [-0.002, +0.022] | 0.95 |
| Mask-prompt generator (segmentation loss) vs. CG-Net (re-trained on 358 images, official init.) (prompter masks) | -0.006 [-0.027, +0.015] | +0.013 [+0.003, +0.024] | +0.004 [-0.008, +0.016] | 0.72 |
| CG-Net (trained from scratch on 358 images) vs. CG-Net (fine-tuned) (prompter masks) | -0.007 [-0.029, +0.016] | +0.013 [+0.003, +0.023] | +0.003 [-0.009, +0.016] | 0.69 |
| Mask-prompt generator (segmentation loss) vs. Multi-scale fusion (prompter masks) | +0.011 [-0.003, +0.025] | +0.010 [+0.005, +0.016] | +0.011 [+0.003, +0.019] | 0.99 |
| Mask-prompt generator (segmentation loss) vs. Logistic regression, ViT block 12 (prompter masks) | +0.044 [+0.025, +0.062] | +0.028 [+0.020, +0.036] | +0.036 [+0.025, +0.046] | 1.00 |
| Mask-prompt generator + raw CG-Net fields vs. Mask-prompt generator (segmentation loss) (prompter masks) | -0.004 [-0.012, +0.002] | -0.004 [-0.007, -0.001] | -0.004 [-0.008, -0.001] | 0.01 |
| Mask-prompt generator + raw CG-Net fields vs. CG-Net (trained from scratch on 358 images) (prompter masks) | -0.003 [-0.023, +0.018] | +0.014 [+0.004, +0.024] | +0.006 [-0.006, +0.019] | 0.82 |
| Multi-scale fusion + token gate vs. Multi-scale fusion (prompter masks) | -0.009 [-0.015, -0.003] | -0.004 [-0.005, -0.002] | -0.006 [-0.009, -0.003] | 0.00 |
| Logistic regression, ViT block 12 vs. Logistic regression, ViT block 1 (prompter masks) | +0.105 [+0.078, +0.133] | +0.066 [+0.053, +0.080] | +0.086 [+0.070, +0.101] | 1.00 |
| Mask-prompt generator (segmentation loss, label smoothing) vs. Mask-prompt generator (segmentation loss) (prompter masks) | -0.006 [-0.012, +0.000] | -0.001 [-0.004, +0.001] | -0.004 [-0.007, -0.000] | 0.01 |
| Multi-scale fusion vs. CG-Net (fine-tuned) (prompter masks) | -0.016 [-0.041, +0.008] | +0.021 [+0.009, +0.033] | +0.003 [-0.011, +0.016] | 0.64 |
| Mask-prompt generator (two-stage): SAM + hybrid vs. segmentation-loss generator mask | -0.007 [-0.013, -0.000] | -0.002 [-0.007, +0.003] | -0.004 [-0.008, -0.000] | 0.02 |
| Mask-prompt generator (end-to-end via SAM): SAM + hybrid vs. segmentation-loss generator mask | -0.011 [-0.020, -0.001] | -0.010 [-0.018, -0.003] | -0.011 [-0.016, -0.005] | 0.00 |
| Learned static prompts (4 tokens / class) vs. segmentation-loss generator mask | -0.107 [-0.142, -0.073] | -0.089 [-0.102, -0.077] | -0.098 [-0.117, -0.079] | 0.00 |
| Learned static prompts (16 tokens / class) vs. segmentation-loss generator mask | -0.097 [-0.131, -0.063] | -0.084 [-0.095, -0.072] | -0.090 [-0.110, -0.071] | 0.00 |


Object-level detection of the prompter masks:

| Prompter | TC recall | TC precision | AR recall | AR precision | TC obj./img | AR obj./img |
|---|---|---|---|---|---|---|
| Ground truth (oracle) | 1.000 | 1.000 | 1.000 | 1.000 | 2.9 | 6.8 |
| CG-Net (official weights) | 0.726 | 0.411 | 0.935 | 0.571 | 5.1 | 11.1 |
| CG-Net (fine-tuned) | 0.682 | 0.513 | 0.908 | 0.567 | 3.9 | 11.9 |
| CG-Net (re-trained on 358 images, official init.) | 0.687 ± 0.008 | 0.521 ± 0.044 | 0.903 ± 0.017 | 0.657 ± 0.011 | 3.9 | 9.8 |
| CG-Net (trained from scratch on 358 images) | 0.637 ± 0.015 | 0.562 ± 0.013 | 0.914 ± 0.018 | 0.623 ± 0.054 | 3.3 | 10.7 |
| Logistic regression, ViT block 1 | 0.767 ± 0.006 | 0.265 ± 0.016 | 0.967 ± 0.004 | 0.324 ± 0.018 | 8.8 | 25.2 |
| Logistic regression, ViT block 12 | 0.739 ± 0.012 | 0.419 ± 0.018 | 0.966 ± 0.004 | 0.418 ± 0.010 | 5.1 | 17.6 |
| Multi-scale fusion | 0.670 ± 0.030 | 0.529 ± 0.061 | 0.956 ± 0.001 | 0.482 ± 0.017 | 4.0 | 17.4 |
| Multi-scale fusion + token gate | 0.670 ± 0.020 | 0.448 ± 0.016 | 0.960 ± 0.005 | 0.444 ± 0.025 | 4.7 | 20.3 |
| Mask-prompt generator (segmentation loss) | 0.661 ± 0.009 | 0.563 ± 0.031 | 0.951 ± 0.001 | 0.514 ± 0.024 | 3.5 | 14.9 |
| Mask-prompt generator (end-to-end via SAM) | 0.652 ± 0.012 | 0.521 ± 0.050 | 0.963 ± 0.007 | 0.362 ± 0.107 | 3.7 | 90.0 |
| Mask-prompt generator (two-stage) | 0.667 ± 0.006 | 0.546 ± 0.039 | 0.962 ± 0.003 | 0.430 ± 0.008 | 3.7 | 18.0 |
| Mask-prompt generator (segmentation loss, label smoothing) | 0.648 ± 0.015 | 0.558 ± 0.026 | 0.953 ± 0.010 | 0.495 ± 0.007 | 3.5 | 15.2 |
| Mask-prompt generator + raw CG-Net fields | 0.657 ± 0.009 | 0.516 ± 0.013 | 0.950 ± 0.010 | 0.502 ± 0.054 | 3.8 | 14.9 |
| Box head on SAM features (YOLO-style) | nan | nan | nan | nan | nan | nan |


Where the IoU is lost (IoU of the prompter mask after fixing one error type):

| Prompter | Class | As is | No false objects | Add missed objects | Perfect detection | Perfect shape |
|---|---|---|---|---|---|---|
| CG-Net (official weights) | TC | 0.327 | 0.453 | 0.405 | 0.562 | 0.571 |
| CG-Net (fine-tuned) | TC | 0.349 | 0.439 | 0.457 | 0.575 | 0.620 |
| CG-Net (re-trained on 358 images, official init.) | TC | 0.350 ± 0.005 | 0.441 ± 0.018 | 0.460 ± 0.008 | 0.580 ± 0.004 | 0.631 ± 0.012 |
| CG-Net (trained from scratch on 358 images) | TC | 0.343 ± 0.007 | 0.410 ± 0.013 | 0.506 ± 0.006 | 0.605 ± 0.008 | 0.628 ± 0.007 |
| Logistic regression, ViT block 1 | TC | 0.228 ± 0.000 | 0.324 ± 0.008 | 0.304 ± 0.005 | 0.432 ± 0.004 | 0.593 ± 0.015 |
| Logistic regression, ViT block 12 | TC | 0.306 ± 0.003 | 0.393 ± 0.012 | 0.401 ± 0.007 | 0.514 ± 0.002 | 0.645 ± 0.012 |
| Multi-scale fusion | TC | 0.333 ± 0.002 | 0.405 ± 0.011 | 0.461 ± 0.018 | 0.560 ± 0.011 | 0.651 ± 0.005 |
| Multi-scale fusion + token gate | TC | 0.325 ± 0.001 | 0.395 ± 0.004 | 0.446 ± 0.009 | 0.542 ± 0.008 | 0.657 ± 0.005 |
| Mask-prompt generator (segmentation loss) | TC | 0.345 ± 0.005 | 0.420 ± 0.002 | 0.481 ± 0.015 | 0.585 ± 0.009 | 0.636 ± 0.012 |
| Mask-prompt generator (end-to-end via SAM) | TC | 0.337 ± 0.008 | 0.410 ± 0.005 | 0.474 ± 0.012 | 0.577 ± 0.010 | 0.637 ± 0.019 |
| Mask-prompt generator (two-stage) | TC | 0.340 ± 0.005 | 0.415 ± 0.013 | 0.470 ± 0.003 | 0.573 ± 0.014 | 0.641 ± 0.019 |
| Mask-prompt generator (segmentation loss, label smoothing) | TC | 0.340 ± 0.010 | 0.407 ± 0.011 | 0.483 ± 0.016 | 0.579 ± 0.015 | 0.645 ± 0.002 |
| Mask-prompt generator + raw CG-Net fields | TC | 0.339 ± 0.001 | 0.422 ± 0.005 | 0.465 ± 0.009 | 0.580 ± 0.007 | 0.627 ± 0.005 |
| CG-Net (official weights) | AR | 0.333 | 0.396 | 0.348 | 0.414 | 0.761 |
| CG-Net (fine-tuned) | AR | 0.382 | 0.437 | 0.421 | 0.482 | 0.787 |
| CG-Net (re-trained on 358 images, official init.) | AR | 0.399 ± 0.005 | 0.456 ± 0.008 | 0.437 ± 0.007 | 0.499 ± 0.005 | 0.805 ± 0.006 |
| CG-Net (trained from scratch on 358 images) | AR | 0.394 ± 0.008 | 0.453 ± 0.023 | 0.427 ± 0.006 | 0.490 ± 0.008 | 0.806 ± 0.022 |
| Logistic regression, ViT block 1 | AR | 0.319 ± 0.001 | 0.396 ± 0.004 | 0.324 ± 0.003 | 0.402 ± 0.002 | 0.744 ± 0.021 |
| Logistic regression, ViT block 12 | AR | 0.385 ± 0.001 | 0.463 ± 0.002 | 0.390 ± 0.002 | 0.468 ± 0.002 | 0.785 ± 0.004 |
| Multi-scale fusion | AR | 0.403 ± 0.004 | 0.468 ± 0.009 | 0.411 ± 0.005 | 0.478 ± 0.011 | 0.818 ± 0.017 |
| Multi-scale fusion + token gate | AR | 0.399 ± 0.003 | 0.462 ± 0.009 | 0.407 ± 0.001 | 0.471 ± 0.007 | 0.824 ± 0.016 |
| Mask-prompt generator (segmentation loss) | AR | 0.413 ± 0.001 | 0.479 ± 0.002 | 0.423 ± 0.001 | 0.490 ± 0.002 | 0.817 ± 0.004 |
| Mask-prompt generator (end-to-end via SAM) | AR | 0.378 ± 0.019 | 0.451 ± 0.018 | 0.383 ± 0.020 | 0.456 ± 0.017 | 0.801 ± 0.053 |
| Mask-prompt generator (two-stage) | AR | 0.409 ± 0.002 | 0.485 ± 0.003 | 0.415 ± 0.003 | 0.491 ± 0.002 | 0.789 ± 0.013 |
| Mask-prompt generator (segmentation loss, label smoothing) | AR | 0.412 ± 0.001 | 0.483 ± 0.006 | 0.421 ± 0.003 | 0.493 ± 0.003 | 0.806 ± 0.008 |
| Mask-prompt generator + raw CG-Net fields | AR | 0.409 ± 0.006 | 0.478 ± 0.003 | 0.419 ± 0.010 | 0.489 ± 0.002 | 0.808 ± 0.025 |


Size and training cost:

| Prompter | Trainable parameters | Training time (min) | Selected epoch |
|---|---|---|---|
| CG-Net (re-trained on 358 images, official init.) | 494,232 | 27.3 | 7 |
| CG-Net (trained from scratch on 358 images) | 494,232 | 27.9 | 17 |
| Logistic regression, ViT block 1 | 1,538 | 1.4 | 46 |
| Logistic regression, ViT block 12 | 1,538 | 2.0 | 39 |
| Multi-scale fusion | 1,406,474 | 82.6 | 38 |
| Multi-scale fusion + token gate | 1,415,180 | 46.0 | 39 |
| Mask-prompt generator (segmentation loss) | 671,138 | 3.7 | 29 |
| Mask-prompt generator (end-to-end via SAM) | 671,138 | 26.4 | 23 |
| Mask-prompt generator (two-stage) | 671,138 | 12.4 | 4 |
| Mask-prompt generator (segmentation loss, label smoothing) | 671,138 | 4.2 | 29 |
| Mask-prompt generator + raw CG-Net fields | 686,306 | 5.9 | 29 |
| Box head on SAM features (YOLO-style) | 608,778 | 100.2 | 26 |
| Learned static prompts (4 tokens / class) | 2,048 | 12.6 | 48 |
| Learned static prompts (16 tokens / class) | 8,192 | 14.2 | 48 |


Inference cost (`04_sam_feature_prompters/speed.csv`):

| Model | Parameters | GFLOPs / image | Latency (ms, A100, batch 1) |
|---|---|---|---|
| CG-Net | 494,232 | 22.93 | 8.89 |
| Logistic regression | 1,538 | 0.01 | 0.05 |
| Multi-scale fusion | 1,406,474 | 702.36 | 36.13 |
| Multi-scale fusion + token gate | 1,415,180 | 702.89 | 38.12 |
| Mask-prompt generator | 671,138 | 5.74 | 0.89 |
| (frozen ClimateSAM encoder, for reference) | 95,987,456 | 973.25 | 122.39 |
| (one SAM decoder call, one box, for reference) | 5,269,508 | 16.80 | 3.39 |


![](figures/sam_minus_prompter.png)
![](figures/prompters_mask_vs_sam.png)
![](figures/training_curves.png)
![](figures/error_decomposition.png)

Examples (map projection; filled = ground truth, lines = prediction, rectangles = box prompts):

![](figures/maps_test_image_17.png)

Architecture of the mask-prompt generator and of the benchmark:

![](figures/diagram_mask_prompt_generator.png)
![](figures/diagram_benchmark_pipeline.png)

## 4. Decoder adaptation (`05_decoder_adaptation/`)

| Prompter, prompts, training prompts | Output | TC IoU | AR IoU | Mean FG IoU | TC recall / prec. | AR recall / prec. |
|---|---|---|---|---|---|---|
| CG-Net (fine-tuned), bbox, in-sample | prompter mask | 0.349 | 0.382 | 0.366 |  |  |
|  | SAM, Phase-1 decoder | 0.341 | 0.369 | 0.355 | 0.67 / 0.56 | 0.90 / 0.53 |
|  | SAM, adapted decoder | 0.345 | 0.384 | 0.364 | 0.64 / 0.58 | 0.88 / 0.58 |
| Mask-prompt generator, bbox, out-of-fold | prompter mask | 0.346 | 0.414 | 0.380 |  |  |
|  | SAM, Phase-1 decoder | 0.344 | 0.392 | 0.368 | 0.65 / 0.59 | 0.95 / 0.50 |
|  | SAM, adapted decoder | 0.342 | 0.404 | 0.373 | 0.65 / 0.57 | 0.93 / 0.62 |
| Mask-prompt generator, bbox, in-sample | prompter mask | 0.346 | 0.414 | 0.380 |  |  |
|  | SAM, Phase-1 decoder | 0.344 | 0.392 | 0.368 | 0.65 / 0.59 | 0.95 / 0.50 |
|  | SAM, adapted decoder | 0.343 | 0.404 | 0.374 | 0.65 / 0.59 | 0.95 / 0.55 |
| Mask-prompt generator, hybrid, out-of-fold | prompter mask | 0.346 | 0.414 | 0.380 |  |  |
|  | SAM, Phase-1 decoder | 0.344 | 0.397 | 0.371 | 0.65 / 0.59 | 0.91 / 0.65 |
|  | SAM, adapted decoder | 0.347 | 0.405 | 0.376 | 0.65 / 0.60 | 0.94 / 0.64 |


Second checkpoint:

| Prompter, prompts, training prompts | Output | TC IoU | AR IoU | Mean FG IoU | TC recall / prec. | AR recall / prec. |
|---|---|---|---|---|---|---|
| Mask-prompt generator, bbox, out-of-fold | prompter mask | 0.337 | 0.414 | 0.376 |  |  |
|  | SAM, Phase-1 decoder | 0.340 | 0.399 | 0.369 | 0.62 / 0.55 | 0.95 / 0.56 |
|  | SAM, adapted decoder | 0.316 | 0.390 | 0.353 | 0.63 / 0.52 | 0.91 / 0.62 |
| Mask-prompt generator, bbox, in-sample | prompter mask | 0.337 | 0.414 | 0.376 |  |  |
|  | SAM, Phase-1 decoder | 0.340 | 0.399 | 0.369 | 0.62 / 0.55 | 0.95 / 0.56 |
|  | SAM, adapted decoder | 0.340 | 0.404 | 0.372 | 0.63 / 0.56 | 0.93 / 0.64 |
| Mask-prompt generator, hybrid, out-of-fold | prompter mask | 0.337 | 0.414 | 0.376 |  |  |
|  | SAM, Phase-1 decoder | 0.339 | 0.413 | 0.376 | 0.62 / 0.56 | 0.92 / 0.63 |
|  | SAM, adapted decoder | 0.338 | 0.416 | 0.377 | 0.63 / 0.55 | 0.94 / 0.62 |


## 5. Robustness: second frozen checkpoint (Infused Token, MLP 0.5)

Same protocol; multi-scale models 1 seed.

| Prompter | Output | TC IoU | AR IoU | BG IoU | Mean IoU | Mean FG IoU |
|---|---|---|---|---|---|---|
| Ground truth (oracle) | Prompter mask (no SAM) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
|  | SAM + box | 0.728 | 0.650 | 0.973 | 0.784 | 0.689 |
|  | SAM + hybrid | 0.857 | 0.905 | 0.993 | 0.918 | 0.881 |
| CG-Net (official weights) | Prompter mask (no SAM) | 0.327 | 0.333 | 0.926 | 0.529 | 0.330 |
|  | SAM + box | 0.318 | 0.342 | 0.931 | 0.530 | 0.330 |
|  | SAM + hybrid | 0.328 | 0.339 | 0.929 | 0.532 | 0.333 |
| CG-Net (fine-tuned) | Prompter mask (no SAM) | 0.349 | 0.382 | 0.940 | 0.557 | 0.366 |
|  | SAM + box | 0.341 | 0.375 | 0.940 | 0.552 | 0.358 |
|  | SAM + hybrid | 0.348 | 0.384 | 0.941 | 0.558 | 0.366 |
| Logistic regression, ViT block 1 | Prompter mask (no SAM) | 0.185 ± 0.001 | 0.305 ± 0.001 | 0.924 ± 0.003 | 0.471 ± 0.002 | 0.245 ± 0.001 |
|  | SAM + box | 0.186 ± 0.002 | 0.312 ± 0.003 | 0.925 ± 0.002 | 0.474 ± 0.000 | 0.249 ± 0.001 |
|  | SAM + hybrid | 0.186 ± 0.001 | 0.310 ± 0.001 | 0.928 ± 0.003 | 0.474 ± 0.001 | 0.248 ± 0.001 |
| Logistic regression, ViT block 12 | Prompter mask (no SAM) | 0.295 ± 0.004 | 0.388 ± 0.002 | 0.941 ± 0.001 | 0.542 ± 0.002 | 0.342 ± 0.003 |
|  | SAM + box | 0.297 ± 0.002 | 0.377 ± 0.001 | 0.937 ± 0.001 | 0.537 ± 0.002 | 0.337 ± 0.002 |
|  | SAM + hybrid | 0.300 ± 0.004 | 0.367 ± 0.006 | 0.946 ± 0.001 | 0.538 ± 0.003 | 0.333 ± 0.004 |
| Multi-scale fusion | Prompter mask (no SAM) | 0.330 | 0.409 | 0.943 | 0.561 | 0.370 |
|  | SAM + box | 0.337 | 0.397 | 0.941 | 0.558 | 0.367 |
|  | SAM + hybrid | 0.337 | 0.404 | 0.946 | 0.562 | 0.370 |
| Multi-scale fusion + token gate | Prompter mask (no SAM) | 0.319 | 0.404 | 0.941 | 0.555 | 0.362 |
|  | SAM + box | 0.327 | 0.390 | 0.938 | 0.552 | 0.358 |
|  | SAM + hybrid | 0.329 | 0.404 | 0.945 | 0.559 | 0.367 |
| Mask-prompt generator (segmentation loss) | Prompter mask (no SAM) | 0.330 ± 0.010 | 0.410 ± 0.005 | 0.944 ± 0.003 | 0.561 ± 0.004 | 0.370 ± 0.008 |
|  | SAM + box | 0.333 ± 0.008 | 0.395 ± 0.003 | 0.941 ± 0.003 | 0.557 ± 0.003 | 0.364 ± 0.006 |
|  | SAM + hybrid | 0.332 ± 0.011 | 0.404 ± 0.010 | 0.947 ± 0.002 | 0.561 ± 0.006 | 0.368 ± 0.010 |
| Mask-prompt generator (end-to-end via SAM) | Prompter mask (no SAM) | 0.323 ± 0.009 | 0.350 ± 0.015 | 0.947 ± 0.001 | 0.540 ± 0.007 | 0.336 ± 0.012 |
|  | SAM + box | 0.328 ± 0.008 | 0.383 ± 0.005 | 0.943 ± 0.003 | 0.552 ± 0.003 | 0.356 ± 0.006 |
|  | SAM + hybrid | 0.328 ± 0.009 | 0.357 ± 0.024 | 0.951 ± 0.001 | 0.546 ± 0.010 | 0.343 ± 0.016 |
| Mask-prompt generator (two-stage) | Prompter mask (no SAM) | 0.326 ± 0.005 | 0.410 ± 0.001 | 0.940 ± 0.002 | 0.559 ± 0.002 | 0.368 ± 0.003 |
|  | SAM + box | 0.333 ± 0.007 | 0.393 ± 0.004 | 0.937 ± 0.002 | 0.554 ± 0.004 | 0.363 ± 0.005 |
|  | SAM + hybrid | 0.331 ± 0.007 | 0.413 ± 0.002 | 0.945 ± 0.001 | 0.563 ± 0.002 | 0.372 ± 0.003 |
| Mask-prompt generator (segmentation loss, label smoothing) | Prompter mask (no SAM) | 0.334 ± 0.004 | 0.409 ± 0.003 | 0.945 ± 0.001 | 0.563 ± 0.002 | 0.371 ± 0.002 |
|  | SAM + box | 0.338 ± 0.008 | 0.398 ± 0.002 | 0.943 ± 0.001 | 0.560 ± 0.003 | 0.368 ± 0.004 |
|  | SAM + hybrid | 0.336 ± 0.004 | 0.403 ± 0.004 | 0.948 ± 0.000 | 0.562 ± 0.001 | 0.369 ± 0.002 |


## 6. Exploratory runs (`07_exploratory_runs/`)

| Run | Best epoch | SAM TC | SAM AR | Generator TC | Generator AR |
|---|---|---|---|---|---|
| end-to-end via SAM (from scratch) | 65 | 0.280 | 0.391 | 0.234 | 0.339 |
| end-to-end, 4 ViT blocks | 20 | 0.288 | 0.379 | 0.187 | 0.304 |
| segmentation loss only | 10 | 0.325 | 0.265 | 0.330 | 0.410 |
| two-stage (seg. loss -> via SAM) | 10 | 0.311 | 0.389 | 0.303 | 0.390 |


## Key findings, phrased for the thesis

- *Automatic prompting* (Section 4.3): a prompter only helps SAM as much as its prompts are correct, and SAM with
  generated prompts is bounded by the prompter. The upper bound with perfect prompts is high (hybrid prompts:
  TC 0.87 / AR 0.92), so the thesis' encoder adaptation works; the limit is prompt quality, i.e. *detection*.
- *Prompt type*: tight boxes are the best sparse prompt (confirming Table 4.12); points are worst; enlarging boxes
  hurts monotonically; dense mask prompts carry the AR shape and are more forgiving to prompt errors than boxes.
- *Where prompters fail*: AR — extent of rivers already detected (perfect shapes of detected ARs would give ~0.8 IoU);
  TC — missed cyclones and false blobs (perfect detection would give ~0.55–0.6 TC IoU).
- *Efficiency*: a 0.67 M-parameter head at 64×64 on the frozen features beats the 1.41 M-parameter multi-scale
  fusion model that runs at 1024×1024 (5.7 vs. 702 GFLOPs, 0.9 vs. 36 ms per image), and beats
  CG-Net on AR; even a linear probe on the last ViT block is close to CG-Net. The token gate adds nothing: the gate
  inputs are fixed decoder tokens, i.e. a learned per-class channel weighting. The adapted encoder is a strong representation for detection; SAM's decoder adds shape refinement only when
  the prompt is already right.
- *Recommendation for the pipeline*: use the prompter's own mask as the prompt-free output, or SAM with hybrid prompts
  when SAM-quality boundaries are wanted — the two are within ~0.01 IoU. To make SAM *improve* on the prompter, the
  decoder must be trained with realistic (erroneous) prompts from the start of Phase 1, or the detection step must
  improve; decoder fine-tuning afterwards is not enough.

## Bugs found in the original code (details in `04_sam_feature_prompters/README.md`)

1. `test_prompt_effect.py` → `ClimateSAM.set_infer_img()` skips the learned input adapter.
2. Multi-scale generators: the background channel is never supervised but takes part in the final argmax.
3. `LogisticRegressionPrompter` classifies the first ViT block (`feat_list[0]`).
4. `StreamSegMetrics.update(pred, gt)` called with swapped arguments (IoU unaffected, accuracies affected).
5. Best epochs selected on the test set in all original scripts.
6. The official CG-Net checkpoint has BatchNorm running statistics that do not match its weights.
7. The fine-tuned CG-Net saw all training images, so its outputs on training images are optimistic.

## Reproduce

```bash
cd ClimateSAM
.venv/bin/python prompter_bench/encoder_check.py                 # 00: checkpoint choice
.venv/bin/python prompter_bench/build_cache.py infused_mlp1 infused_mlp05   # encoder features (≈35 GB each, exp/feature_cache)
.venv/bin/python prompter_bench/oracle_sweep.py --encoder infused_mlp1      # 01
.venv/bin/python prompter_bench/table413_recheck.py                          # 02
.venv/bin/python prompter_bench/yolo_eval.py                                 # 03
bash prompter_bench/run_queue.sh infused_mlp1 light; bash prompter_bench/run_queue.sh infused_mlp1 heavy   # 04 training
bash prompter_bench/run_extra.sh infused_mlp1                                # 05 + label smoothing
bash prompter_bench/run_learned_prompt.sh infused_mlp1                       # learned static prompts
.venv/bin/python prompter_bench/evaluate.py --encoder infused_mlp1           # test evaluation of everything
.venv/bin/python prompter_bench/speed.py                                     # parameters / FLOPs / latency
.venv/bin/python prompter_bench/make_report.py                               # tables, figures, this README
.venv/bin/python prompter_bench/diagrams.py                                  # architecture diagrams
```
