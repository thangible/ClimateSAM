# 04 — Prompters, compared under one protocol (Section 4.3)

Scripts: `prompter_bench/train.py`, `prompter_bench/prompters.py`, `prompter_bench/evaluate.py` · per-run logs and
checkpoints: `../runs/<encoder>/<run>/` (`log.csv` = every epoch, `summary.json`, `best.pth`) · test results:
`../eval/<encoder>/<run>.json` and `../eval/<encoder>/all_results.csv` · tables: `../tables/main_*.tex`,
`../tables/prompt_conversion_*.tex`, `../tables/object_level_*.tex`, `../tables/error_decomposition_*.tex`,
`../tables/cost_*.tex` · figures: `../figures/prompters_mask_vs_sam.png`, `../figures/training_curves.png`,
`../figures/error_decomposition.png`, `../figures/maps_test_image_*.png`.

Protocol: `../00_setup/README.md` (frozen primary checkpoint, 358 / 40 / 61 split, Table 4.4 loss, 60 epochs,
selection on validation, 3 seeds).

## The prompters

All learned prompters read the cached outputs of the frozen ClimateSAM encoder and predict two logit maps (TC, AR).
They were re-implemented behind one interface (`prompter_bench/prompters.py`); the network code is the thesis code in
`model/`.

| Prompter | Input | Architecture | Trainable params | Notes |
|---|---|---|---|---|
| CG-Net (official) | TMQ, U850, V850, PSL | ClimateNet CG-Net, official weights (`pretrained/weights_cgnet.pth`) | 494 k | not retrained |
| CG-Net (fine-tuned) | same | the thesis' fine-tuned CG-Net (`exp/cgnet_weight.pth`) | 494 k | fine-tuned on **all** 398 training images |
| Logistic regression, ViT block 1 | block-1 output (64×64×768) | 1×1 conv 768 → 2 | 1.5 k | = thesis `LogisticRegressionPrompter`, which reads `feat_list[0]`, i.e. the **first** ViT block |
| Logistic regression, ViT block 12 | last block | same | 1.5 k | the obvious fix: the last block carries the semantics |
| Multi-scale fusion | all 12 blocks | thesis Figure 3.9 (`model/prompt_generator.py`), 4 groups of 3 blocks, nearest upsampling to 1024², deep supervision | 1.41 M | 2-channel head (see bug 2 below) |
| Multi-scale fusion + token gate | all 12 blocks | `model/prompt_generator_token.py`: the fused features gated per channel by the decoder's refined TC / AR HQ tokens, two binary heads (these produce the prompts) + the multiclass head as auxiliary output | 1.42 M | the gate inputs are fixed vectors (the decoder is frozen), so the gate is a learned per-class channel weighting |
| **Mask-prompt generator** (new) | neck embedding + blocks 6 and 12 | 1×1 projections → 3 ConvNeXt blocks at 64×64 (dilation 1/2/4) → 2 transposed convs to 256×256 → 2 logits (`model/mask_prompt_generator.py`, `../figures/diagram_mask_prompt_generator.png`) | 0.67 M | outputs directly at SAM's dense-prompt resolution |

Training modes of the mask-prompt generator (`../figures/diagram_training_modes.png`):

- **segmentation loss** — like every other prompter: loss on its own maps.
- **end-to-end via SAM** — the maps are turned into prompts differentiably (AR: logit map → dense prompt;
  TC: one box per blob + TC logit map as dense prompt) and passed through the frozen prompt encoder and decoder;
  loss on SAM's output + 0.5 × loss on the own maps. The box coordinates are not differentiated; the gradient reaches
  the generator through the dense prompts.
- **two-stage** — the segmentation-loss model (same seed), then 30 epochs end-to-end at lr 1e-4 (own-map loss
  weight 1).
- **label smoothing** (ablation) — segmentation loss with the Gaussian-smoothed targets of Section 3.5
  (TC kernel 3 σ 5, AR kernel 9 σ 20, Table 4.5).

## Bugs in the original prompter pipelines (fixed here)

1. `test_prompt_effect.py` decodes images from `ClimateSAM.set_infer_img()`, which **skips the input adapter**
   (see `02_table_4_13_recheck`; −0.00…−0.03 IoU).
2. The multi-scale generators predict 3 channels but `compute_generator_loss` supervises only channels 1 (TC) and
   2 (AR) as independent sigmoids; the **background channel receives no loss**, yet the final prediction is the
   3-channel argmax, i.e. TC / AR compete with an untrained random projection. Here all prompters use two sigmoid
   channels thresholded at 0.
3. The logistic-regression prompter classifies the **first** ViT block (`feat_list[0]`), whose features are close
   to the input; both blocks are compared here.
4. Several scripts call `StreamSegMetrics.update(pred, gt)` although the signature is `update(gt, pred)`. IoU is
   symmetric (unaffected); Mean / FreqW accuracy are not.
5. All original scripts select the best epoch on the test set (optimistic numbers).
6. `exp/cgnet_weight.pth` was fine-tuned on all 398 training images, so its masks on training / validation images
   are far better than on test images (validation mean FG IoU 0.45 vs. 0.37 on test) — relevant whenever its outputs
   on training images are used (`05_decoder_adaptation`).
7. The official ClimateNet CG-Net checkpoint (`pretrained/weights_cgnet.pth`) predicts background everywhere in
   `eval()` mode: its stored BatchNorm running statistics do not match its weights. It is evaluated here with batch
   statistics (batch of 4 images, running statistics untouched), which restores sensible predictions; this is probably
   also why it needed fine-tuning.

## Results (primary checkpoint; full tables in the top-level `README.md` and `../tables/`)

Prompter masks (no SAM), mean over 3 seeds, test set:

| Prompter | Params | TC IoU | AR IoU | Mean FG IoU |
|---|---|---|---|---|
| CG-Net, official weights | 494 k | 0.327 | 0.333 | 0.330 |
| CG-Net, fine-tuned | 494 k | 0.349 | 0.382 | 0.366 |
| Logistic regression, ViT block 1 | 1.5 k | 0.195 | 0.319 | 0.257 |
| Logistic regression, ViT block 12 | 1.5 k | 0.301 | 0.385 | 0.343 |
| Multi-scale fusion | 1.41 M | 0.333 | 0.403 | 0.368 |
| Multi-scale fusion + token gate | 1.42 M | 0.324 | 0.399 | 0.362 |
| **Mask-prompt generator** | 0.67 M | **0.344** | **0.413** | **0.379** |
| Mask-prompt generator, label smoothing | 0.67 M | 0.338 | 0.412 | 0.375 |
| Mask-prompt generator, end-to-end via SAM | 0.67 M | 0.335 | 0.378 | 0.356 |
| Mask-prompt generator, two-stage | 0.67 M | 0.338 | 0.409 | 0.374 |

SAM prompted by these masks (`../tables/prompt_conversion_*`, `../tables/bootstrap_*`,
`../figures/sam_minus_prompter.png`):

- Points are always clearly worse than the prompter mask (−0.03 … −0.08 mean FG IoU).
- Tight boxes: roughly equal for TC, −0.01 … −0.02 for AR.
- Hybrid prompts (TC box + mask, AR mask): the best conversion; within ±0.01 of the prompter mask, slightly positive
  only for the weaker prompters (logistic regression on block 1: +0.007; end-to-end generator: +0.012) and slightly
  negative for the strong ones (multi-scale fusion −0.009, mask-prompt generator −0.009).
- Averaging the prompter's and SAM's probabilities (`fused_hybrid`) behaves like the hybrid output.

Significance (paired bootstrap over the 61 test images, seeds pooled): the mask-prompt generator beats the fine-tuned
CG-Net on AR (+0.031, 95 % CI [+0.020, +0.043]) with equal TC (−0.005 [−0.030, +0.020]), multi-scale fusion on mean
FG IoU (+0.011 [+0.003, +0.019]) and the block-12 linear probe (+0.036 [+0.025, +0.046]). The token gate lowers
multi-scale fusion by −0.006 [−0.009, −0.003]; label smoothing −0.004 [−0.007, −0.000].

Object level (`../tables/object_level_*`): all prompters find 91–97 % of the ARs but only 64–77 % of the TCs; the
precision of the predicted blobs is 0.4–0.6 (logistic regression: 0.27–0.42, i.e. many fragments). Error decomposition
(`../tables/error_decomposition_*`): for AR, giving every detected river its true shape lifts IoU from ~0.41 to
~0.8, while perfect detection only reaches ~0.49; for TC, perfect detection (~0.58) and perfect shapes (~0.64)
matter about equally.

Efficiency (`speed.csv`): parameters, forward FLOPs and latency per prompter.
