# 00 — Common protocol of all Section 4.3 experiments

Every number in `results/` (except `07_exploratory_runs/`) was produced with the protocol below, so the rows of all
tables are directly comparable.

## Frozen SAM (Phase-1 checkpoint)

The prompters are evaluated with one frozen Phase-1 model. `encoder_candidates_gt_prompts.csv` shows every candidate
checkpoint prompted with ground-truth boxes, points and ±10 mask logits (61 test images, same metric as below).

| Tag used in `results/` | Checkpoint (`exp/…pth`) | Matches thesis row | GT box TC / AR |
|---|---|---|---|
| `infused_mlp1` (**primary**) | `infused_token_vit_b_1.0_infused_token_vit_b_mlp1_CORRECTED` | Table 4.12, Infused Token 1.0, bbox: 0.7242 / 0.6323 | 0.723 / 0.625 |
| `infused_mlp05` (robustness check) | `infused_token_vit_b_0.5_infused_token_vit_b_mlp05_CORRECTED` | Table 4.12, Infused Token 0.5, bbox: 0.7244 / 0.6448 | 0.728 / 0.650 |

The primary checkpoint is the model Section 4.3 says it uses (Infused Token, MLP ratio 1). The small differences to
Table 4.12 come from the connected-component threshold (20 px here, as in `PromptMaker`, vs. 50 px in the dataset's
prompt extraction).

Note from the same table: whether the decoder understands **dense mask prompts** depends strongly on the checkpoint
(GT mask logits give TC IoU between 0.0006 and 0.77 depending on the checkpoint). The primary checkpoint handles them
well (TC 0.77 / AR 0.92), which matters for the mask-prompt experiments.

## Data split

- ClimateNet train set (398 images) → **358 train / 40 validation** (fixed random permutation, seed 0).
  The validation images are used only to select the checkpoint of every learned prompter.
- ClimateNet test set (61 images) → **test**, touched once per final model.
- `split.json` lists the file names of every split and the 5 folds used for out-of-fold prompts (`05_decoder_adaptation`).

The original thesis scripts selected checkpoints on the test set; numbers selected that way are optimistic, so the
learned prompters here never see the test set before the final evaluation.

## Features

The image encoder (input adapter + ViT-B + Infused Token adapters) is frozen in Phase 2, and no augmentation is used
(Section 2.2.2), so the encoder is run once per image and its outputs are cached (`prompter_bench/build_cache.py`,
float16): the neck embedding (256×64×64), all 12 ViT block outputs (64×64×768) and the four CG-Net input fields.
This makes a 60-epoch training run of a small prompter take a few minutes instead of hours.

## Training (all learned prompters)

| Setting | Value |
|---|---|
| Loss | Tversky + Focal, parameters of Table 4.4 (TC: Tversky 0.3/0.7, Focal α 0.95 γ 5; AR: Tversky 0.5/0.5, Focal α 0.85 γ 5) |
| Outputs | two sigmoid channels (TC, AR); prediction = logit > 0 |
| Optimiser | AdamW, lr 1e-3, weight decay 1e-4, cosine decay to 1e-5, 60 epochs, batch 8 (gradient accumulation where memory requires) |
| Deep supervision | multi-scale fusion models only, as in thesis Eq. 3.4 (weights 0.4/(i+1)) |
| Selection | validation mean foreground IoU, evaluated every 2 epochs |
| Seeds | 3 (0, 1, 2); tables report mean ± standard deviation |

## Evaluation

- **Pixel IoU** per class, summed over all test pixels (TP / (TP + FP + FN)), as in the thesis tables.
  Where both classes are predicted on a pixel, AR wins (as in `test_prompt_effect.py`). Mean IoU = mean of TC, AR, BG;
  Mean FG IoU = mean of TC and AR.
- **Prompter mask**: the prompter's own TC / AR maps (logits upsampled bilinearly to 768×1152, threshold 0).
  CG-Net: its 3-class argmax.
- **SAM + …**: the prompter mask converted into prompts by the thesis `PromptMaker` (connected components ≥ 20 px),
  decoded by the frozen SAM, per-object masks merged by union (`ClimateSAM.assemble_raw_masks` logic):

  | Output | Prompts per connected component |
  |---|---|
  | SAM + box | tight box |
  | SAM + box (+10%) | box enlarged by 10% of its size |
  | SAM + points | 5 positive (eroded region) + centroid + 5 negative (ring around the object); TC uses min(3, n/3) |
  | SAM + box + points | both of the above |
  | SAM + mask logits | one dense prompt per class: the prompter's logit map resized to 256×256 (clamped to ±20); hard masks → ±10 |
  | SAM + box + mask | every box together with the class logit map |
  | SAM + hybrid | TC: box + mask logits, AR: mask logits only |

- **Object level**: a ground-truth object (≥ 20 px) is *found* if any predicted pixel overlaps it; a predicted object
  is *correct* if it overlaps any ground-truth object. Recall = found / GT objects, precision = correct / predicted.
- **Error decomposition** of the prompter mask: IoU after (a) deleting predicted objects that touch no ground truth,
  (b) adding the missed ground-truth objects, (c) both, (d) replacing every detected ground-truth object by its exact
  shape (false objects kept). The gap between "as is" and each variant is the IoU lost to that error type.

## Hardware / cost

Single NVIDIA A100 80 GB. Caching one encoder: ~3 min for 459 images. Training times are in `tables/cost_*.tex`.
