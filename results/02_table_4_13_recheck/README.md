# 02 — Re-check of Table 4.13 (SAM with CG-Net as prompter)

Script: `prompter_bench/table413_recheck.py` · data: `table_4_13_recheck.csv` · LaTeX: `../tables/table_4_13_corrected.tex`
(primary checkpoint) and `../tables/table_4_13_nosmooth.tex` · figures: `../figures/table_4_13_recheck.png`.

## What was re-run

Every row of Table 4.13, with the thesis pipeline itself: the fine-tuned CG-Net (`exp/cgnet_weight.pth`), its 3-class
argmax, `PromptMaker.make_prompts` with the same prompt types / point numbers / enlarge ratios, and
`ClimateSAM.infer` / `ClimateSAM.forward` for decoding. Metric: dataset-level IoU on the 61 test images
(the prompt configurations are evaluated on the test set, as in the original table).

Two checkpoints (both Infused Token, MLP ratio 1.0):

- `infused_token_vit_b_1.0_infused_token_vit_b_mlp1_CORRECTED` — the checkpoint of Table 4.12 (primary)
- `best_weights/infused_token_vit_b_1.0_infused_token_vit_b_mlp1_CORRECTED_NOSMOOTH` — the one used by the last run
  of `test_prompt_effect.py` (`exp/CGTEST_VIT_B_MLP1/prompt_effect_results_20260410_221245.csv`)

and two image paths:

- **orig. script** — `ClimateSAM.set_infer_img()`, which `test_prompt_effect.py` uses. It feeds the first three raw
  channels to the encoder and **skips the learned input adapter** (bug: `encode_images()` applies it, and the encoder
  was trained with it).
- **fixed** — `ClimateSAM.encode_images()`, the training path.

## Result

| | Table 4.13 (thesis) | re-run, orig. script, NOSMOOTH | re-run, fixed, primary |
|---|---|---|---|
| CG-Net alone | AR 0.375, TC 0.342 | AR 0.382, TC 0.349 | AR 0.382, TC 0.349 |
| SAM + bbox | AR **0.482**, TC **0.430** | AR 0.376, TC 0.340 | AR 0.369, TC 0.341 |
| SAM + point+bbox (10/5) | 0.461 / 0.422 | 0.356 / 0.343 | 0.343 / 0.324 |
| SAM + mask ({0,1} noisy masks) | 0.048 / 0.000 | 0.372 / 0.000 | 0.376 / 0.334 |

1. **The SAM rows of Table 4.13 do not reproduce.** The re-run of the original script reproduces the user's own later
   run exactly (bbox: 0.3404 / 0.3758 vs. 0.3409 / 0.3758 in the April-10 CSV), but not the table. No file in the
   repository, the logs or wandb contains the table's SAM values; they are most likely from an earlier state of the code.
2. **With correct numbers, SAM never beats the CG-Net it is prompted by**, for any prompt type or enlarge ratio
   (best: tight boxes, −0.008 TC / −0.013 AR vs. CG-Net alone).
3. The **ranking** of prompt types in the thesis table is confirmed: tight boxes > boxes + points > points; enlarging
   or shrinking boxes hurts monotonically; adding noisy masks to points + boxes hurts TC strongly.
4. The skipped input adapter costs 0.00–0.03 IoU — a real bug, but not the explanation of the table.
5. The "mask" row depends on the checkpoint: the NOSMOOTH checkpoint ignores {0, 1} dense prompts for TC (TC = 0.000,
   as in the thesis table), the primary checkpoint does not (0.334).
