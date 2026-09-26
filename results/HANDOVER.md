# Handover: Section 4.3 (automatic prompting) of the ClimateSAM thesis

For an agent that will write or finish the thesis. It explains (1) what the thesis author did before, (2) what was
done in the final experiment campaign (September 2026), (3) where every result lives, and (4) how the material maps
onto the thesis. All paths are relative to the repository root (`ClimateSAM/`, branch `lastrun`).

**Read in this order:** this file → `results/REPORT.md` (the complete scientific write-up, ~10k words, every number
verified against the raw outputs) → `results/latex/` (draft LaTeX for Chapter 3 additions, Section 4.3, Discussion,
Appendix) → `results/README.md` (all generated tables). The thesis PDF itself is **not** in the repository; ask the
author for it. Section / table / figure numbers below (e.g. "Table 4.13") refer to that PDF.

---

## 1. The thesis in brief

The thesis adapts the Segment Anything Model (SAM) to segment **tropical cyclones (TC)** and **atmospheric rivers (AR)**
in the ClimateNet dataset (16 climate variables per image, 768×1152 grid; 398 training and 61 test images; expert
labels background / TC / AR). It has two phases:

- **Phase 1 — promptable ClimateSAM.** SAM ViT-B with a learned linear input adapter (16 → 3 channels), encoder
  adapters ("Infused Token"; variants with MLP ratio 1.0 and 0.5; LoRA variants were also explored), and an HQ-SAM-style
  mask decoder with separate HQ tokens for TC and AR. Images are stretched to 1024×1024. Trained with prompts derived
  from the ground truth ("perfect prompts"). Loss chosen in Section 4.1 (Tversky + Focal, parameters in Table 4.4);
  Gaussian label smoothing studied in Table 4.5. Result with ground-truth box prompts (Table 4.12): Infused Token 1.0
  TC 0.724 / AR 0.632 IoU, Infused Token 0.5 TC 0.724 / AR 0.645.
- **Phase 2 — prompt-free ClimateSAM (Section 4.3).** Replace the ground-truth prompts by prompts from an automatic
  **prompter**. The thesis structure for this section: 4.3.1 Prompt Analysis, 4.3.2 SAM with CGNet as Prompter,
  4.3.3 Generator from SAM Inputs. Section 3.6 (metrics / evaluation protocol) was empty.

Other thesis objects referenced by the new material: Table 2.1 (mean expert IoU vs. the consensus labels: AR 0.341 /
TC 0.257), Section 2.2.2 (ClimateNet data, TC latitude range, no augmentation), Section 3.1 (perfect prompts),
Figure 3.9 (multi-scale fusion generator), Eq. 3.4 (deep supervision), Section 3.2.3 (CG-Net, Jaccard loss).

## 2. What the author did before this campaign

All in the repository history before September 2026 (commits by Thang / thangible, April 2025 – April 2026).

- **Phase 1** training pipeline and checkpoints: `model/climatesam.py`, `model/image_encoder.py`, `model/mask_decoder.py`,
  `model/input_adapter.py`, training in `train_script/`; checkpoints in `exp/` (not in git), e.g.
  `exp/infused_token_vit_b_1.0_infused_token_vit_b_mlp1_CORRECTED.pth` and `..._0.5_..._mlp05_CORRECTED.pth`.
- **Phase 2 prompters** (code in `model/` and `train_script/official/`):
  - CG-Net (ClimateNet baseline) as prompter; fine-tuned checkpoint `exp/cgnet_weight.pth`; prompt conversion with
    `model/prompt/prompt_maker.py` (`PromptMaker`); evaluation script `train_script/official/test_prompt_effect.py`
    → **Table 4.13** (SAM + CG-Net boxes reported as TC 0.430 / AR 0.482 vs. CG-Net alone 0.342 / 0.375).
  - CG-Net with a YOLO-style box head (`CGNetBBoxPrompter`, `train_cgnet_bbox*.py`).
  - Logistic-regression prompter on ViT features (`model/logistic_regression_prompter.py`).
  - Multi-scale fusion generator (Figure 3.9, `model/prompt_generator.py`, `train_generator.py`) and token-gated
    variants: `prompt_generator_token.py`, `..._token_cgblock.py` (CG blocks), `..._sp_token.py` (shared weights).
  - Token-gated YOLO detection head on SAM features (`model/detection_head.py`, `train_det_head.py`).
  - Further scripts that were **not** evaluated in this campaign: `propose_and_score.py`,
    `train_adaptation_with_cgnet*.py`, `train_adaptation_with_det_head.py`, LoRA scripts, `train_tune_token_concat.py`.
- **The problem:** the author reported that the results of Section 4.3 were incorrect, and none of the prompters made
  SAM work without prompts. This campaign re-did Section 4.3 from scratch under one controlled protocol.

## 3. What was done in this campaign

### 3.1 New benchmark code (`prompter_bench/`)

One frozen Phase-1 model, a feature cache (the frozen encoder is run once; outputs stored in `exp/feature_cache/`, not
in git), one prompter interface, one evaluation. Key files:

| File | Role |
|---|---|
| `common.py` | loading ClimateSAM, split, feature cache, SAM decoding, prompt construction, metrics |
| `prompters.py` | every prompter behind one interface (`build(arch)`) |
| `train.py` | trains any prompter (segmentation loss, or through SAM: `--mode sam`) |
| `evaluate.py` | test evaluation: prompter mask + 7 prompt conversions, object metrics, error decomposition, per-image counts |
| `oracle_sweep.py` | ground-truth prompts with controlled errors (`--decoder` to use a replaced decoder) |
| `table413_recheck.py` | re-run of Table 4.13 with the original pipeline |
| `train_decoder.py`, `train_robust_decoder.py` | decoder adaptation / prompt-robust decoder |
| `train_learned_prompt.py`, `train_det_head_bench.py`, `yolo_eval.py` | static prompts, SAM-feature box head, CG-Net YOLO boxes |
| `posthoc.py` | ensembles, threshold calibration, post-processing, iterative SAM refinement |
| `speed.py` | parameters / FLOPs / latency |
| `make_report.py`, `figures.py`, `diagrams.py`, `style.py` | all tables, figures (one colour grammar) and diagrams |
| `run_*.sh` | the job queues that produced every run |

New model: **mask-prompt generator (MPG)**, `model/mask_prompt_generator.py` (0.67 M parameters, 5.7 GFLOPs).

### 3.2 Protocol (the content for the empty Section 3.6)

- Frozen primary model: Infused Token, MLP ratio 1.0 (`infused_mlp1`); robustness check: MLP ratio 0.5 (`infused_mlp05`).
- Split: 398 training images → 358 train / 40 validation (fixed permutation); 61 test images used once per final model.
  (The original scripts selected epochs on the test set.)
- Training: loss of Table 4.4, AdamW lr 1e-3, cosine, 60 epochs, batch 8, selection on validation mean FG IoU,
  3 seeds (some experiments 2 seeds, marked in the tables).
- Seven prompt conversions: box, box +10 %, points, box + points, mask logits, box + mask, **hybrid** (TC: box + mask,
  AR: mask). Connected components ≥ 20 px (as `PromptMaker`).
- Metrics: dataset-level IoU per class (AR wins overlaps), mean FG IoU = mean of TC and AR; object recall / precision;
  error decomposition (IoU after removing false objects / adding missed objects / giving detected objects their true
  shape); paired bootstrap 95 % CIs over the 61 test images (2000 resamples, seeds pooled).

### 3.3 Experiments and findings

| # | Experiment | Result | Where |
|---|---|---|---|
| 1 | Oracle: SAM with ground-truth prompts, prompt formats, controlled prompt errors | hybrid prompts TC 0.872 / AR 0.920; box 0.723 / 0.625. SAM reproduces prompt errors: false box → false object, missed object stays missed, ±20 % box size costs 0.1–0.3 IoU. Dense-prompt handling is checkpoint-dependent (MLP 0.5 ignores {0,1} TC masks) | `01_oracle_prompts/` |
| 2 | Re-check of Table 4.13 with the original pipeline | **not reproducible**: SAM + CG-Net boxes TC 0.341 / AR 0.369, *below* CG-Net alone 0.349 / 0.382. Ranking of prompt types confirmed. `test_prompt_effect.py` skips the input adapter (−0.00…−0.03). | `02_table_4_13_recheck/` |
| 3 | CG-Net YOLO box head → SAM | weakest: TC 0.19–0.22 / AR 0.22–0.29 | `03_cgnet_yolo_boxes/` |
| 4 | All prompters under one protocol | MPG best single prompter: TC 0.344 / AR 0.413 (mean FG 0.379); MSF 0.368; token gate 0.362; CG blocks 0.340; shared-weight token gate 0.367; LogReg block 12 0.343 (block 1: 0.257); box head on SAM features → SAM 0.205 | `04_sam_feature_prompters/` |
| 5 | CG-Net re-trained under the protocol | 0.375 (official init, 2 seeds) / 0.369 (scratch) vs. stored checkpoint 0.366. MPG better only for AR (+0.013 / +0.018, significant); mean FG difference not significant | `04_…`, `tables/bootstrap_infused_mlp1` |
| 6 | SAM prompted by each prompter | **never better than its prompter** (hybrid within ±0.01; boxes −0.01…−0.02 AR; points −0.03…−0.08). Only exception: prompters with a poor mask (CG-block MSF +0.017) — still below MPG | `figures/sam_minus_prompter` |
| 7 | Training MPG through SAM (end-to-end, two-stage) | own mask worse; best SAM output 0.374 < MPG mask 0.379 | `04_…` |
| 8 | Error analysis | AR: found (>90 %) but extent wrong (perfect shape of detected ARs → 0.82 IoU); TC: one third missed, half of predicted blobs false | `tables/error_decomposition_*`, `object_level_*` |
| 9 | Decoder fine-tuned on generated (out-of-fold) prompts | rejects some false prompts, still ≤ prompter (best −0.004) | `05_decoder_adaptation/` |
| 10 | Prompt-robust decoder (corrupted prompts, empty targets for false boxes) | robust to synthetic box errors (+20 % boxes 0.474 → 0.56 mean FG), only +0.005 with real prompts, still below prompter | `05_…`, `figures/robust_decoder` |
| 11 | Learned static prompts (no prompter) | 0.280 (4 tokens) / 0.288 (16): a fixed query cannot decide which region is a TC/AR | `04_…` |
| 12 | MPG + raw CG-Net fields | 0.375, −0.004: the fields add nothing to the SAM features | `04_…` |
| 13 | Evaluation-only: seed ensemble, calibration, post-processing, iterative SAM refinement | ensemble +0.006 (MPG 0.385); calibration / post-processing no gain; refinement loses every round | `06_posthoc/` |
| 14 | Ensemble MPG + CG-Net (3 + 3 models) | **best mask of the study: 0.389** (TC 0.363 / AR 0.416); gain over MPG ensemble not significant | `06_posthoc/` |
| 15 | Second frozen checkpoint (MLP 0.5) | same conclusions; MPG ties MSF (0.370) | `tables/main_infused_mlp05` |
| 16 | Cost | MPG 5.7 GFLOPs / 0.9 ms vs. MSF 702 / 36 ms; encoder 973 GFLOPs; CG-Net 23 | `tables/speed` |

**The storyline for Section 4.3 and the Discussion:**
1. With perfect prompts the adapted SAM is excellent, so Phase 1 works (RQ1).
2. The earlier Table 4.13 claim (SAM improves on CG-Net) does not hold (RQ2).
3. A small generator on the frozen encoder features (MPG) is the best single prompter and far cheaper than MSF; a CG-Net
   trained under the same protocol nearly ties it (RQ3).
4. SAM never improves on its prompter, whatever the prompter, prompt type, training through SAM, or decoder adaptation /
   robustness training (RQ4).
5. The reason: the remaining errors (missed TCs, AR extent) are not in the prompt, and the decoder was trained to trust
   its prompt (RQ5). The foundation model's contribution is the encoder; the promptable decoder is useful for
   interactive use, not as a refinement stage.

### 3.4 Bugs found in the original code (Appendix material)

1. `test_prompt_effect.py` uses `ClimateSAM.set_infer_img()`, which skips the learned input adapter.
2. Multi-scale generators: 3 output channels, only TC/AR supervised, untrained background channel in the argmax.
3. `LogisticRegressionPrompter` reads the first ViT block (`feat_list[0]`), not the last.
4. `StreamSegMetrics.update(gt, pred)` called with swapped arguments (IoU unaffected).
5. Original scripts select the best epoch on the test set.
6. Official CG-Net weights: BatchNorm running statistics do not match (works only with batch statistics).
7. The fine-tuned CG-Net saw all 398 training images (incl. our validation images).

## 4. Key numbers (test set, primary checkpoint, mean FG IoU unless stated)

| System | TC | AR | Mean FG |
|---|---|---|---|
| SAM + ground-truth hybrid prompts (upper bound) | 0.872 | 0.920 | 0.896 |
| SAM + ground-truth boxes (≈ Table 4.12) | 0.723 | 0.625 | 0.674 |
| CG-Net, stored fine-tuned checkpoint | 0.349 | 0.382 | 0.366 |
| SAM + its boxes (corrected Table 4.13) | 0.341 | 0.369 | 0.355 |
| CG-Net re-trained, official init / scratch | 0.350 / 0.343 | 0.399 / 0.394 | 0.375 / 0.369 |
| LogReg ViT block 1 / block 12 | 0.195 / 0.301 | 0.319 / 0.385 | 0.257 / 0.343 |
| MSF / + token gate / + CG blocks / shared weights | 0.333 / 0.324 / 0.324 / 0.332 | 0.403 / 0.399 / 0.355 / 0.402 | 0.368 / 0.362 / 0.340 / 0.367 |
| **MPG** (3 seeds, ± ≤ 0.003) | **0.344** | **0.413** | **0.379** |
| SAM + MPG hybrid / box | 0.342 / 0.343 | 0.398 / 0.391 | 0.370 / 0.367 |
| MPG 3-seed ensemble | 0.353 | 0.418 | 0.385 |
| MPG + CG-Net ensemble (best) | 0.363 | 0.416 | 0.389 |
| Learned static prompts (4 / class) | 0.237 | 0.324 | 0.280 |
| Mean single expert vs. consensus (Table 2.1) | 0.257 | 0.341 | — |

Every number above is in `results/tables/*.csv|md|tex` or `results/REPORT.md`; take numbers from there, not from
memory. Bootstrap CIs: `results/tables/bootstrap_infused_mlp1.*`.

## 5. How the material maps onto the thesis

| Thesis place | Source | Status |
|---|---|---|
| Section 3.6 (evaluation protocol, was empty) + new method sections (MPG, training modes, decoder adaptation / robust decoder) | `results/latex/approach.tex` | draft, complete |
| Section 4.3 (4.3.1 Prompt Analysis, 4.3.2 SAM with CGNet as Prompter, 4.3.3 Generator from SAM Inputs, plus error analysis, decoder adaptation, evaluation-only improvements, static prompts, second checkpoint, qualitative, summary) | `results/latex/section_4_3.tex` | draft, complete; replaces the old Section 4.3 |
| Discussion (why SAM cannot improve on its prompter, why a small generator wins, prompt design, implications, limitations, future work) | `results/latex/discussion.tex` | draft |
| Appendix (complete tables, second example map, code issues) | `results/latex/appendix.tex` | draft |
| Tables | `results/tables/*.tex` (booktabs + adjustbox), `\input` by the LaTeX files | generated by `make_report.py` |
| Figures | `results/figures/*.pdf`; single panels in `results/figures/panels/` | generated by `figures.py` / `diagrams.py` |

`results/latex/main.tex` compiles everything standalone (32 pages, e.g. with `tectonic main.tex`);
`results/latex/README.md` explains the `\ResultsDir` macro and required packages.

**What the thesis writer must still do:**
1. Replace the fixed cross-references in the LaTeX drafts (e.g. "Table~4.12", "Figure~3.9", "Eq.~3.4",
   "Section~2.2.2", "Table~2.1") by `\ref{...}` with the thesis' own labels.
2. **Remove or correct everything elsewhere in the thesis that relies on the old Table 4.13** (abstract, introduction,
   contributions, conclusion): SAM prompted by CG-Net is *not* better than CG-Net. The old table's SAM values could not
   be reproduced; say so neutrally ("could not be reproduced with the current code and checkpoints; most likely from an
   earlier state of the code"), do not speculate further.
3. State anywhere relevant that model selection now uses a held-out validation split (the original scripts used the
   test set).
4. Keep the research-question structure (RQ1–RQ5) or fold it into the thesis' own framing; the conclusions of
   `REPORT.md` §5 are the agreed answers.
5. Adapt the tone and length to the rest of the thesis; the drafts are dense.

## 6. Rules for claims (please keep)

- n = 61 test images: differences below ~0.01 IoU are within the bootstrap intervals. Say "not significant" where the
  95 % CI contains 0 (see `tables/bootstrap_*`).
- Distinguish the **stored fine-tuned CG-Net** (0.366, trained outside the protocol) from the **CG-Net re-trained under
  the protocol** (0.369–0.375). Against the latter, MPG is better only for ARs; do not write "MPG clearly beats CG-Net".
- "Best single prompter" = MPG; "best mask" = MPG + CG-Net ensemble (six models vs. three, gain over the MPG ensemble
  not significant).
- SAM never exceeds its prompter in this study; the only positive SAM effects are for prompters with a poor own mask.
- Robust decoder: robust to *synthetic* errors, +0.005 with real prompts. Do not present it as a fix.
- Some configurations have 2 seeds (CG-Net from official init, CG-block MSF, box head, robust decoders); tables say so.
- The CG-Net re-trained from the official ClimateNet weights inherits that those weights saw the validation images
  (selection optimistic, test numbers fine); CG-Net from scratch is clean.
- Label noise argument: good prompters (TC 0.33–0.35, AR 0.38–0.41) already exceed the mean single expert (Table 2.1).

## 7. Where the raw data is

| Path | Content |
|---|---|
| `results/runs/<checkpoint>/<run>/` | `log.csv` (every epoch: losses, lr, validation metrics), `summary.json` (all arguments, selected epoch, validation metrics, training time), checkpoint (`best.pth`, decoder runs `best_decoder.pth` / `last_decoder.pth`) |
| `results/eval/<checkpoint>/<run>.json` | test metrics of every output, error decomposition, per-image counts (for the bootstrap); `all_results.csv` = everything in one table; `decoder/<decoder run>/` = evaluations with a replaced decoder |
| `results/01_…/` – `07_…/` | per-experiment folders with a README (design, results, interpretation) and CSVs |
| `results/tables/` | every table as `.tex`, `.csv`, `.md` |
| `results/figures/` | every figure as `.pdf` / `.png` (+ `panels/`) |
| `results/logs/` | stdout of every job |
| `results/PLAN_20h.md` | the plan of the final 20 hours and its status |

Colour grammar of all figures (keep it if new figures are made; `prompter_bench/style.py`): TC green `#008300`, AR blue
`#2a78d6` (fill = ground truth, outline = prediction); prompter mask grey, SAM + box orange, SAM + hybrid violet,
points yellow, static prompts magenta, adapted decoder aqua; MPG highlighted red, other methods grey.

No experiment tracker is used any more (wandb was removed); everything is in git.

## 8. Naming (run keys → names used in the thesis)

| Key | Name |
|---|---|
| `infused_mlp1` / `infused_mlp05` | frozen Phase-1 model, Infused Token MLP 1.0 (primary) / 0.5 |
| `cgnet_official`, `cgnet_finetuned` | CG-Net official weights / stored fine-tuned checkpoint |
| `cgnet_train_seg`, `cgnet_scratch_seg` | CG-Net re-trained from official init / from scratch |
| `logreg_l0_seg`, `logreg_last_seg` | logistic regression on ViT block 1 / block 12 |
| `msf_seg`, `msf_token_seg`, `msf_token_cg_seg`, `msf_sp_token_seg` | multi-scale fusion (MSF), + token gate, + CG blocks, shared weights |
| `mpg_seg`, `mpg_seg_smooth`, `mpg_sam_e2e`, `mpg_twostage`, `mpg_fields_seg` | mask-prompt generator: segmentation loss, label smoothing, end-to-end via SAM, two-stage, + raw fields |
| `det_head` | YOLO-style box head on SAM features |
| `learned_prompt_k4/_k16` | learned static prompts |
| `decoder_adapt_*`, `robust_decoder_*` | decoder adaptation / prompt-robust decoder |
| `_s0/_s1/_s2` | seed |

## 9. Regenerating

```bash
.venv/bin/python prompter_bench/make_report.py     # tables + results/README.md (from results/runs, results/eval)
.venv/bin/python prompter_bench/figures.py         # figures + panels
cd results/latex && tectonic main.tex              # standalone PDF of the chapter
```
Re-running experiments needs the feature cache (`prompter_bench/build_cache.py`, ~35 GB per checkpoint) and the
Phase-1 checkpoints in `exp/`; see `REPORT.md` Appendix C.
