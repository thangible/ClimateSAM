# Plan for the remaining 20 hours

Goal: finish Section 4.3 with (1) the thesis prompters that were never run under the controlled protocol,
(2) the most promising new experiments from REPORT.md §6, (3) a LaTeX chapter, (4) wandb logs.
Budget: experiments finished after ~10–12 h, leaving time for writing. One A100; 2–3 jobs in parallel.

## Experiments (in launch order)

| # | Experiment | Why | GPU time | Lane |
|---|---|---|---|---|
| A1 | MSF + token gate + CGBlocks (`prompt_generator_token_cgblock.py`, thesis script `train_generator_token_cg.py`), 3 seeds | thesis prompter never run under the protocol | ~3 h | heavy |
| A2 | MSF + shared-parameter token gate (`prompt_generator_sp_token.py`, `train_generator_sp_token.py`), 3 seeds | same | ~2 h | heavy |
| B1 | CG-Net retrained on the 358-image split (official init, Jaccard loss as in the thesis plan, 3 seeds, selected on validation) | fair CG-Net baseline; the fine-tuned CG-Net saw the validation images | ~1.5 h | light |
| B2 | SAM-feature YOLO box head (`TokenGatedDetectionHead`, `train_det_head.py`) on the frozen embedding, 3 seeds, boxes -> SAM | thesis prompter never evaluated | ~0.5 h | light |
| B3 | **Prompt-robust decoder**: HQ decoder parts retrained with corrupted ground-truth prompts (loose / tight / shifted boxes, dropped objects, *false boxes with empty targets*, eroded / dilated masks) and out-of-fold generator prompts, one target per prompt; box and hybrid variants, 2–3 seeds | REPORT §6 idea 1 — the only intervention on the cause (decoder trusts its prompts) | ~3 h | light |
| C1 | Seed ensembling + per-class threshold calibration on validation | REPORT §6 idea 2, evaluation only | ~0.5 h | eval |
| C2 | Object post-processing: minimum blob size + TC latitude prior, tuned on validation | idea 3, evaluation only | (in C1) | eval |
| C3 | Iterative SAM refinement (SAM output fed back as dense prompt, 1–3 rounds) | idea 4, evaluation only | ~0.3 h | eval |

Every new model goes through the existing benchmark (`evaluate.py`): prompter mask, all prompt conversions, object
level, error decomposition, per-image counts for the bootstrap. B3 additionally: the oracle prompt-error study
(REPORT Figure 6) and all prompters with the robust decoder.

## Writing / logging (in parallel with the runs)

- D  wandb: every run logged in **offline mode** (no API key on this machine) under project `climatesam-section-4.3`;
     upload with `wandb login` then `wandb sync wandb/offline-run-*`.
- E  LaTeX: `results/latex/` — additions to the Approach (MPG architecture, training modes, evaluation protocol for the
     empty Section 3.6), Section 4.3 results, Discussion; tables `\input` from `results/tables/`, figures from
     `results/figures/*.pdf`.
- F  REPORT.md / README / figures updated with A–C.

## Stop rules

- Anything not finished at hour 14 is dropped from the report (logged as "not completed").
- Seeds are cut to 2 before any experiment is dropped.

## Status (updated during the run)

| # | Status |
|---|---|
| A1 | msf_token_cg is slow (≈210–240 s/epoch, ≈4 h per seed): cut to seed 0 (+ seed 1 only if time remains) |
| A2 | queued after A1 seed 0, 3 seeds |
| B1 | re-trained CG-Net (official init) and CG-Net from scratch, both with the Jaccard loss; scratch 3 seeds, official init 2 seeds |
| B2 | SAM-feature box head, 3 seeds (validation mean FG IoU of SAM with its boxes ≈ 0.18) |
| B3 | hybrid s0/s1 done; box mode needed ≈23 GB and ran out of memory next to other jobs → capped at 12 training boxes per class and image, re-queued; GT-only control re-run to also keep the last-epoch decoder (its selected epoch was 0) |
| C1–C3 | done for all finished prompters (`results/06_posthoc/`); seed ensemble +0.006–0.008, calibration / post-processing / SAM refinement do not help |
| new | **MPG + raw CG-Net fields** (`mpg_fields`): added because the fairly trained CG-Net nearly ties MPG — tests whether the raw fields add information to the SAM features; 3 seeds |
| new | cross-family ensemble MPG + CG-Net (scratch), evaluation only |
| D | new runs log to wandb offline; the 65 earlier runs were logged afterwards (`prompter_bench/wandb_log_existing.py runs`); test tables go into one run `test_results` at the end |
| E | `results/latex/` compiles (tectonic); sections for pending experiments are marked `%% PENDING` |
