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

## Status — finished (2026-09-26, 08:30)

| # | Status | Result |
|---|---|---|
| A1 | done, 2 seeds (≈1.5–3.5 h per seed) | CG-block token gate 0.340 mean FG IoU, −0.023 vs. token gate |
| A2 | done, 3 seeds | shared-weight token gate 0.367, +0.006 vs. token gate, −0.011 vs. MPG |
| B1 | done: official init 2 seeds, scratch 3 seeds | 0.375 / 0.369 (stored checkpoint 0.366); MPG better only for AR (+0.013 / +0.018) |
| B2 | done, 2 seeds (third dropped: far below all others) | SAM + its boxes 0.205 |
| B3 | done: box / hybrid × 2 seeds + GT-only control (re-run to keep the last epoch) | robust to synthetic box errors, +0.005 with real prompts, still below the prompter |
| C1–C3 | done for all prompters | seed ensemble +0.006–0.011; calibration / post-processing / SAM refinement: no gain |
| new | MPG + raw fields, 3 seeds | 0.375, −0.004 vs. MPG |
| new | MPG + CG-Net ensemble | 0.389, best mask of the study |
| D | 92 offline wandb runs (new runs + 65 earlier runs + `test_results` tables) | upload: `wandb login && wandb sync wandb/offline-run-*` |
| E | `results/latex/` complete, compiles (32 pages standalone) | |
| F | REPORT.md, README, folder READMEs, tables, figures updated | |
