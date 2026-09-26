# 05 — Adapting SAM's decoder to automatically generated prompts

Script: `prompter_bench/train_decoder.py` · runs: `../runs/infused_mlp1/decoder_adapt_*/` (`log.csv`, `summary.json`,
`best_decoder.pth` = only the adapted decoder weights).

## Motivation

`01_oracle_prompts` shows that the Phase-1 decoder reproduces whatever it is prompted with: every false box becomes a
false object, loose boxes produce loose masks. That is the expected behaviour of a decoder trained with prompts made
from the ground truth ("perfect prompts", Section 3.1) — it never saw a wrong prompt. The question here: if the decoder
is fine-tuned on the prompts a real prompter produces, does it learn to *correct* them (ignore false blobs, fix
extents) so that SAM ends up better than the prompter?

## Design

- Prompter frozen; its prompts for every train / validation / test image computed once with the same conversion as
  in the benchmark (connected components ≥ 20 px; `bbox` or `hybrid`).
- Trainable: the decoder's HQ parts that Phase 1 trained — `hf_mlp_ar`, `hf_mlp_tc`, `compress_vit_feat`,
  `embedding_encoder`, `embedding_maskfeature`. SAM's original decoder layers stay frozen, and so do
  `hf_token_ar/tc`: the encoder's token adapters read the same weights, so changing them would change the (cached)
  encoder features.
- Target: the union of the per-prompt masks against the **full** class ground truth (so predicting nothing for a
  false prompt is rewarded, and missing an unprompted object is penalised but cannot be fixed); loss of Table 4.4.
- AdamW lr 3e-4, cosine, 30 epochs, batch 8 images; selection on the validation images; test once.
- **In-sample vs. out-of-fold prompts.** A prompter is much better on the images it was trained on than on new images
  (the fine-tuned CG-Net: validation FG IoU 0.45 vs. 0.37 on test, because it saw the "validation" images during its
  fine-tuning). A decoder trained on such prompts learns to trust prompts that are cleaner than the ones it gets at test
  time. The out-of-fold variant (`_oof`) avoids this: the 358 training images are split into 5 folds, five copies of the
  mask-prompt generator are trained without one fold each, and every training image gets its prompts from the copy that
  never saw it (validation and test prompts: the normal generator, which never saw them either).

## Runs

| Run | Prompter | Prompts | Training prompts |
|---|---|---|---|
| `decoder_adapt_cgnet_finetuned_bbox_s0` | fine-tuned CG-Net | boxes | in-sample (CG-Net saw all training images) |
| `decoder_adapt_mpg_seg_s0_bbox_s0` | mask-prompt generator (seed 0) | boxes | in-sample |
| `decoder_adapt_mpg_seg_s0_bbox_oof_s0` | mask-prompt generator (seed 0) | boxes | out-of-fold |
| `decoder_adapt_mpg_seg_s0_hybrid_oof_s0` | mask-prompt generator (seed 0) | hybrid | out-of-fold |

## Results (`../tables/decoder_adaptation_*`, `../figures/decoder_adaptation.png`)

Primary checkpoint, test set (prompter mask → SAM with the Phase-1 decoder → SAM with the adapted decoder):

| Prompter / prompts / training prompts | Prompter mask FG | SAM, Phase-1 decoder FG | SAM, adapted decoder FG |
|---|---|---|---|
| CG-Net / boxes / in-sample | 0.366 | 0.355 | 0.364 |
| Mask-prompt generator / boxes / in-sample | 0.380 | 0.368 | 0.374 |
| Mask-prompt generator / boxes / out-of-fold | 0.380 | 0.368 | 0.373 |
| Mask-prompt generator / hybrid / out-of-fold | 0.380 | 0.371 | 0.376 |

Second checkpoint (mask-prompt generator mask: 0.376 FG): adapted decoder with in-sample boxes 0.372, out-of-fold
boxes 0.353, out-of-fold hybrid 0.377 — the same pattern.

- The adapted decoder does learn to reject some false prompts: AR object precision 0.50 → 0.62 with out-of-fold
  boxes, 0.53 → 0.58 with CG-Net boxes.
- It recovers most of what SAM loses relative to the prompter, but in no configuration does SAM end up clearly above
  the prompter's own mask (best: +0.001 … −0.004).
- Out-of-fold vs. in-sample prompts makes no difference on the primary checkpoint and hurts box prompts on the
  second one — the limiting factor is not the prompt distribution, but that the decoder cannot recover missed
  objects or infer the extent of an AR better than the prompter already did.

## Prompt-robust decoder (`prompter_bench/train_robust_decoder.py`, `run_robust*.sh`, `run_robust_eval.sh`)

**Idea.** Attack the cause directly: the Phase-1 decoder trusts its prompts because it only ever saw correct ones.
Train the same decoder parts (1.21 M parameters) on *corrupted* ground-truth prompts mixed 1:1 with out-of-fold MPG
prompts, where the target of a box is the complete ground-truth object(s) it touches and an **empty mask for a false
box**. Corruptions: object left unprompted p = 0.15; box sides moved by U(−0.15, 0.3) × box size; Binomial(3, 0.25)
false boxes per class; dense prompts eroded / dilated (kernel ≤ 5×5 px) plus a false blob with p = 0.3 (a real object
moved elsewhere). At most 12 training boxes per class and image (memory). 25 epochs, lr 3e-4, selection on validation
with MPG prompts; modes *box* and *hybrid*, 2 seeds each; control: corrupted GT prompts only.

**Evaluation.** Every decoder goes through the oracle study (`../01_oracle_prompts/oracle_sweep_infused_mlp1@<decoder>.csv`)
and the benchmark with real prompters (`../eval/infused_mlp1/decoder/<decoder>/`).
Table: `../tables/robust_decoder_infused_mlp1.*`; figure: `../figures/robust_decoder.png`.

| Decoder (mean FG IoU, test) | GT box | GT box +20 % | GT box + 1 false | GT box + mask | MPG: box | MPG: hybrid | CG-Net: box | CG-Net: hybrid |
|---|---|---|---|---|---|---|---|---|
| Phase-1 decoder | 0.674 | 0.474 | 0.576 | 0.877 | 0.367 | 0.370 | 0.355 | 0.367 |
| robust, box (seeds 0 / 1) | 0.630 / 0.636 | 0.559 / 0.564 | 0.609 / 0.608 | 0.838 / 0.847 | 0.371 / 0.372 | 0.360 / 0.364 | 0.363 / 0.365 | 0.368 / 0.369 |
| robust, hybrid (seeds 0 / 1) | 0.592 / 0.548 | 0.511 / 0.527 | 0.524 / 0.503 | 0.856 / 0.867 | 0.332 / 0.317 | 0.375 / 0.376 | 0.320 / 0.309 | 0.367 / 0.369 |
| corrupted GT only, last epoch | 0.517 | 0.468 | 0.477 | 0.881 | 0.291 | 0.360 | 0.287 | 0.367 |

(MPG's own mask: 0.379; the fine-tuned CG-Net's: 0.366.)

**Findings.**
- Robust to synthetic box errors: enlarged boxes 0.474 → 0.56, false boxes 0.576 → 0.61; tight boxes 0.674 → 0.63.
- Little transfer to real prompters: MPG boxes +0.005 [+0.002, +0.008], MPG hybrid +0.005–0.006 [+0.004, +0.007]
  (0.375–0.376, still −0.003 … −0.004 below MPG's mask) — the same gain as the plain out-of-fold adaptation.
- Each decoder specialises to its prompt mode (hybrid-trained with boxes: −0.035 … −0.050).
- Corrupted GT prompts alone: validation IoU with MPG prompts falls every epoch, so selection keeps epoch 0 (= Phase-1
  decoder); the last-epoch decoder is worse with real prompts (MPG hybrid −0.010, boxes −0.076). Hand-designed
  corruptions do not reproduce the errors of a real prompter.
- Conclusion: making the decoder robust does not lift SAM above its prompter; the remaining errors (missed cyclones,
  AR extent) are not in the prompt.

Note: the first run of the GT-only control (`runs/infused_mlp1/robust_decoder_hybrid_gtonly_s0_v1`) saved only the
selected (epoch-0) decoder; it was re-run to keep the last-epoch decoder as well.
