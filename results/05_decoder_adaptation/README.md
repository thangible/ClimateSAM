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

## Results (`../tables/decoder_adaptation_*`, `../figures/decoder_adaptation_*.png`)

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
