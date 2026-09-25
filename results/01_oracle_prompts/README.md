# 01 — Oracle prompts: what SAM can do with prompts built from the ground truth

Scripts: `prompter_bench/oracle_sweep.py`, `prompter_bench/evaluate.py` (method `oracle_gt`) · data:
`oracle_sweep_<encoder>.csv`, `../eval/<encoder>/oracle_gt.json` · figures: `../figures/oracle_mask_format_*.png`,
`../figures/oracle_degradation_*.png`.

These experiments use no prompter at all. They answer two questions that decide how a prompter should talk to SAM:
(1) which prompt *format* the frozen decoder understands, and (2) how SAM reacts to the kinds of errors a real
prompter makes (boxes too large / too small, masks too thin / too thick, missed objects, false objects).
Test set, 61 images; TC / AR IoU.

## 1. Prompt types with perfect (ground-truth) prompts — primary checkpoint

| Prompt per object | TC | AR |
|---|---|---|
| tight box | 0.723 | 0.625 |
| box enlarged 10 % | 0.586 | 0.569 |
| 5 + 1 positive, 5 negative points | 0.527 | 0.577 |
| box + points | 0.583 | 0.604 |
| dense mask (±10 logits) | 0.774 | 0.920 |
| box + dense mask | 0.872 | 0.883 |
| **hybrid**: TC box + mask, AR mask | **0.872** | **0.920** |

The box row reproduces thesis Table 4.12 (Infused Token 1.0, bbox: 0.724 / 0.632). A mask prompt carries the shape
and gives the best AR; a box is needed to localise the small TCs. These are *upper bounds*: a GT mask prompt already
contains the answer.

## 2. How the decoder reads dense mask prompts (`study = mask_format`)

| Mask prompt | primary (MLP 1.0) TC / AR | MLP 0.5 TC / AR |
|---|---|---|
| {0, 1} union of all objects | 0.850 / 0.895 | **0.000** / 0.909 |
| logits ±2 | 0.534 / 0.860 | 0.002 / 0.499 |
| logits ±5 | 0.794 / 0.916 | 0.185 / 0.913 |
| logits ±10 | 0.774 / 0.920 | 0.225 / 0.905 |
| logits ±20 | 0.841 / 0.916 | 0.461 / 0.898 |
| box | 0.723 / 0.625 | 0.728 / 0.650 |
| box + logits ±10 | 0.872 / 0.883 | 0.856 / 0.867 |

- One prompt per class (union of all objects) works as well as one prompt per object; one decoder call per class
  and image is enough for dense prompts.
- Whether a checkpoint accepts dense prompts for **TC** is a property of its Phase-1 training: the MLP 0.5 checkpoint
  ignores them for TC (the prompt encoder shrinks 256×256 to 64×64, so a cyclone is a few cells), the MLP 1.0 one does
  not. **A box + mask prompt is robust for TC on both checkpoints** (0.86–0.87), hence the hybrid design.
- On the first exploratory checkpoint {0, 1} masks failed completely for both classes (`07_exploratory_runs`), which
  also explains the "mask" row of Table 4.13.

## 3. Controlled prompt errors (`study = degradation`, primary; MLP 0.5 in the CSV is within ±0.03 for boxes)

| Error | TC | AR |
|---|---|---|
| none (tight GT box) | 0.723 | 0.625 |
| box shrunk 10 % / 20 % / 30 % | 0.664 / 0.426 / 0.200 | 0.505 / 0.318 / 0.157 |
| box enlarged 10 % / 20 % / 30 % / 50 % | 0.589 / 0.463 / 0.357 / 0.234 | 0.569 / 0.484 / 0.409 / 0.294 |
| 25 % / 50 % of the objects not prompted | 0.548 / 0.375 | 0.483 / 0.350 |
| 1 / 3 random false boxes added per class | 0.574 / 0.413 | 0.578 / 0.484 |
| GT mask prompt (±10) | 0.774 | 0.920 |
| mask eroded 3 / 5 / 9 px | 0.706 / 0.612 / 0.455 | 0.905 / 0.864 / 0.764 |
| mask dilated 3 / 5 / 9 px | 0.826 / 0.852 / 0.795 | 0.916 / 0.890 / 0.817 |

What this means for automatic prompting:

1. **SAM does not correct prompt errors, it reproduces them.** Every false box becomes a false object (one extra box
   per class costs ~0.15 TC / 0.05 AR), and an object without a prompt is never found.
2. **Boxes must be tight.** ±10–20 % box error already costs 0.1–0.3 IoU — the same order of magnitude as the whole
   gap between a good prompter and the oracle. Box extent is exactly what prompters get wrong for ARs.
3. **Dense prompts are forgiving:** a mask that is 3–5 px too thick even *helps* TC, and a thinner one degrades
   gracefully. This is why the mask / hybrid prompts are the best way to pass an imperfect prompter output to SAM.
