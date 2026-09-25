# 03 — CG-Net with a YOLO box head as prompter

Script: `prompter_bench/yolo_eval.py` · data: `cgnet_yolo_<encoder>.csv`.

## Design (existing thesis model, `model/prompt/cgnet_bbox.py`)

CG-Net's 3-class classifier is replaced by a grid head predicting, per cell, an objectness score, a box (dx, dy, w, h)
and a class (TC / AR), trained with a YOLO-style loss on boxes extracted from the ground truth. At test time:
cells with objectness > threshold → boxes → NMS (IoU 0.4) → merging of fragmented boxes (distance 20 px). The boxes
are fed directly to SAM. This removes the connected-component step, but a box detector has no mask of its own.

All three saved checkpoints were evaluated with the primary frozen SAM, confidence thresholds 0.3 / 0.5 / 0.7.

## Result (primary checkpoint)

| Checkpoint | conf. | SAM TC | SAM AR | filled boxes TC / AR | TC recall / precision | AR recall / precision | objects / image (TC / AR) |
|---|---|---|---|---|---|---|---|
| `exp/cgnet_bbox_weight.pth` | 0.3 | 0.212 | 0.283 | 0.165 / 0.146 | 0.66 / 0.49 | 0.84 / 0.38 | 3.5 / 15.2 |
| `exp/cgnet_bbox_weight.pth` | 0.5 | 0.217 | 0.276 | 0.168 / 0.155 | 0.58 / 0.58 | 0.76 / 0.45 | 2.6 / 11.7 |
| `exp/cgnet_bbox_weight.pth` | 0.7 | 0.218 | 0.247 | 0.176 / 0.160 | 0.54 / 0.61 | 0.62 / 0.50 | 2.3 / 9.0 |
| `exp/best_weights/cgnet_bbox_weight.pth` | 0.3 | 0.209 | 0.283 | 0.162 / 0.150 | 0.63 / 0.53 | 0.79 / 0.39 | 3.0 / 14.1 |
| `exp/best_weights/cgnet_bbox_weight.pth` | 0.5 | 0.197 | 0.258 | 0.157 / 0.152 | 0.53 / 0.56 | 0.69 / 0.43 | 2.4 / 11.0 |
| `exp/best_weights/cgnet_bbox_weight.pth` | 0.7 | 0.197 | 0.234 | 0.167 / 0.157 | 0.47 / 0.58 | 0.58 / 0.47 | 2.1 / 8.6 |
| `exp/best_weights/best_exp_cgnet_bbox_weight.pth` | 0.3 | 0.196 | 0.289 | 0.153 / 0.155 | 0.60 / 0.52 | 0.79 / 0.44 | 3.0 / 12.4 |
| `exp/best_weights/best_exp_cgnet_bbox_weight.pth` | 0.5 | 0.192 | 0.258 | 0.158 / 0.160 | 0.54 / 0.58 | 0.68 / 0.46 | 2.4 / 10.6 |
| `exp/best_weights/best_exp_cgnet_bbox_weight.pth` | 0.7 | 0.190 | 0.225 | 0.157 / 0.155 | 0.53 / 0.61 | 0.55 / 0.47 | 2.2 / 8.4 |

Recall / precision / objects are measured on SAM's output (ground truth: 2.6 TC and 7.4 AR objects per image).



- SAM prompted by the YOLO boxes reaches **TC ≈ 0.21, AR ≈ 0.28**, clearly below SAM prompted by boxes derived
  from CG-Net's segmentation (0.34 / 0.37) and below CG-Net alone (0.35 / 0.38).
- The boxes themselves are imprecise: filled boxes give TC ≈ 0.16 / AR ≈ 0.15, and SAM's AR output has too many objects
  (8–15 per image vs. 7.4 in the ground truth) with low precision (0.38–0.50). From `01_oracle_prompts`: even ground-truth boxes enlarged
  by 20 % lose 0.26 (TC) and 0.14 (AR) IoU, so SAM needs *tight* boxes; a coarse grid detector cannot deliver them for thin, curved ARs.
- Raising the confidence threshold trades recall for precision but does not change the picture.

Conclusion: a dedicated box detector is the weakest prompter tested; deriving prompts from a segmentation is better.
