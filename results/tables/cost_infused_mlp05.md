| Prompter | Trainable parameters | Training time (min) | Selected epoch |
|---|---|---|---|
| Logistic regression, ViT block 1 | 1,538 | 2.9 | 41 |
| Logistic regression, ViT block 12 | 1,538 | 3.0 | 39 |
| Multi-scale fusion | 1,406,474 | 35.5 | 42 |
| Multi-scale fusion + token gate | 1,415,180 | 37.9 | 32 |
| Mask-prompt generator (segmentation loss) | 671,138 | 4.8 | 25 |
| Mask-prompt generator (end-to-end via SAM) | 671,138 | 18.5 | 39 |
| Mask-prompt generator (two-stage) | 671,138 | 8.0 | 9 |
| Mask-prompt generator (segmentation loss, label smoothing) | 671,138 | 3.3 | 33 |
