| Prompter | TC recall | TC precision | AR recall | AR precision | TC obj./img | AR obj./img |
|---|---|---|---|---|---|---|
| Ground truth (oracle) | 1.000 | 1.000 | 1.000 | 1.000 | 2.9 | 6.8 |
| CG-Net (official weights) | 0.726 | 0.411 | 0.935 | 0.571 | 5.1 | 11.1 |
| CG-Net (fine-tuned) | 0.682 | 0.513 | 0.908 | 0.567 | 3.9 | 11.9 |
| Logistic regression, ViT block 1 | 0.745 ± 0.003 | 0.250 ± 0.002 | 0.940 ± 0.000 | 0.395 ± 0.020 | 8.7 | 18.8 |
| Logistic regression, ViT block 12 | 0.749 ± 0.015 | 0.362 ± 0.034 | 0.957 ± 0.006 | 0.447 ± 0.023 | 6.1 | 16.7 |
| Multi-scale fusion | 0.682 | 0.528 | 0.964 | 0.505 | 3.9 | 15.2 |
| Multi-scale fusion + token gate | 0.682 | 0.524 | 0.964 | 0.453 | 4.0 | 17.2 |
| Mask-prompt generator (segmentation loss) | 0.650 ± 0.023 | 0.548 ± 0.008 | 0.944 ± 0.013 | 0.558 ± 0.021 | 3.7 | 13.3 |
| Mask-prompt generator (end-to-end via SAM) | 0.670 ± 0.031 | 0.486 ± 0.040 | 0.965 ± 0.001 | 0.449 ± 0.055 | 4.3 | 26.2 |
| Mask-prompt generator (two-stage) | 0.655 ± 0.012 | 0.526 ± 0.006 | 0.960 ± 0.003 | 0.474 ± 0.024 | 3.8 | 16.9 |
| Mask-prompt generator (segmentation loss, label smoothing) | 0.669 ± 0.013 | 0.537 ± 0.027 | 0.941 ± 0.004 | 0.545 ± 0.027 | 3.7 | 14.3 |
