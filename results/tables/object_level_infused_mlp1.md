| Prompter | TC recall | TC precision | AR recall | AR precision | TC obj./img | AR obj./img |
|---|---|---|---|---|---|---|
| Ground truth (oracle) | 1.000 | 1.000 | 1.000 | 1.000 | 2.9 | 6.8 |
| CG-Net (official weights) | 0.726 | 0.411 | 0.935 | 0.571 | 5.1 | 11.1 |
| CG-Net (fine-tuned) | 0.682 | 0.513 | 0.908 | 0.567 | 3.9 | 11.9 |
| CG-Net (re-trained on 358 images, official init.) | 0.687 ± 0.008 | 0.521 ± 0.044 | 0.903 ± 0.017 | 0.657 ± 0.011 | 3.9 | 9.8 |
| CG-Net (trained from scratch on 358 images) | 0.637 ± 0.015 | 0.562 ± 0.013 | 0.914 ± 0.018 | 0.623 ± 0.054 | 3.3 | 10.7 |
| Logistic regression, ViT block 1 | 0.767 ± 0.006 | 0.265 ± 0.016 | 0.967 ± 0.004 | 0.324 ± 0.018 | 8.8 | 25.2 |
| Logistic regression, ViT block 12 | 0.739 ± 0.012 | 0.419 ± 0.018 | 0.966 ± 0.004 | 0.418 ± 0.010 | 5.1 | 17.6 |
| Multi-scale fusion | 0.670 ± 0.030 | 0.529 ± 0.061 | 0.956 ± 0.001 | 0.482 ± 0.017 | 4.0 | 17.4 |
| Multi-scale fusion + token gate | 0.670 ± 0.020 | 0.448 ± 0.016 | 0.960 ± 0.005 | 0.444 ± 0.025 | 4.7 | 20.3 |
| Multi-scale fusion + token gate, CG blocks | 0.698 ± 0.040 | 0.492 ± 0.073 | 0.978 ± 0.007 | 0.331 ± 0.060 | 4.2 | 20.5 |
| Multi-scale fusion + token gate, shared weights | 0.680 ± 0.018 | 0.472 ± 0.018 | 0.956 ± 0.005 | 0.459 ± 0.013 | 4.4 | 17.7 |
| Mask-prompt generator (segmentation loss) | 0.661 ± 0.009 | 0.563 ± 0.031 | 0.951 ± 0.001 | 0.514 ± 0.024 | 3.5 | 14.9 |
| Mask-prompt generator (end-to-end via SAM) | 0.652 ± 0.012 | 0.521 ± 0.050 | 0.963 ± 0.007 | 0.362 ± 0.107 | 3.7 | 90.0 |
| Mask-prompt generator (two-stage) | 0.667 ± 0.006 | 0.546 ± 0.039 | 0.962 ± 0.003 | 0.430 ± 0.008 | 3.7 | 18.0 |
| Mask-prompt generator (segmentation loss, label smoothing) | 0.648 ± 0.015 | 0.558 ± 0.026 | 0.953 ± 0.010 | 0.495 ± 0.007 | 3.5 | 15.2 |
| Mask-prompt generator + raw CG-Net fields | 0.657 ± 0.009 | 0.516 ± 0.013 | 0.950 ± 0.010 | 0.502 ± 0.054 | 3.8 | 14.9 |
