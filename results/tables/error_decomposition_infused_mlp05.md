| Prompter | Class | As is | No false objects | Add missed objects | Perfect detection | Perfect shape |
|---|---|---|---|---|---|---|
| CG-Net (official weights) | TC | 0.327 | 0.453 | 0.405 | 0.562 | 0.571 |
| CG-Net (fine-tuned) | TC | 0.349 | 0.439 | 0.457 | 0.575 | 0.620 |
| Logistic regression, ViT block 1 | TC | 0.192 ± 0.001 | 0.280 ± 0.002 | 0.272 ± 0.002 | 0.397 ± 0.003 | 0.567 ± 0.008 |
| Logistic regression, ViT block 12 | TC | 0.296 ± 0.003 | 0.392 ± 0.018 | 0.383 ± 0.016 | 0.506 ± 0.003 | 0.632 ± 0.016 |
| Multi-scale fusion | TC | 0.330 | 0.410 | 0.450 | 0.560 | 0.632 |
| Multi-scale fusion + token gate | TC | 0.319 | 0.389 | 0.436 | 0.531 | 0.658 |
| Mask-prompt generator (segmentation loss) | TC | 0.329 ± 0.010 | 0.408 ± 0.015 | 0.470 ± 0.020 | 0.581 ± 0.014 | 0.615 ± 0.003 |
| Mask-prompt generator (end-to-end via SAM) | TC | 0.323 ± 0.009 | 0.400 ± 0.017 | 0.446 ± 0.021 | 0.553 ± 0.007 | 0.636 ± 0.025 |
| Mask-prompt generator (two-stage) | TC | 0.327 ± 0.006 | 0.401 ± 0.008 | 0.463 ± 0.010 | 0.568 ± 0.011 | 0.630 ± 0.009 |
| Mask-prompt generator (segmentation loss, label smoothing) | TC | 0.335 ± 0.004 | 0.418 ± 0.013 | 0.461 ± 0.017 | 0.575 ± 0.008 | 0.625 ± 0.030 |
| CG-Net (official weights) | AR | 0.333 | 0.396 | 0.348 | 0.414 | 0.761 |
| CG-Net (fine-tuned) | AR | 0.382 | 0.437 | 0.421 | 0.482 | 0.787 |
| Logistic regression, ViT block 1 | AR | 0.305 ± 0.001 | 0.369 ± 0.001 | 0.316 ± 0.001 | 0.382 ± 0.000 | 0.755 ± 0.012 |
| Logistic regression, ViT block 12 | AR | 0.388 ± 0.002 | 0.456 ± 0.004 | 0.397 ± 0.002 | 0.466 ± 0.003 | 0.808 ± 0.007 |
| Multi-scale fusion | AR | 0.409 | 0.477 | 0.415 | 0.483 | 0.817 |
| Multi-scale fusion + token gate | AR | 0.404 | 0.471 | 0.410 | 0.478 | 0.810 |
| Mask-prompt generator (segmentation loss) | AR | 0.410 ± 0.005 | 0.474 ± 0.015 | 0.423 ± 0.002 | 0.489 ± 0.009 | 0.817 ± 0.019 |
| Mask-prompt generator (end-to-end via SAM) | AR | 0.350 ± 0.015 | 0.397 ± 0.024 | 0.358 ± 0.016 | 0.406 ± 0.024 | 0.859 ± 0.018 |
| Mask-prompt generator (two-stage) | AR | 0.410 ± 0.001 | 0.484 ± 0.002 | 0.417 ± 0.003 | 0.492 ± 0.002 | 0.797 ± 0.008 |
| Mask-prompt generator (segmentation loss, label smoothing) | AR | 0.409 ± 0.003 | 0.473 ± 0.008 | 0.424 ± 0.003 | 0.490 ± 0.007 | 0.816 ± 0.011 |
