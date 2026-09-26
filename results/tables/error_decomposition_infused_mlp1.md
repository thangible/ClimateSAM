| Prompter | Class | As is | No false objects | Add missed objects | Perfect detection | Perfect shape |
|---|---|---|---|---|---|---|
| CG-Net (official weights) | TC | 0.327 | 0.453 | 0.405 | 0.562 | 0.571 |
| CG-Net (fine-tuned) | TC | 0.349 | 0.439 | 0.457 | 0.575 | 0.620 |
| CG-Net (re-trained on 358 images, official init.) | TC | 0.350 ± 0.005 | 0.441 ± 0.018 | 0.460 ± 0.008 | 0.580 ± 0.004 | 0.631 ± 0.012 |
| CG-Net (trained from scratch on 358 images) | TC | 0.343 ± 0.007 | 0.410 ± 0.013 | 0.506 ± 0.006 | 0.605 ± 0.008 | 0.628 ± 0.007 |
| Logistic regression, ViT block 1 | TC | 0.228 ± 0.000 | 0.324 ± 0.008 | 0.304 ± 0.005 | 0.432 ± 0.004 | 0.593 ± 0.015 |
| Logistic regression, ViT block 12 | TC | 0.306 ± 0.003 | 0.393 ± 0.012 | 0.401 ± 0.007 | 0.514 ± 0.002 | 0.645 ± 0.012 |
| Multi-scale fusion | TC | 0.333 ± 0.002 | 0.405 ± 0.011 | 0.461 ± 0.018 | 0.560 ± 0.011 | 0.651 ± 0.005 |
| Multi-scale fusion + token gate | TC | 0.325 ± 0.001 | 0.395 ± 0.004 | 0.446 ± 0.009 | 0.542 ± 0.008 | 0.657 ± 0.005 |
| Mask-prompt generator (segmentation loss) | TC | 0.345 ± 0.005 | 0.420 ± 0.002 | 0.481 ± 0.015 | 0.585 ± 0.009 | 0.636 ± 0.012 |
| Mask-prompt generator (end-to-end via SAM) | TC | 0.337 ± 0.008 | 0.410 ± 0.005 | 0.474 ± 0.012 | 0.577 ± 0.010 | 0.637 ± 0.019 |
| Mask-prompt generator (two-stage) | TC | 0.340 ± 0.005 | 0.415 ± 0.013 | 0.470 ± 0.003 | 0.573 ± 0.014 | 0.641 ± 0.019 |
| Mask-prompt generator (segmentation loss, label smoothing) | TC | 0.340 ± 0.010 | 0.407 ± 0.011 | 0.483 ± 0.016 | 0.579 ± 0.015 | 0.645 ± 0.002 |
| Mask-prompt generator + raw CG-Net fields | TC | 0.339 ± 0.001 | 0.422 ± 0.005 | 0.465 ± 0.009 | 0.580 ± 0.007 | 0.627 ± 0.005 |
| CG-Net (official weights) | AR | 0.333 | 0.396 | 0.348 | 0.414 | 0.761 |
| CG-Net (fine-tuned) | AR | 0.382 | 0.437 | 0.421 | 0.482 | 0.787 |
| CG-Net (re-trained on 358 images, official init.) | AR | 0.399 ± 0.005 | 0.456 ± 0.008 | 0.437 ± 0.007 | 0.499 ± 0.005 | 0.805 ± 0.006 |
| CG-Net (trained from scratch on 358 images) | AR | 0.394 ± 0.008 | 0.453 ± 0.023 | 0.427 ± 0.006 | 0.490 ± 0.008 | 0.806 ± 0.022 |
| Logistic regression, ViT block 1 | AR | 0.319 ± 0.001 | 0.396 ± 0.004 | 0.324 ± 0.003 | 0.402 ± 0.002 | 0.744 ± 0.021 |
| Logistic regression, ViT block 12 | AR | 0.385 ± 0.001 | 0.463 ± 0.002 | 0.390 ± 0.002 | 0.468 ± 0.002 | 0.785 ± 0.004 |
| Multi-scale fusion | AR | 0.403 ± 0.004 | 0.468 ± 0.009 | 0.411 ± 0.005 | 0.478 ± 0.011 | 0.818 ± 0.017 |
| Multi-scale fusion + token gate | AR | 0.399 ± 0.003 | 0.462 ± 0.009 | 0.407 ± 0.001 | 0.471 ± 0.007 | 0.824 ± 0.016 |
| Mask-prompt generator (segmentation loss) | AR | 0.413 ± 0.001 | 0.479 ± 0.002 | 0.423 ± 0.001 | 0.490 ± 0.002 | 0.817 ± 0.004 |
| Mask-prompt generator (end-to-end via SAM) | AR | 0.378 ± 0.019 | 0.451 ± 0.018 | 0.383 ± 0.020 | 0.456 ± 0.017 | 0.801 ± 0.053 |
| Mask-prompt generator (two-stage) | AR | 0.409 ± 0.002 | 0.485 ± 0.003 | 0.415 ± 0.003 | 0.491 ± 0.002 | 0.789 ± 0.013 |
| Mask-prompt generator (segmentation loss, label smoothing) | AR | 0.412 ± 0.001 | 0.483 ± 0.006 | 0.421 ± 0.003 | 0.493 ± 0.003 | 0.806 ± 0.008 |
| Mask-prompt generator + raw CG-Net fields | AR | 0.409 ± 0.006 | 0.478 ± 0.003 | 0.419 ± 0.010 | 0.489 ± 0.002 | 0.808 ± 0.025 |
