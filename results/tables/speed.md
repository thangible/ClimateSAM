| Model | Parameters | GFLOPs / image | Latency (ms, A100, batch 1) |
|---|---|---|---|
| CG-Net | 494,232 | 22.93 | 8.89 |
| Logistic regression | 1,538 | 0.01 | 0.05 |
| Multi-scale fusion | 1,406,474 | 702.36 | 36.13 |
| Multi-scale fusion + token gate | 1,415,180 | 702.89 | 38.12 |
| Mask-prompt generator | 671,138 | 5.74 | 0.89 |
| (frozen ClimateSAM encoder, for reference) | 95,987,456 | 973.25 | 122.39 |
| (one SAM decoder call, one box, for reference) | 5,269,508 | 16.80 | 3.39 |
