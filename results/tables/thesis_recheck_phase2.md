| Prompter checkpoint (run) | Prompts | Logged mask (best) | Re-run mask | Logged SAM (best) | Re-run SAM | Same arch. retrained |
|---|---|---|---|---|---|---|
| MSF (Generator) | point | -- / -- | 0.251 / 0.310 | -- / -- | 0.158 / 0.223 | 0.333 / 0.403 |
| MSF (generator_128_vit_b_bbox) | box | 0.305 / 0.402 | 0.304 / 0.401 | 0.308 / 0.389 | 0.294 / 0.383 | 0.333 / 0.403 |
| MSF + token gate (FIRST) | point | 0.321 / 0.056 | 0.256 / 0.056 | 0.304 / 0.291 | 0.295 / 0.074 | 0.324 / 0.399 |
| token gate + CG blocks (CG) | point | 0.349 / 0.407 | 0.157 / 0.374 | 0.329 / 0.346 | 0.274 / 0.291 | 0.324 / 0.355 |
| token gate + CG blocks (CG_256) | box | 0.315 / 0.375 | 0.144 / 0.360 | 0.318 / 0.331 | 0.175 / 0.373 | 0.324 / 0.355 |
| token gate + CG blocks (CG_256_REAL) | box | 0.327 / 0.387 | 0.118 / 0.352 | 0.322 / 0.332 | 0.141 / 0.371 | 0.324 / 0.355 |
| token gate, shared weights (SP_TOKEN) | box | 0.326 / 0.393 | 0.205 / 0.376 | 0.333 / 0.306 | 0.190 / 0.361 | 0.332 / 0.402 |
| logistic regression, block 1 (Logistic) | box | 0.191 / 0.295 | 0.190 / 0.287 | 0.183 / 0.255 | 0.194 / 0.298 | 0.195 / 0.319 |
