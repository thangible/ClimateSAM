"""Parameters, forward FLOPs and latency of every prompter (batch 1). Run on an otherwise idle GPU."""
import os
import csv

import torch
from torch.utils.flop_counter import FlopCounterMode

from common import RESULTS, load_climatesam
from build_cache import ENCODERS
import prompters


def measure(fn, reps=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return sorted(times)[len(times) // 2]


@torch.no_grad()
def main():
    device = torch.device('cuda')
    ckpt, mlp = ENCODERS['infused_mlp1']
    sam = load_climatesam(ckpt, mlp, device)
    batch = {'emb': torch.randn(1, 256, 64, 64, device=device),
             'vit': {l: torch.randn(1, 64, 64, 768, device=device) for l in range(12)},
             'cgnet': torch.randn(1, 4, 768, 1152, device=device)}
    rows = []
    for arch, label in (('cgnet_finetuned', 'CG-Net'), ('logreg_l0', 'Logistic regression'), ('msf', 'Multi-scale fusion'),
                        ('msf_token', 'Multi-scale fusion + token gate'), ('mpg', 'Mask-prompt generator')):
        model = prompters.build(arch, climatesam=sam)[0].to(device).eval()
        fwd = lambda: model(batch)
        with FlopCounterMode(display=False) as fc:
            fwd()
        rows.append({'prompter': label, 'params': sum(p.numel() for p in model.parameters()),
                     'GFLOPs': fc.get_total_flops() / 1e9, 'latency_ms': measure(fwd)})
        print(rows[-1], flush=True)

    # context: the frozen parts every SAM-feature prompter already pays for
    x = torch.randn(1, 16, 768, 1152, device=device)
    with FlopCounterMode(display=False) as fc:
        sam.encode_images(x)
    rows.append({'prompter': '(frozen ClimateSAM encoder, for reference)', 'params': sum(p.numel() for p in sam.image_encoder.parameters()),
                 'GFLOPs': fc.get_total_flops() / 1e9, 'latency_ms': measure(lambda: sam.encode_images(x))})
    emb, feats, _, _ = sam.encode_images(x)
    box = torch.tensor([[100., 100., 300., 300.]], device=device)
    dec = lambda: sam.mask_decoder(type='AR', image_embeddings=emb, image_pe=[sam.prompt_encoder.get_dense_pe()],
                                   sparse_prompt_embeddings=[sam.prompt_encoder(None, box, None)[0]],
                                   dense_prompt_embeddings=[sam.prompt_encoder(None, box, None)[1]],
                                   multimask_output=False, interm_embeddings=feats)
    with FlopCounterMode(display=False) as fc:
        dec()
    rows.append({'prompter': '(one SAM decoder call, one box, for reference)', 'params': sum(p.numel() for p in sam.mask_decoder.parameters()),
                 'GFLOPs': fc.get_total_flops() / 1e9, 'latency_ms': measure(dec)})
    for r in rows[-2:]:
        print(r)

    out = os.path.join(RESULTS, '04_sam_feature_prompters')
    with open(os.path.join(out, 'speed.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()
