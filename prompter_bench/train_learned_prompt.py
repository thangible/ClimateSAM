"""
Learned static prompts: the most minimal prompt-free SAM.

Instead of a prompter that looks at the image, each class gets K learnable sparse prompt tokens (K x 256 values),
fed to the frozen prompt-encoder / HQ decoder together with the "no mask" dense embedding. The same tokens are used for
every image; all image-specific information has to come from the decoder's attention to the image embedding.
Trained through the frozen decoder with the Table 4.4 loss; one decoder call per class and image.
"""
import os
import csv
import json
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import RESULTS, CLASSES, SegMetrics, ObjectMetrics, FeatureCache, load_climatesam, upsample, seed_everything
from build_cache import ENCODERS
from train import Batches, class_loss


class StaticPrompts(nn.Module):
    def __init__(self, init, k):
        super().__init__()
        # initialised from SAM's learned "positive point" embedding plus noise, so the decoder starts in-distribution
        self.tokens = nn.Parameter(init[None, None].repeat(2, k, 1) + 0.02 * torch.randn(2, k, init.numel(), device=init.device))


def decode(climatesam, prompts, emb, interm0, cls_idx, cls):
    B = len(emb)
    pe = climatesam.prompt_encoder.get_dense_pe()
    dense = climatesam.prompt_encoder.sam_prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(1, -1, 64, 64)
    sparse = prompts.tokens[cls_idx][None]
    _, m = climatesam.mask_decoder(type=cls, image_embeddings=emb, image_pe=[pe] * B, sparse_prompt_embeddings=[sparse] * B,
                                   dense_prompt_embeddings=[dense] * B, multimask_output=False, interm_embeddings=[interm0])
    return torch.cat(m)[:, 0]  # (B, 256, 256)


@torch.no_grad()
def evaluate(climatesam, prompts, data, device, objects=False):
    m, o = SegMetrics(), ObjectMetrics()
    for s in range(0, len(data), 4):
        batch = data.get(np.arange(s, min(s + 4, len(data))))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = {cls: decode(climatesam, prompts, batch['emb'], batch['vit'][0], i, cls) for i, (cls, _) in enumerate(CLASSES)}
        for b in range(len(batch['gt'])):
            pred = {cls: upsample(out[cls][b].float()) > 0 for cls, _ in CLASSES}
            m.update(pred['TC'], pred['AR'], batch['gt'][b])
            if objects:
                for cls, label in CLASSES:
                    o.update(pred[cls], batch['gt'][b] == label, cls)
    return {**m.compute(), **(o.compute() if objects else {})}, m.per_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoder', default='infused_mlp1')
    ap.add_argument('--k', type=int, default=4, help='tokens per class')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--lr', type=float, default=1e-3)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device('cuda')
    ckpt, mlp = ENCODERS[args.encoder]
    climatesam = load_climatesam(ckpt, mlp, device)
    init = climatesam.prompt_encoder.sam_prompt_encoder.point_embeddings[1].weight[0].detach()
    prompts = StaticPrompts(init, args.k).to(device)
    name = f'learned_prompt_k{args.k}_s{args.seed}'
    out_dir = os.path.join(RESULTS, 'runs', args.encoder, name)
    os.makedirs(out_dir, exist_ok=True)

    train = Batches(FeatureCache(args.encoder, 'train'), [0], device)
    val = Batches(FeatureCache(args.encoder, 'val'), [0], device)
    test = Batches(FeatureCache(args.encoder, 'test'), [0], device)
    opt = torch.optim.AdamW(prompts.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)

    rows, best, best_tokens, t0 = [], None, None, time.time()
    for epoch in range(1, args.epochs + 1):
        perm, total, steps = np.random.permutation(len(train)), 0.0, 0
        for s in range(0, len(perm), 8):
            batch = train.get(perm[s:s + 8])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = torch.stack([decode(climatesam, prompts, batch['emb'], batch['vit'][0], i, cls)
                                      for i, (cls, _) in enumerate(CLASSES)], dim=1)
            up = F.interpolate(logits.float(), size=batch['gt'].shape[-2:], mode='bilinear', align_corners=False)
            loss = sum(class_loss(up[:, i], (batch['gt'] == label).float(), cls) for i, (cls, label) in enumerate(CLASSES))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total, steps = total + float(loss.detach()), steps + 1
        sched.step()
        row = {'epoch': epoch, 'train_loss': total / steps}
        if epoch % 2 == 0 or epoch == args.epochs:
            row.update({f'val_{k}': v for k, v in evaluate(climatesam, prompts, val, device)[0].items()})
            if best is None or row['val_Mean FG IoU'] > best['val_Mean FG IoU']:
                best, best_tokens = dict(row), prompts.tokens.detach().clone()
        rows.append(row)
        print(f"[{name}] epoch {epoch} loss {row['train_loss']:.4f}" +
              (f" val FG {row['val_Mean FG IoU']:.4f}" if 'val_Mean FG IoU' in row else ''), flush=True)
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(os.path.join(out_dir, 'log.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    prompts.tokens.data.copy_(best_tokens)
    torch.save({'tokens': best_tokens.cpu()}, os.path.join(out_dir, 'best.pth'))
    test_metrics, per_image = evaluate(climatesam, prompts, test, device, objects=True)
    summary = {'name': name, 'k': args.k, 'seed': args.seed, 'trainable_params': prompts.tokens.numel(),
               'best_epoch': best['epoch'], 'train_time_min': (time.time() - t0) / 60,
               **{k: v for k, v in best.items() if k.startswith('val_')}, 'test': test_metrics}
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, 'test_per_image_counts.json'), 'w') as f:
        json.dump(per_image, f)
    print(json.dumps(summary['test'], indent=2))


if __name__ == '__main__':
    main()
