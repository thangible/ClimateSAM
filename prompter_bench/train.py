"""
Unified trainer for the SAM-feature prompters (Section 4.3.3).

Identical protocol for every architecture:
  frozen ClimateSAM (cached features), 358 train / 40 validation images, 60 epochs, AdamW + cosine,
  loss = Tversky + Focal with the parameters selected in Section 4.1 (Table 4.4),
  checkpoint selected on validation mean foreground IoU, test set untouched.

--mode seg : the prompter is trained as a segmenter (loss on its own TC / AR maps)
--mode sam : the prompter's maps are turned into SAM prompts in a differentiable way and the loss is also put
             on SAM's output (AR: dense mask prompt; TC: one box per blob + the TC logit map as mask prompt)
"""
import os
import csv
import json
import time
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from common import (RESULTS, CLASSES, EMPTY_LOGIT, LOWRES, FeatureCache, SegMetrics, load_climatesam,
                    seed_everything, upsample, fold_of)
from build_cache import ENCODERS
from loss_function import calculate_tversky_loss, calculate_focal_loss
from model.prompt.cgnet import jaccard_loss
import prompters

WANDB_PROJECT = 'climatesam-section-4.3'


def wandb_init(args, name, encoder, kind):
    """Offline wandb run (no API key on this machine): upload later with `wandb sync wandb/offline-run-*`."""
    if not getattr(args, 'wandb', False):
        return None
    import wandb
    os.environ.setdefault('WANDB_MODE', 'offline')
    return wandb.init(project=WANDB_PROJECT, name=f'{encoder}/{name}', group=kind, config=vars(args), dir=RESULTS.rsplit('/results', 1)[0],
                      reinit=True, mode='offline')

LOSS = {  # Table 4.4
    'TC': dict(t_alpha=0.3, t_beta=0.7, f_alpha=0.95, f_gamma=5.0),
    'AR': dict(t_alpha=0.5, t_beta=0.5, f_alpha=0.85, f_gamma=5.0),
}
TC_MIN_AREA, TC_MAX_BOXES = 2, 16  # TC blobs used as box prompts, in 256x256 cells
SMOOTH = {'TC': (3, 5.0), 'AR': (9, 20.0)}  # Gaussian label smoothing (kernel, sigma), Table 4.5 optimum
USE_SMOOTH = False


# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------
class Batches:
    """Serves cached features. Few layers: everything on the GPU. Many layers: read from the memmap per batch."""

    def __init__(self, cache, layers, device, with_cgnet=False):
        self.cache, self.layers, self.device = cache, layers, device
        self.eager = len(layers) <= 3
        if self.eager:
            self.data = cache.load_to(device, layers, with_cgnet=with_cgnet)
        else:
            self.data = cache.load_to(device, [], with_cgnet=with_cgnet)

    def __len__(self):
        return len(self.cache)

    def get(self, pos):
        pos = np.sort(np.asarray(pos))
        out = {'emb': self.data['emb'][pos].float(), 'gt': self.data['gt'][pos].long()}
        if 'cgnet' in self.data:
            out['cgnet'] = self.data['cgnet'][pos]
        if self.eager:
            out['vit'] = {l: self.data['vit'][l][pos].float() for l in self.layers}
        else:
            arr = torch.from_numpy(np.ascontiguousarray(self.cache.vit[self.cache.index[pos]])).pin_memory()
            arr = arr.to(self.device, non_blocking=True).float()
            out['vit'] = {l: arr[:, l] for l in self.layers}
        return out


# ------------------------------------------------------------
# LOSS
# ------------------------------------------------------------
def gaussian_smooth(target, kernel_size, sigma):
    """Same kernel as ClimateLoss._smooth_label_tensor. target: (B, H, W) in {0, 1}."""
    half = kernel_size // 2
    ax = torch.arange(-half, half + 1, dtype=torch.float32, device=target.device)
    kern = torch.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / (2 * sigma ** 2))
    kern = (kern / kern.sum())[None, None]
    return F.conv2d(target[:, None], kern, padding=half)[:, 0]


def class_loss(logits, target, cls):
    if USE_SMOOTH:
        target = gaussian_smooth(target, *SMOOTH[cls])
    p = LOSS[cls]
    return calculate_tversky_loss(logits, target, alpha=p['t_alpha'], beta=p['t_beta']) \
        + calculate_focal_loss(logits, target, gamma=p['f_gamma'], alpha=p['f_alpha'])


def seg_loss(logits, gt):
    """logits: (B, 2, h, w) TC / AR, gt: (B, H, W). Loss at the label resolution."""
    if logits.shape[-2:] != gt.shape[-2:]:
        logits = F.interpolate(logits.float(), size=gt.shape[-2:], mode='bilinear', align_corners=False)
    return {cls: class_loss(logits[:, ch].float(), (gt == label).float(), cls) for ch, (cls, label) in enumerate(CLASSES)}


def aux_loss(aux, gt):
    """Deep supervision (thesis Eq. 3.4): coarse outputs against the nearest-downsampled labels, weights 0.4/(i+1)."""
    total = 0.0
    for i, a in enumerate(aux):
        small = F.interpolate(gt[:, None].float(), size=a.shape[-2:], mode='nearest')[:, 0]
        weight = 1.0 if a.shape[-2:] == gt.shape[-2:] else 0.4 / (i + 1)
        total = total + weight * sum(seg_loss(a, small).values())
    return total


# ------------------------------------------------------------
# DIFFERENTIABLE SAM PROMPTING (mode sam, and validation of it)
# ------------------------------------------------------------
def tc_boxes(tc_logit_256):
    """TC blobs of a (256, 256) logit map -> (N, 4) boxes in the 1024 frame (largest TC_MAX_BOXES blobs)."""
    n, _, stats, _ = cv2.connectedComponentsWithStats((tc_logit_256 > 0).cpu().numpy().astype(np.uint8), connectivity=8)
    stats = sorted(stats[1:], key=lambda s: -s[4])[:TC_MAX_BOXES]
    boxes = [[x, y, x + w, y + h] for x, y, w, h, a in stats if a >= TC_MIN_AREA]
    return torch.tensor(boxes, dtype=torch.float32, device=tc_logit_256.device).reshape(-1, 4) * 4


def sam_hybrid(climatesam, emb, interm0, logits):
    """
    logits: (B, 2, h, w) prompter output -> SAM HQ logits {TC, AR}: (B, 256, 256).
    AR: the AR logit map is the dense prompt. TC: one box per TC blob, each with the TC logit map as dense prompt.
    """
    lo = F.interpolate(logits.float(), (LOWRES, LOWRES), mode='bilinear', align_corners=False).clamp(-20, 20)
    B = len(emb)
    pe = climatesam.prompt_encoder.get_dense_pe()

    sparse, dense = zip(*[climatesam.prompt_encoder(points=None, boxes=None, masks=lo[b:b + 1, 1:2]) for b in range(B)])
    _, ar = climatesam.mask_decoder(type='AR', image_embeddings=emb, image_pe=[pe] * B, sparse_prompt_embeddings=list(sparse),
                                    dense_prompt_embeddings=list(dense), multimask_output=False, interm_embeddings=[interm0])
    out = {'AR': torch.cat(ar)[:, 0]}

    boxes = [tc_boxes(lo[b, 0].detach()) for b in range(B)]
    keep = [b for b in range(B) if len(boxes[b])]
    tc = {}
    if keep:
        sparse, dense = zip(*[climatesam.prompt_encoder(points=None, boxes=boxes[b],
                                                        masks=lo[b:b + 1, 0:1].expand(len(boxes[b]), -1, -1, -1)) for b in keep])
        _, m = climatesam.mask_decoder(type='TC', image_embeddings=emb[keep], image_pe=[pe] * len(keep),
                                       sparse_prompt_embeddings=list(sparse), dense_prompt_embeddings=list(dense),
                                       multimask_output=False, interm_embeddings=[interm0[keep]])
        tc = {b: mm[:, 0].max(dim=0).values for b, mm in zip(keep, m)}
    empty = torch.full((LOWRES, LOWRES), EMPTY_LOGIT, device=emb.device)
    out['TC'] = torch.stack([tc.get(b, empty).float() for b in range(B)])
    return out


# ------------------------------------------------------------
# TRAIN / VALIDATE
# ------------------------------------------------------------
def forward(model, climatesam, batch, mode):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out = model(batch)
        if mode == 'sam':
            out['sam'] = sam_hybrid(climatesam, batch['emb'], batch['vit'][0], out['logits'])
    return out


def train_one_epoch(model, climatesam, data, optimizer, args):
    model.train()
    perm = np.random.permutation(len(data))
    sums, steps = {}, 0
    for s in range(0, len(perm), args.bs):
        optimizer.zero_grad(set_to_none=True)
        chunk = perm[s:s + args.bs]
        micro = [chunk[m:m + args.micro_bs] for m in range(0, len(chunk), args.micro_bs)]
        for mb in micro:  # gradient accumulation: the effective batch is always args.bs
            batch = data.get(mb)
            out = forward(model, climatesam, batch, args.mode)
            if 'raw' in out:  # CG-Net: 3-class softmax trained with the Jaccard loss (thesis Section 3.2.3)
                total = jaccard_loss(out['raw'].float(), batch['gt'])
                (total * len(mb) / len(chunk)).backward()
                sums['total'] = sums.get('total', 0.0) + float(total) * len(mb) / len(chunk)
                continue
            gen = seg_loss(out['logits'], batch['gt'])
            losses = {f'gen_{k}': v for k, v in gen.items()}
            if out['aux']:
                losses['deep_sup'] = aux_loss(out['aux'], batch['gt'])
            total = args.aux_weight * (sum(gen.values()) + losses.get('deep_sup', 0.0))
            if args.mode == 'sam':
                sam = seg_loss(torch.stack([out['sam']['TC'], out['sam']['AR']], dim=1), batch['gt'])
                losses.update({f'sam_{k}': v for k, v in sam.items()})
                total = total + sum(sam.values())
            losses['total'] = total
            (total * len(mb) / len(chunk)).backward()
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + float(v) * len(mb) / len(chunk)
        optimizer.step()
        steps += 1
    return {f'train_{k}': v / steps for k, v in sums.items()}


@torch.no_grad()
def validate(model, climatesam, data, mode, bs=2):
    model.eval()
    own, sam = SegMetrics(), SegMetrics()
    for s in range(0, len(data), bs):
        batch = data.get(np.arange(s, min(s + bs, len(data))))
        out = forward(model, climatesam, batch, 'sam' if mode == 'sam' else 'seg')
        up = F.interpolate(out['logits'].float(), size=batch['gt'].shape[-2:], mode='bilinear', align_corners=False) > 0
        if 'argmax' in out:  # CG-Net predicts with its 3-class argmax
            up = torch.stack([out['argmax'] == 1, out['argmax'] == 2], dim=1)
        for b in range(len(batch['gt'])):
            own.update(up[b, 0], up[b, 1], batch['gt'][b])
            if mode == 'sam':
                sam.update(upsample(out['sam']['TC'][b]) > 0, upsample(out['sam']['AR'][b]) > 0, batch['gt'][b])
    res = {f'val_own_{k}': v for k, v in own.compute().items()}
    if mode == 'sam':
        res.update({f'val_sam_{k}': v for k, v in sam.compute().items()})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoder', default='infused_mlp1')
    ap.add_argument('--arch', required=True)
    ap.add_argument('--mode', default='seg', choices=['seg', 'sam'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--micro_bs', type=int, default=8, help='micro-batch for gradient accumulation (memory only)')
    ap.add_argument('--aux_weight', type=float, default=1.0, help='weight of the loss on the prompter\'s own maps')
    ap.add_argument('--init', default=None, help='checkpoint to start from (two-stage training)')
    ap.add_argument('--val_every', type=int, default=2)
    ap.add_argument('--name', default=None)
    ap.add_argument('--smooth', action='store_true', help='Gaussian label smoothing of the targets (Section 3.5)')
    ap.add_argument('--exclude_fold', type=int, default=None, help='leave this fold (of 5) out of the train split')
    ap.add_argument('--wandb', action='store_true', help='log to wandb (offline mode)')
    args = ap.parse_args()
    global USE_SMOOTH
    USE_SMOOTH = args.smooth

    name = args.name or f'{args.arch}_{args.mode}_s{args.seed}'
    out_dir = os.path.join(RESULTS, 'runs', args.encoder, name)
    os.makedirs(out_dir, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device('cuda')

    ckpt, mlp = ENCODERS[args.encoder]
    climatesam = load_climatesam(ckpt, mlp, device)
    model, layers = prompters.build(args.arch, climatesam=climatesam)
    model = model.to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device)['state_dict'])
    if args.mode == 'sam':
        layers = sorted(set(layers) | {0})  # block 0 feeds the HQ decoder
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    train_cache = FeatureCache(args.encoder, 'train')
    if args.exclude_fold is not None:  # out-of-fold models for the decoder-adaptation experiment
        keep = fold_of(len(train_cache)) != args.exclude_fold
        train_cache.index, train_cache.names = train_cache.index[keep], [n for n, k in zip(train_cache.names, keep) if k]
    with_cgnet = prompters.needs_fields(args.arch)
    train = Batches(train_cache, layers, device, with_cgnet)
    val = Batches(FeatureCache(args.encoder, 'val'), layers, device, with_cgnet)
    run = wandb_init(args, name, args.encoder, args.arch)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    select = 'val_sam_Mean FG IoU' if args.mode == 'sam' else 'val_own_Mean FG IoU'

    log_path = os.path.join(out_dir, 'log.csv')
    best, rows, t_start = None, [], time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        row = {'epoch': epoch, 'lr': scheduler.get_last_lr()[0]}
        row.update(train_one_epoch(model, climatesam, train, optimizer, args))
        scheduler.step()
        row['epoch_time_s'] = time.time() - t0
        if epoch % args.val_every == 0 or epoch == args.epochs:
            row.update(validate(model, climatesam, val, args.mode))
            if best is None or row[select] > best[select]:
                best = dict(row)
                torch.save({'state_dict': model.state_dict(), 'arch': args.arch, 'mode': args.mode, 'encoder': args.encoder,
                            'epoch': epoch, 'val': best}, os.path.join(out_dir, 'best.pth'))
        rows.append(row)
        if run:
            run.log(row, step=epoch)
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k != 'epoch', k))
        with open(log_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"[{name}] epoch {epoch} {row['epoch_time_s']:.1f}s loss {row['train_total']:.4f}"
              + (f" | {select} {row[select]:.4f}" if select in row else ''), flush=True)

    summary = {'name': name, 'arch': args.arch, 'mode': args.mode, 'seed': args.seed, 'encoder': args.encoder,
               'trainable_params': n_params, 'best_epoch': best['epoch'], 'train_time_min': (time.time() - t_start) / 60,
               'args': vars(args), **{k: v for k, v in best.items() if k.startswith('val_')}}
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    if run:
        run.summary.update({k: v for k, v in summary.items() if not isinstance(v, dict)})
        run.finish()
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
