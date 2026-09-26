"""
Prompt-robust decoder (PLAN_20h B3).

The Phase-1 decoder was trained with perfect prompts and reproduces its prompts (REPORT §3.1). Here the decoder's
trainable HQ parts are retrained with *imperfect* prompts and one target per prompt, so that it can learn to reject a
prompt and to correct its extent:

  prompt source (per image): with p_real the out-of-fold mask-prompt generator (realistic errors), otherwise corrupted
  ground truth -- boxes with every side moved by U(-15 %, +30 %) of the object size, objects dropped with p=0.15,
  0-3 false boxes of realistic size; dense masks eroded / dilated by up to 5 px, objects dropped, false blobs added.
  targets: a box prompt -> the full ground-truth objects of its class that it touches (empty for a false box);
           a class-level dense prompt (AR in hybrid mode) -> the full class ground truth.
  mode 'box': boxes for both classes; mode 'hybrid': TC boxes + TC dense map, AR dense map only.

Same trainable parameters as train_decoder.py (1.21 M; hf_token_* stay frozen because the encoder shares them).
"""
import os
import csv
import copy
import json
import time
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from common import (RESULTS, CLASSES, SegMetrics, FeatureCache, load_climatesam, sam_decode, union_logits, upsample,
                    make_prompts, binary_to_score, score_to_prompt, to_sam_frame, seed_everything, fold_of, MIN_OBJECT_PX,
                    IMG_H, IMG_W, LOWRES)
from build_cache import ENCODERS
from train import Batches, class_loss, wandb_init
from train_decoder import TRAINABLE
from evaluate import kind_for
import prompters

DEFAULT_SIZE = {'TC': (45, 40), 'AR': (180, 70)}  # (w, h) of a false box when the class has no object in the image


def components(mask):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = [k for k in range(1, n) if st[k, 4] >= MIN_OBJECT_PX]
    boxes = [[st[k, 0], st[k, 1], st[k, 0] + st[k, 2] - 1, st[k, 1] + st[k, 3] - 1] for k in keep]
    return lab, keep, boxes


MAX_TRAIN_BOXES = 12


def box_targets(boxes, gt_lab):
    """(N, 4) boxes -> (N, H, W) targets: union of the ground-truth components each box touches."""
    t = np.zeros((len(boxes), IMG_H, IMG_W), dtype=np.float32)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        x1, y1 = int(max(0, np.floor(x1))), int(max(0, np.floor(y1)))
        x2, y2 = int(min(IMG_W - 1, np.ceil(x2))), int(min(IMG_H - 1, np.ceil(y2)))
        ids = np.unique(gt_lab[y1:y2 + 1, x1:x2 + 1])
        ids = ids[ids > 0]
        if len(ids):
            t[i] = np.isin(gt_lab, ids)
    return t


def corrupt_boxes(gt_boxes, cls, rng):
    out = []
    for x1, y1, x2, y2 in gt_boxes:
        if rng.random() < 0.15:
            continue
        w, h = x2 - x1 + 1, y2 - y1 + 1
        d = rng.uniform(-0.15, 0.3, size=4)
        out.append([x1 - d[0] * w, y1 - d[1] * h, x2 + d[2] * w, y2 + d[3] * h])
    for _ in range(rng.binomial(3, 0.25)):
        if gt_boxes:
            rx1, ry1, rx2, ry2 = gt_boxes[rng.integers(len(gt_boxes))]
            w, h = (rx2 - rx1 + 1), (ry2 - ry1 + 1)
        else:
            w, h = DEFAULT_SIZE[cls]
        w, h = w * rng.uniform(0.7, 1.3), h * rng.uniform(0.7, 1.3)
        x, y = rng.uniform(0, IMG_W - w), rng.uniform(0, IMG_H - h)
        out.append([x, y, x + w, y + h])
    out = [[max(0, a), max(0, b), min(IMG_W - 1, c), min(IMG_H - 1, d)] for a, b, c, d in out]
    return [b for b in out if b[2] > b[0] + 1 and b[3] > b[1] + 1]


def corrupt_mask(gt_mask, rng):
    lab, keep, boxes = components(gt_mask)
    m = np.isin(lab, [k for k in keep if rng.random() >= 0.15]).astype(np.uint8)
    k = int(rng.integers(-5, 6))
    if k:
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (abs(k), abs(k)))
        m = cv2.dilate(m, kern) if k > 0 else cv2.erode(m, kern)
    if keep and rng.random() < 0.3:  # a false blob: a real object shifted somewhere else
        obj = (lab == keep[rng.integers(len(keep))]).astype(np.uint8)
        m = np.maximum(m, np.roll(obj, (int(rng.integers(-150, 150)), int(rng.integers(0, IMG_W))), axis=(0, 1)))
    return m


def make_training_prompts(gt, gt_comps, oof_logits, mode, rng, device):
    """-> {cls: (prompt kwargs, targets (N, H, W) tensor)} for one image. gt_comps: {cls: components(gt == label)}."""
    use_real = oof_logits is not None and rng.random() < 0.5
    out = {}
    for ch, (cls, label) in enumerate(CLASSES):
        g = gt == label
        gt_lab, gt_keep, gt_boxes = gt_comps[cls]
        if use_real:
            logit = oof_logits[ch].float()
            m = (F.interpolate(logit[None, None], (IMG_H, IMG_W), mode='bilinear', align_corners=False)[0, 0] > 0).cpu().numpy()
            boxes = components(m)[2]
            dense = score_to_prompt(logit)
        else:
            boxes = corrupt_boxes(gt_boxes, cls, rng)
            dense = score_to_prompt(binary_to_score(torch.from_numpy(corrupt_mask(g, rng)).to(device)))
        if mode == 'hybrid' and cls == 'AR':
            kw = {'masks': dense}
            target = torch.from_numpy(g.astype(np.float32))[None].to(device)
        else:
            if not boxes:
                continue
            if len(boxes) > MAX_TRAIN_BOXES:  # memory: every box has a full-resolution target
                boxes = [boxes[i] for i in rng.choice(len(boxes), MAX_TRAIN_BOXES, replace=False)]
            b = to_sam_frame(torch.tensor(boxes, dtype=torch.float32, device=device))
            kw = {'boxes': b}
            if mode == 'hybrid':
                kw['masks'] = dense.expand(len(boxes), -1, -1, -1)
            target = torch.from_numpy(box_targets(boxes, gt_lab)).to(device)
        out[cls] = (kw, target)
    return out


@torch.no_grad()
def predict_logits(model, data, device, hard=False):
    """Prompter logits for every image of a split, at 256x256 (float16, on the GPU)."""
    out = []
    for s in range(0, len(data), 4):
        batch = data.get(np.arange(s, min(s + 4, len(data))))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            o = model(batch)
        out.append(F.interpolate(o['logits'].float(), (LOWRES, LOWRES), mode='bilinear', align_corners=False).half())
    return torch.cat(out)


def load_prompter(name, encoder, climatesam, device):
    ck = torch.load(os.path.join(RESULTS, 'runs', encoder, name, 'best.pth'), map_location=device)
    model, layers = prompters.build(ck['arch'], climatesam=climatesam)
    model.load_state_dict(ck['state_dict'])
    return model.to(device).eval(), layers


@torch.no_grad()
def evaluate_split(climatesam, data, logits, mode, device):
    """SAM output with prompts made from prompter logits (the benchmark conversion), dataset-level IoU."""
    climatesam.mask_decoder.eval()
    m = SegMetrics()
    for i in range(len(data)):
        gt = data.data['gt'][i].long()
        emb, interm0 = data.data['emb'][i:i + 1].float(), data.data['vit'][0][i:i + 1].float()
        full = F.interpolate(logits[i:i + 1].float(), (IMG_H, IMG_W), mode='bilinear', align_corners=False)[0]
        preds = {}
        for ch, (cls, _) in enumerate(CLASSES):
            mk = (full[ch] > 0).cpu().numpy().astype(np.uint8)
            p = make_prompts(mk, kind_for('bbox' if mode == 'box' else 'hybrid', cls), score=full[ch], device=device)
            lg = sam_decode(climatesam, emb, interm0, cls, **p) if p else None
            preds[cls] = upsample(union_logits(lg).to(device)) > 0
        m.update(preds['TC'], preds['AR'], gt)
    return m.compute()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoder', default='infused_mlp1')
    ap.add_argument('--mode', default='box', choices=['box', 'hybrid'])
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no_real', action='store_true', help='only corrupted ground-truth prompts (no out-of-fold prompts)')
    ap.add_argument('--wandb', action='store_true')
    args = ap.parse_args()

    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device('cuda')
    ckpt, mlp = ENCODERS[args.encoder]
    climatesam = load_climatesam(ckpt, mlp, device)
    name = f'robust_decoder_{args.mode}{"_gtonly" if args.no_real else ""}_s{args.seed}'
    out_dir = os.path.join(RESULTS, 'runs', args.encoder, name)
    os.makedirs(out_dir, exist_ok=True)
    run = wandb_init(args, name, args.encoder, 'robust_decoder')

    train_cache, val_cache = FeatureCache(args.encoder, 'train'), FeatureCache(args.encoder, 'val')
    # out-of-fold generator logits for the training images, generator seed 0 for the validation images
    oof = None
    if not args.no_real:
        folds = fold_of(len(train_cache))
        oof = torch.zeros(len(train_cache), 2, LOWRES, LOWRES, dtype=torch.float16, device=device)
        for k in range(5):
            model_k, layers = load_prompter(f'mpg_seg_fold{k}', args.encoder, climatesam, device)
            sub = copy.copy(train_cache)
            sub.index = train_cache.index[folds == k]
            oof[torch.from_numpy(np.where(folds == k)[0]).to(device)] = predict_logits(model_k, Batches(sub, layers, device), device)
    prompter, layers = load_prompter('mpg_seg_s0', args.encoder, climatesam, device)
    val_logits = predict_logits(prompter, Batches(val_cache, layers, device), device)
    del prompter
    train, val = Batches(train_cache, [0], device), Batches(val_cache, [0], device)
    gt_np = train.data['gt'].cpu().numpy()
    gt_comps = [{cls: components(gt_np[i] == label) for cls, label in CLASSES} for i in range(len(train))]

    dec = climatesam.mask_decoder
    params = []
    for n, p in dec.named_parameters():
        p.requires_grad = n.split('.')[0] in TRAINABLE
        if p.requires_grad:
            params.append(p)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)

    rows = [{'epoch': 0, **{f'val_{k}': v for k, v in evaluate_split(climatesam, val, val_logits, args.mode, device).items()}}]
    best, best_state = dict(rows[0]), {k: v.clone() for k, v in dec.state_dict().items()}
    print(f"[{name}] epoch 0 (Phase-1 decoder) val FG {rows[0]['val_Mean FG IoU']:.4f}", flush=True)
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        dec.train()
        for mod_name, mod in dec.named_children():
            if mod_name not in TRAINABLE:
                mod.eval()
        order = rng.permutation(len(train))
        total, steps = 0.0, 0
        for s in range(0, len(order), 8):
            opt.zero_grad(set_to_none=True)
            loss, n_terms = 0.0, 0
            for i in order[s:s + 8]:
                emb, interm0 = train.data['emb'][i:i + 1].float(), train.data['vit'][0][i:i + 1].float()
                prompts = make_training_prompts(gt_np[i], gt_comps[i], oof[i] if oof is not None else None, args.mode, rng, device)
                for cls, (kw, target) in prompts.items():
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        lg = sam_decode(climatesam, emb, interm0, cls, **kw)
                    up = F.interpolate(lg[:, None].float(), (IMG_H, IMG_W), mode='bilinear', align_corners=False)[:, 0]
                    loss = loss + class_loss(up, target, cls)
                    n_terms += 1
            if n_terms:
                (loss / n_terms).backward()
                opt.step()
                total += float(loss.detach()) / n_terms
                steps += 1
        sched.step()
        row = {'epoch': epoch, 'train_loss': total / max(steps, 1), 'lr': sched.get_last_lr()[0],
               **{f'val_{k}': v for k, v in evaluate_split(climatesam, val, val_logits, args.mode, device).items()}}
        rows.append(row)
        if row['val_Mean FG IoU'] > best['val_Mean FG IoU']:
            best, best_state = dict(row), {k: v.clone() for k, v in dec.state_dict().items()}
        if run:
            run.log(row, step=epoch)
        print(f"[{name}] epoch {epoch} loss {row['train_loss']:.4f} val FG {row['val_Mean FG IoU']:.4f} "
              f"(TC {row['val_TC IoU']:.3f} AR {row['val_AR IoU']:.3f})", flush=True)
        with open(os.path.join(out_dir, 'log.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[-1].keys()))
            w.writeheader()
            w.writerows([{k: r.get(k, '') for k in rows[-1]} for r in rows])

    torch.save({k: v for k, v in best_state.items() if k.split('.')[0] in TRAINABLE}, os.path.join(out_dir, 'best_decoder.pth'))
    # the last epoch too: the selected one can be epoch 0 (= Phase-1 decoder), which says nothing about robustness
    torch.save({k: v for k, v in dec.state_dict().items() if k.split('.')[0] in TRAINABLE}, os.path.join(out_dir, 'last_decoder.pth'))
    summary = {'name': name, 'mode': args.mode, 'seed': args.seed, 'real_prompts': not args.no_real,
               'trainable_params': sum(p.numel() for p in params), 'best_epoch': best['epoch'],
               'train_time_min': (time.time() - t0) / 60, **{k: v for k, v in best.items() if k.startswith('val_')}}
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    if run:
        run.summary.update(summary)
        run.finish()
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
