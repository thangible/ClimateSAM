"""
SAM-feature YOLO box head (thesis train_script/official/train_det_head.py) under the benchmark protocol (PLAN_20h B2):
the token-gated detection head on the frozen image embedding, YOLO targets from the ground-truth boxes (8-connected
components >= 20 px), 60 epochs, AdamW + cosine, selection on the 40 validation images (SAM IoU with its boxes);
the objectness threshold is chosen on validation among {0.3, 0.5, 0.7}. Test output in the evaluate.py format.
"""
import os
import sys
import csv
import json
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from common import (ROOT, RESULTS, CLASSES, SegMetrics, ObjectMetrics, FeatureCache, load_climatesam, sam_decode,
                    union_logits, upsample, to_sam_frame, seed_everything, IMG_H, IMG_W)
from build_cache import ENCODERS
from train import Batches
from train_robust_decoder import components
import prompters
sys.path.append(os.path.join(ROOT, 'train_script', 'official'))
from train_det_head import SAMBBoxPrompter, build_grid_targets_single_class, yolo_detection_loss_single_class  # noqa: E402

THRESHOLDS = (0.3, 0.5, 0.7)
MAX_BOXES = 32  # per class and image (ground truth: <= 13 TCs); an untrained head proposes thousands


def gt_boxes_normalised(gt_np):
    """[[class_id, cx, cy, w, h], ...] per image; class 0 = TC, 1 = AR (as in the thesis script)."""
    out = []
    for g in gt_np:
        b = []
        for cid, (cls, label) in ((0.0, CLASSES[0]), (1.0, CLASSES[1])):
            for x1, y1, x2, y2 in components(g == label)[2]:
                b.append([cid, (x1 + x2 + 1) / 2 / IMG_W, (y1 + y2 + 1) / 2 / IMG_H, (x2 - x1 + 1) / IMG_W, (y2 - y1 + 1) / IMG_H])
        out.append(b)
    return out


@torch.no_grad()
def evaluate(climatesam, head, tokens, data, device, threshold, objects=False):
    head.eval()
    seg, fill, obj = SegMetrics(), SegMetrics(), ObjectMetrics()
    for i in range(len(data)):
        gt = data.data['gt'][i].long()
        emb, interm0 = data.data['emb'][i:i + 1].float(), data.data['vit'][0][i:i + 1].float()
        ar_logits, tc_logits = head(emb, tokens['AR'], tokens['TC'])
        pr = head.get_prompts(ar_logits.float(), tc_logits.float(), IMG_H, IMG_W, conf_threshold=threshold)
        preds, filled = {}, {}
        for cls, label in CLASSES:
            boxes = pr[f'{cls.lower()}_bbox_prompts'][0]
            f = torch.zeros_like(gt, dtype=torch.bool)
            if boxes is None:
                preds[cls] = f.clone()
            else:
                b = boxes[:MAX_BOXES, 0, :4].float()  # NMS output is sorted by confidence
                for x1, y1, x2, y2 in b.round().long().tolist():
                    f[y1:y2 + 1, x1:x2 + 1] = True
                preds[cls] = upsample(union_logits(sam_decode(climatesam, emb, interm0, cls, boxes=to_sam_frame(b))).to(device)) > 0
            filled[cls] = f
            if objects:
                obj.update(preds[cls], gt == label, cls)
        seg.update(preds['TC'], preds['AR'], gt)
        fill.update(filled['TC'], filled['AR'], gt)
    return seg, fill, obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoder', default='infused_mlp1')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--lr', type=float, default=1e-3)
    args = ap.parse_args()
    seed_everything(args.seed)
    device = torch.device('cuda')
    climatesam = load_climatesam(*ENCODERS[args.encoder], device)
    tokens = prompters.refined_tokens(climatesam)
    name = f'det_head_s{args.seed}'
    out_dir = os.path.join(RESULTS, 'runs', args.encoder, name)
    os.makedirs(out_dir, exist_ok=True)

    train = Batches(FeatureCache(args.encoder, 'train'), [0], device)
    val = Batches(FeatureCache(args.encoder, 'val'), [0], device)
    test = Batches(FeatureCache(args.encoder, 'test'), [0], device)
    targets = gt_boxes_normalised(train.data['gt'].cpu().numpy())

    head = SAMBBoxPrompter().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)
    rows, best, t0 = [], None, time.time()
    for epoch in range(1, args.epochs + 1):
        head.train()
        perm, total, steps = np.random.permutation(len(train)), 0.0, 0
        for s in range(0, len(perm), 8):
            idx = np.sort(perm[s:s + 8])
            emb = train.data['emb'][idx].float()
            ar_logits, tc_logits = head(emb, tokens['AR'], tokens['TC'])
            boxes = [targets[i] for i in idx]
            ar_t = build_grid_targets_single_class(boxes, ar_logits.shape[2:], device, 1.0)
            tc_t = build_grid_targets_single_class(boxes, tc_logits.shape[2:], device, 0.0)
            loss = yolo_detection_loss_single_class(ar_logits.float(), ar_t)[0] + yolo_detection_loss_single_class(tc_logits.float(), tc_t)[0]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total, steps = total + float(loss), steps + 1
        sched.step()
        row = {'epoch': epoch, 'train_loss': total / steps}
        if epoch % 2 == 0 or epoch == args.epochs:
            seg, _, _ = evaluate(climatesam, head, tokens, val, device, 0.5)
            row.update({f'val_{k}': v for k, v in seg.compute().items()})
            if best is None or row['val_Mean FG IoU'] > best['val_Mean FG IoU']:
                best = dict(row)
                torch.save({'state_dict': head.state_dict()}, os.path.join(out_dir, 'best.pth'))
        rows.append(row)
        print(f"[{name}] epoch {epoch} loss {row['train_loss']:.4f}" + (f" val FG {row['val_Mean FG IoU']:.4f}" if 'val_Mean FG IoU' in row else ''), flush=True)
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(os.path.join(out_dir, 'log.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    head.load_state_dict(torch.load(os.path.join(out_dir, 'best.pth'))['state_dict'])
    val_fg = {t: evaluate(climatesam, head, tokens, val, device, t)[0].compute()['Mean FG IoU'] for t in THRESHOLDS}
    thr = max(val_fg, key=val_fg.get)
    seg, fill, obj = evaluate(climatesam, head, tokens, test, device, thr, objects=True)
    res = {'rows': [{'method': name, 'output': 'own', **fill.compute()},
                    {'method': name, 'output': 'sam_bbox', **seg.compute(), **obj.compute()}],
           'error_decomposition': {}, 'per_image_counts': {'own': fill.per_image, 'sam_bbox': seg.per_image},
           'threshold': thr, 'val_fg_per_threshold': val_fg}
    with open(os.path.join(RESULTS, 'eval', args.encoder, f'{name}.json'), 'w') as f:
        json.dump(res, f)
    summary = {'name': name, 'seed': args.seed, 'trainable_params': sum(p.numel() for p in head.parameters()),
               'best_epoch': best['epoch'], 'train_time_min': (time.time() - t0) / 60, 'threshold': thr,
               **{k: v for k, v in best.items() if k.startswith('val_')}, 'test_sam_bbox': seg.compute(), 'args': vars(args)}
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
