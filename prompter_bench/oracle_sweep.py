"""
Ground-truth prompt studies on the test set (no prompter involved):

  mask_format   how the decoder reads dense mask prompts: {0,1} maps vs logit maps of scale a (+-a),
                one prompt per class (union of all objects) vs one prompt per object, with / without a box
  degradation   how SAM reacts to imperfect prompts built from the ground truth:
                boxes enlarged / shrunk, masks eroded / dilated, objects dropped, false boxes added
"""
import os
import csv
import argparse

import cv2
import numpy as np
import torch
from tqdm import tqdm

from common import (RESULTS, CLASSES, SegMetrics, FeatureCache, load_climatesam, sam_decode, union_logits, upsample,
                    score_to_prompt, binary_to_score, to_sam_frame, seed_everything, MIN_OBJECT_PX)
from build_cache import ENCODERS
from model.prompt.prompt_maker import make_bbox_prompts


def objects(mask_np):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask_np.astype(np.uint8), connectivity=8)
    return [(lab == k) for k in range(1, n) if st[k, 4] >= MIN_OBJECT_PX]


def box_of(obj):
    ys, xs = np.nonzero(obj)
    return [xs.min(), ys.min(), xs.max(), ys.max()]


def resize_box(b, r, w=1152, h=768):
    bw, bh = b[2] - b[0], b[3] - b[1]
    return [max(0, b[0] - r * bw), max(0, b[1] - r * bh), min(w - 1, b[2] + r * bw), min(h - 1, b[3] + r * bh)]


def mask_format_prompts(obj_list, cls_mask, device):
    """name -> kwargs for sam_decode."""
    out = {}
    if not obj_list:
        return out
    boxes = to_sam_frame(torch.tensor([box_of(o) for o in obj_list], dtype=torch.float32, device=device))
    union01 = score_to_prompt(torch.from_numpy(cls_mask).to(device).float())
    per01 = torch.cat([score_to_prompt(torch.from_numpy(o).to(device).float()) for o in obj_list])
    out['union {0,1}'] = {'masks': union01}
    out['per-object {0,1}'] = {'masks': per01}
    for a in (2, 5, 10, 20):
        out[f'union logits +-{a}'] = {'masks': union01 * 2 * a - a}
        out[f'per-object logits +-{a}'] = {'masks': per01 * 2 * a - a}
    out['box'] = {'boxes': boxes}
    out['box + per-object logits +-10'] = {'boxes': boxes, 'masks': per01 * 20 - 10}
    out['box + union logits +-10'] = {'boxes': boxes, 'masks': (union01 * 20 - 10).expand(len(obj_list), -1, -1, -1)}
    return out


def degradation_prompts(obj_list, cls_mask, rng, device):
    out = {}
    if not obj_list:
        return out
    base = [box_of(o) for o in obj_list]
    for r in (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5):
        b = torch.tensor([resize_box(x, r) for x in base], dtype=torch.float32, device=device)
        out[f'box enlarge {r:+.1f}'] = {'boxes': to_sam_frame(b)}
    for k in (-9, -5, -3, 0, 3, 5, 9):  # erode (<0) / dilate (>0) the mask prompt by |k| px
        m = cls_mask.astype(np.uint8)
        if k:
            kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (abs(k), abs(k)))
            m = cv2.dilate(m, kern) if k > 0 else cv2.erode(m, kern)
        out[f'mask morph {k:+d}px'] = {'masks': score_to_prompt(binary_to_score(torch.from_numpy(m).to(device)))}
    for drop in (0.25, 0.5):  # remove a fraction of the objects from the box prompts
        keep = [b for b in base if rng.random() > drop]
        if keep:
            out[f'box drop {int(drop * 100)}% objects'] = {
                'boxes': to_sam_frame(torch.tensor(keep, dtype=torch.float32, device=device))}
    for extra in (1, 3):  # add random false boxes of a typical object size
        fake = []
        for _ in range(extra):
            ref = base[rng.integers(len(base))]
            bw, bh = ref[2] - ref[0], ref[3] - ref[1]
            x, y = rng.uniform(0, 1152 - bw), rng.uniform(0, 768 - bh)
            fake.append([x, y, x + bw, y + bh])
        out[f'box + {extra} false boxes'] = {
            'boxes': to_sam_frame(torch.tensor(base + fake, dtype=torch.float32, device=device))}
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoder', default='infused_mlp1')
    args = ap.parse_args()
    device = torch.device('cuda')
    seed_everything(0)
    rng = np.random.default_rng(0)
    ckpt, mlp = ENCODERS[args.encoder]
    sam = load_climatesam(ckpt, mlp, device)
    data = FeatureCache(args.encoder, 'test').load_to(device, [0])

    # every prompt variant, so an image where a variant produced no prompt still counts (as an empty prediction)
    fake = np.zeros((768, 1152), bool)
    fake[100:140, 100:140] = True
    keys = [('mask_format', n) for n in mask_format_prompts([fake], fake, device)] + \
           [('degradation', n) for n in degradation_prompts([fake], fake, np.random.default_rng(1), device)
            if 'drop' not in n] + [('degradation', f'box drop {d}% objects') for d in (25, 50)]
    metrics = {k: SegMetrics() for k in keys}
    for i in tqdm(range(len(data['names'])), desc=args.encoder):
        gt = data['gt'][i].long()
        emb, interm0 = data['emb'][i:i + 1].float(), data['vit'][0][i:i + 1].float()
        preds = {}
        for cls, label in CLASSES:
            cls_mask = (gt == label).cpu().numpy()
            objs = objects(cls_mask)
            for study, prompts in (('mask_format', mask_format_prompts(objs, cls_mask, device)),
                                   ('degradation', degradation_prompts(objs, cls_mask, rng, device))):
                for name, kw in prompts.items():
                    lg = sam_decode(sam, emb, interm0, cls, **kw)
                    preds.setdefault((study, name), {})[cls] = upsample(union_logits(lg).to(device)) > 0
        empty = torch.zeros_like(gt, dtype=torch.bool)
        for key in keys:
            p = preds.get(key, {})
            metrics[key].update(p.get('TC', empty), p.get('AR', empty), gt)

    out = os.path.join(RESULTS, '01_oracle_prompts')
    os.makedirs(out, exist_ok=True)
    rows = [{'encoder': args.encoder, 'study': s, 'prompt': n, **{k: round(v, 4) for k, v in m.compute().items()}}
            for (s, n), m in metrics.items()]
    with open(os.path.join(out, f'oracle_sweep_{args.encoder}.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == '__main__':
    main()
