"""
Test-set evaluation of every prompter with every way of turning its output into SAM prompts.

For each method:
  own        the prompter's own TC / AR masks (no SAM)
  sam_<kind> SAM output when the prompter's masks are converted into <kind> prompts:
             bbox, bbox_e10 (boxes enlarged 10%), point (5 pos + 5 neg per blob), bbox+point,
             mask (dense logit map), bbox+mask, hybrid (TC: bbox+mask, AR: mask)
Plus object-level recall / precision and an error decomposition of the own masks.
"""
import os
import sys
import csv
import json
import glob
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from common import (RESULTS, CLASSES, SegMetrics, ObjectMetrics, FeatureCache, load_climatesam, sam_decode,
                    union_logits, upsample, make_prompts, binary_to_score, seed_everything, MIN_OBJECT_PX)
from build_cache import ENCODERS
from train import Batches
import prompters

KINDS = ['bbox', 'bbox_e10', 'point', 'bbox+point', 'mask', 'bbox+mask', 'hybrid']
EXAMPLES = [3, 17, 31, 45]  # fixed test images saved for the qualitative figures


def kind_for(kind, cls):
    if kind == 'hybrid':
        return 'bbox+mask' if cls == 'TC' else 'mask'
    return 'bbox' if kind == 'bbox_e10' else kind


def decompose(pred, gt_mask):
    """IoU of the own mask after fixing one error type at a time (numpy bool masks) -> dict of (inter, union)."""
    n_p, pl = cv2.connectedComponents(pred.astype(np.uint8), connectivity=8)
    n_g, gl = cv2.connectedComponents(gt_mask.astype(np.uint8), connectivity=8)
    hit_gt = set(np.unique(gl[pred & (gl > 0)]))
    hit_pred = set(np.unique(pl[gt_mask & (pl > 0)]))
    fp = np.isin(pl, [k for k in range(1, n_p) if k not in hit_pred])
    missed = np.isin(gl, [k for k in range(1, n_g) if k not in hit_gt])
    detected = np.isin(gl, [k for k in range(1, n_g) if k in hit_gt])
    variants = {
        'as_is': pred,
        'no_false_objects': pred & ~fp,
        'add_missed_objects': pred | missed,
        'perfect_detection': (pred & ~fp) | missed,
        'perfect_shape_of_detected': fp | detected,
    }
    return {k: ((m & gt_mask).sum(), (m | gt_mask).sum()) for k, m in variants.items()}


@torch.no_grad()
def evaluate_method(name, predict, data, climatesam, device, out_dir, kinds=KINDS, hard_masks=None):
    """
    predict(batch) -> (B, 2, h, w) logits (TC, AR). hard_masks(batch) -> optional (B, H, W) label map that
    replaces "logits > 0" for the own mask (CG-Net uses its 3-class argmax).
    """
    seed_everything(0)
    own, own_obj = SegMetrics(), ObjectMetrics()
    sam = {k: SegMetrics() for k in kinds}
    sam_obj = {k: ObjectMetrics() for k in kinds}
    fused, fused_obj = SegMetrics(), ObjectMetrics()  # mean of prompter and SAM (hybrid) probabilities > 0.5
    dec = {cls: {k: [0, 0] for k in ('as_is', 'no_false_objects', 'add_missed_objects', 'perfect_detection',
                                     'perfect_shape_of_detected')} for cls, _ in CLASSES}
    examples = {}
    for s in tqdm(range(0, len(data), 4), desc=name, leave=False):
        pos = np.arange(s, min(s + 4, len(data)))
        batch = data.get(pos)
        logits = F.interpolate(predict(batch).float(), size=batch['gt'].shape[-2:], mode='bilinear', align_corners=False)
        hard = hard_masks(batch) if hard_masks else None
        for b in range(len(pos)):
            gt = batch['gt'][b]
            masks = {cls: (hard[b] == label) if hard is not None else logits[b, ch] > 0
                     for ch, (cls, label) in enumerate(CLASSES)}
            own.update(masks['TC'], masks['AR'], gt)
            ex = {'gt': gt.cpu().numpy().astype(np.uint8)} if pos[b] in EXAMPLES else None
            for ch, (cls, label) in enumerate(CLASSES):
                own_obj.update(masks[cls], gt == label, cls)
                for k, (i, u) in decompose(masks[cls].cpu().numpy(), (gt == label).cpu().numpy()).items():
                    dec[cls][k][0] += i
                    dec[cls][k][1] += u
                if ex is not None:
                    ex[f'own_{cls}'] = masks[cls].cpu().numpy()
            emb, interm0 = batch['emb'][b:b + 1], batch['vit'][0][b:b + 1]
            for kind in kinds:
                preds, fused_preds = {}, {}
                for ch, (cls, label) in enumerate(CLASSES):
                    m = masks[cls].cpu().numpy().astype(np.uint8)
                    score = logits[b, ch] if hard is None else binary_to_score(masks[cls])
                    prompts = make_prompts(m, kind_for(kind, cls), score=score,
                                           enlarge_ratio=0.1 if kind == 'bbox_e10' else 0.0, device=device)
                    lg = sam_decode(climatesam, emb, interm0, cls, **prompts) if prompts else None
                    sam_logits = upsample(union_logits(lg).to(device))
                    preds[cls] = sam_logits > 0
                    if kind == 'hybrid' and hard is None:
                        fz = (torch.sigmoid(sam_logits) + torch.sigmoid(logits[b, ch])) / 2 > 0.5
                        fused_preds[cls] = fz
                        fused_obj.update(fz, gt == label, cls)
                    if ex is not None and kind in ('bbox', 'hybrid'):
                        ex[f'sam_{kind}_{cls}'] = preds[cls].cpu().numpy()
                        if prompts and 'boxes' in prompts:
                            ex[f'boxes_{kind}_{cls}'] = prompts['boxes'].cpu().numpy() * np.array([1152, 768, 1152, 768]) / 1024
                    sam_obj[kind].update(preds[cls], gt == label, cls)
                sam[kind].update(preds['TC'], preds['AR'], gt)
                if kind == 'hybrid' and hard is None:
                    fused.update(fused_preds['TC'], fused_preds['AR'], gt)
            if ex is not None:
                examples[int(pos[b])] = ex

    rows = [{'method': name, 'output': 'own', **own.compute(), **own_obj.compute()}]
    for k in kinds:
        rows.append({'method': name, 'output': f'sam_{k}', **sam[k].compute(), **sam_obj[k].compute()})
    if hard is None and 'hybrid' in kinds:
        rows.append({'method': name, 'output': 'fused_hybrid', **fused.compute(), **fused_obj.compute()})
    decomposition = {cls: {k: i / max(u, 1) for k, (i, u) in d.items()} for cls, d in dec.items()}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f'{name}.json'), 'w') as f:
        json.dump({'rows': rows, 'error_decomposition': decomposition}, f, indent=2)
    np.savez_compressed(os.path.join(out_dir, f'{name}_examples.npz'),
                        **{f'{i}__{k}': v for i, ex in examples.items() for k, v in ex.items()})
    return rows, decomposition


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoder', default='infused_mlp1')
    ap.add_argument('--methods', nargs='+', default=['oracle', 'cgnet_official', 'cgnet_finetuned', 'runs'])
    ap.add_argument('--runs', default='*', help='glob over results/runs/<encoder>/')
    args = ap.parse_args()

    device = torch.device('cuda')
    ckpt, mlp = ENCODERS[args.encoder]
    climatesam = load_climatesam(ckpt, mlp, device)
    out_dir = os.path.join(RESULTS, 'eval', args.encoder)
    test_cache = FeatureCache(args.encoder, 'test')

    def data_for(layers, cgnet=False):
        d = Batches(test_cache, sorted(set(layers) | {0}), device)
        if cgnet:  # the raw CG-Net fields are only loaded for the CG-Net prompters
            base = d.get
            d.get = lambda pos: {**base(pos), 'cgnet': torch.from_numpy(
                np.ascontiguousarray(test_cache.cgnet[np.sort(np.asarray(pos))])).to(device)}
        return d

    for method in args.methods:
        if method == 'oracle':
            data = data_for([0])
            evaluate_method('oracle_gt', lambda b: binary_to_score(
                torch.stack([b['gt'] == 1, b['gt'] == 2], dim=1)), data, climatesam, device, out_dir)
        elif method.startswith('cgnet'):
            model = prompters.build(method)[0].to(device).eval()
            evaluate_method(method, lambda b: model(b)['logits'], data_for([0], cgnet=True), climatesam, device, out_dir,
                            hard_masks=lambda b: model(b)['argmax'])
        elif method == 'runs':
            for run_dir in sorted(glob.glob(os.path.join(RESULTS, 'runs', args.encoder, args.runs))):
                ck = torch.load(os.path.join(run_dir, 'best.pth'), map_location=device)
                model, layers = prompters.build(ck['arch'], climatesam=climatesam)
                model.load_state_dict(ck['state_dict'])
                model = model.to(device).eval()

                def predict(b, model=model):
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        return model(b)['logits']
                evaluate_method(os.path.basename(run_dir), predict, data_for(layers), climatesam, device, out_dir)
                del model
                torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
