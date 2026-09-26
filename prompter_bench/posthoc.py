"""
Evaluation-only improvements of the prompter masks (PLAN_20h C1-C3), all tuned on the 40 validation images:

  C1  seed ensemble (mean of the three seeds' logits) and a per-class logit threshold chosen on validation
  C2  object post-processing on top of C1: minimum blob size per class and, for TCs, a maximum |latitude| of the
      blob centroid (TCs form between 5 and 30 degrees, thesis Section 2.2.2)
  C3  iterative SAM refinement: SAM + hybrid prompts, then SAM's own low-resolution output fed back as the dense
      prompt (TC: with boxes of the new mask) for up to 3 rounds
"""
import os
import re
import sys
import json
import glob
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from common import (RESULTS, CLASSES, SegMetrics, ObjectMetrics, FeatureCache, load_climatesam, sam_decode, union_logits,
                    upsample, make_prompts, score_to_prompt, to_sam_frame, seed_everything, IMG_H, IMG_W, LOWRES, MIN_OBJECT_PX)
from build_cache import ENCODERS
from train import Batches
from evaluate import kind_for
import prompters
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_report as mr

THRESHOLDS = np.round(np.linspace(-4.0, 4.0, 33), 2)
MIN_AREAS = [0, 50, 100, 200, 400, 800, 1600]
MAX_LAT = [90, 45, 40, 35]
LAT = np.linspace(-90, 90, IMG_H)


@torch.no_grad()
def logits_for(run, encoder, split, climatesam, device):
    """Full-resolution TC / AR logits (float16, CPU) of one trained run on a split."""
    ck = torch.load(os.path.join(RESULTS, 'runs', encoder, run, 'best.pth'), map_location=device)
    model, layers = prompters.build(ck['arch'], climatesam=climatesam)
    model.load_state_dict(ck['state_dict'])
    model = model.to(device).eval()
    data = Batches(FeatureCache(encoder, split), layers, device, prompters.needs_fields(ck['arch']))
    out = []
    for s in range(0, len(data), 2):
        batch = data.get(np.arange(s, min(s + 2, len(data))))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            o = model(batch)
        out.append(F.interpolate(o['logits'].float(), (IMG_H, IMG_W), mode='bilinear', align_corners=False).half().cpu())
    return torch.cat(out)


def iou(pred, gt):
    i = np.logical_and(pred, gt).sum()
    return i / max(np.logical_or(pred, gt).sum(), 1)


def postprocess(mask, min_area, max_lat):
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = [k for k in range(1, n) if st[k, 4] >= min_area and abs(LAT[int(cen[k, 1])]) <= max_lat]
    return np.isin(lab, keep)


def class_masks(logits, t, pp=None):
    """logits (N, 2, H, W) -> per class list of numpy masks, threshold t[c] and optional post-processing pp[c]."""
    out = {}
    for ch, (cls, _) in enumerate(CLASSES):
        ms = (logits[:, ch].float() > t[cls]).numpy()
        if pp is not None:
            ms = np.stack([postprocess(m, *pp[cls]) for m in ms])
        out[cls] = ms
    return out


def dataset_iou(masks, gts, cls):
    label = dict(CLASSES)[cls]
    inter = sum(np.logical_and(m, g == label).sum() for m, g in zip(masks, gts))
    union = sum(np.logical_or(m, g == label).sum() for m, g in zip(masks, gts))
    return inter / max(union, 1)


def tune(val_logits, val_gt):
    """Per-class threshold, then per-class post-processing, each maximising the validation IoU of that class."""
    t = {}
    for ch, (cls, _) in enumerate(CLASSES):
        scores = [dataset_iou((val_logits[:, ch].float() > th).numpy(), val_gt, cls) for th in THRESHOLDS]
        t[cls] = float(THRESHOLDS[int(np.argmax(scores))])
    pp = {}
    for ch, (cls, _) in enumerate(CLASSES):
        base = (val_logits[:, ch].float() > t[cls]).numpy()
        best, arg = -1, None
        for a in MIN_AREAS:
            for lat in (MAX_LAT if cls == 'TC' else [90]):
                v = dataset_iou([postprocess(m, a, lat) for m in base], val_gt, cls)
                if v > best:
                    best, arg = v, (a, lat)
        pp[cls] = arg
    return t, pp


def metrics_of(masks, gts):
    m, o = SegMetrics(), ObjectMetrics()
    for i, g in enumerate(gts):
        gt = torch.from_numpy(g.astype(np.int64))
        m.update(torch.from_numpy(masks['TC'][i]), torch.from_numpy(masks['AR'][i]), gt)
        for cls, label in CLASSES:
            o.update(torch.from_numpy(masks[cls][i]), gt == label, cls)
    return {**m.compute(), **o.compute()}, m.per_image


@torch.no_grad()
def sam_rounds(climatesam, data, logits, masks, rounds, device):
    """SAM + hybrid prompts from (logits, masks), then SAM's own output fed back `rounds - 1` times."""
    results = []
    cur_logits = [logits[i].float().to(device) for i in range(len(masks['TC']))]
    cur_masks = {c: list(masks[c]) for c in masks}
    for r in range(rounds):
        seg = SegMetrics()
        new_logits, new_masks = [], {'TC': [], 'AR': []}
        for i in range(len(cur_logits)):
            gt = data.data['gt'][i].long()
            emb, interm0 = data.data['emb'][i:i + 1].float(), data.data['vit'][0][i:i + 1].float()
            lg_img, preds = [], {}
            for ch, (cls, _) in enumerate(CLASSES):
                m = cur_masks[cls][i].astype(np.uint8)
                p = make_prompts(m, kind_for('hybrid', cls), score=cur_logits[i][ch], device=device)
                lg = sam_decode(climatesam, emb, interm0, cls, **p) if p else None
                low = union_logits(lg).to(device)
                full = upsample(low, mode='bilinear')
                preds[cls] = upsample(low) > 0
                lg_img.append(full)
                new_masks[cls].append((full > 0).cpu().numpy())
            new_logits.append(torch.stack(lg_img))
            seg.update(preds['TC'], preds['AR'], gt)
        results.append({'round': r + 1, **seg.compute(), 'per_image': seg.per_image})
        cur_logits, cur_masks = new_logits, new_masks
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoder', default='infused_mlp1')
    ap.add_argument('--methods', nargs='+', default=['mpg_seg', 'msf_seg', 'msf_token_seg', 'logreg_last_seg', 'mpg_seg_smooth',
                                                     'mpg_twostage', 'cgnet_scratch_seg', 'msf_sp_token_seg'])
    ap.add_argument('--rounds', type=int, default=3)
    args = ap.parse_args()
    seed_everything(0)
    device = torch.device('cuda')
    climatesam = load_climatesam(*ENCODERS[args.encoder], device)
    val_gt = FeatureCache(args.encoder, 'val').gt[FeatureCache(args.encoder, 'val').index]
    test_cache = FeatureCache(args.encoder, 'test')
    test_gt = np.asarray(test_cache.gt[:])
    test_data = Batches(test_cache, [0], device)
    out_dir = os.path.join(RESULTS, '06_posthoc')
    os.makedirs(out_dir, exist_ok=True)
    for method in args.methods:
        # 'a+b': ensemble across two prompter families (the runs of a first, so seed 0 of a is the reference)
        runs = [r for part in method.split('+') for r in sorted(
            os.path.basename(d) for d in glob.glob(os.path.join(RESULTS, 'runs', args.encoder, f'{part}_s*'))
            if re.fullmatch(rf'{part}_s\d+', os.path.basename(d)) and os.path.exists(os.path.join(d, 'summary.json')))]
        if not runs:
            print(f'skip {method}: no finished runs')
            continue
        val = [logits_for(r, args.encoder, 'val', climatesam, device) for r in runs]
        test = [logits_for(r, args.encoder, 'test', climatesam, device) for r in runs]
        zero = {'TC': 0.0, 'AR': 0.0}
        res = {'method': method, 'runs': runs, 'variants': {}}
        # single seeds, threshold 0 (the benchmark numbers) and calibrated
        singles, singles_cal = [], []
        for v, t_ in zip(val, test):
            singles.append(metrics_of(class_masks(t_, zero), test_gt))
            th, _ = tune(v, val_gt)
            singles_cal.append(metrics_of(class_masks(t_, th), test_gt))
        res['variants']['single seed'] = {k: float(np.mean([s[0][k] for s in singles])) for k in singles[0][0]}
        res['variants']['single seed, calibrated'] = {k: float(np.mean([s[0][k] for s in singles_cal])) for k in singles_cal[0][0]}
        per_image = {'single seed (seed 0)': singles[0][1]}
        ens_val, ens_test = torch.stack(val).float().mean(0).half(), torch.stack(test).float().mean(0).half()
        m, pi = metrics_of(class_masks(ens_test, zero), test_gt)
        res['variants']['ensemble'], per_image['ensemble'] = m, pi
        th, pp = tune(ens_val, val_gt)
        res['threshold'], res['postprocess'] = th, {c: {'min_area': a, 'max_lat': l} for c, (a, l) in pp.items()}
        m, pi = metrics_of(class_masks(ens_test, th), test_gt)
        res['variants']['ensemble, calibrated'], per_image['ensemble, calibrated'] = m, pi
        final_masks = class_masks(ens_test, th, pp)
        m, pi = metrics_of(final_masks, test_gt)
        res['variants']['ensemble, calibrated, post-processed'], per_image['ensemble, calibrated, post-processed'] = m, pi
        # SAM refinement of the final masks (dense prompt = logits shifted by the threshold)
        shifted = torch.stack([ens_test[:, ch].float() - th[c] for ch, (c, _) in enumerate(CLASSES)], dim=1)
        for r in sam_rounds(climatesam, test_data, shifted, final_masks, args.rounds, device):
            key = f'SAM hybrid, round {r["round"]}'
            per_image[key] = r.pop('per_image')
            res['variants'][key] = {k: v for k, v in r.items() if k != 'round'}
        # paired bootstrap of every variant against the single-seed benchmark mask (seed 0)
        base = np.array(per_image['single seed (seed 0)'], dtype=np.float64)
        res['bootstrap_vs_seed0'] = {k: {m_: list(v) for m_, v in mr.bootstrap(base, np.array(pi_, dtype=np.float64)).items()}
                                     for k, pi_ in per_image.items() if k != 'single seed (seed 0)'}
        res['per_image_counts'] = per_image
        with open(os.path.join(out_dir, f'posthoc_{args.encoder}_{method}.json'), 'w') as f:
            json.dump(res, f)
        print(method, {k: (round(v['TC IoU'], 3), round(v['AR IoU'], 3), round(v['Mean FG IoU'], 3)) for k, v in res['variants'].items()},
              res['threshold'], res['postprocess'], flush=True)


if __name__ == '__main__':
    main()
