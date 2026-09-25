"""
CG-Net with a YOLO-style box head (model/prompt/cgnet_bbox.py) as a box prompter for the frozen SAM.
Its boxes go straight to SAM; "own" is the union of the filled boxes (a box detector has no mask of its own).
"""
import os
import csv
import argparse
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm

from common import (RESULTS, CLASSES, SegMetrics, ObjectMetrics, FeatureCache, load_climatesam, sam_decode,
                    union_logits, upsample, to_sam_frame, seed_everything)
from build_cache import ENCODERS
from model.prompt.cgnet_bbox import CGNetBBoxPrompter

WEIGHTS = ['exp/cgnet_bbox_weight.pth', 'exp/best_weights/cgnet_bbox_weight.pth', 'exp/best_weights/best_exp_cgnet_bbox_weight.pth']


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoder', default='infused_mlp1')
    args = ap.parse_args()
    device = torch.device('cuda')
    seed_everything(0)
    ckpt, mlp = ENCODERS[args.encoder]
    sam = load_climatesam(ckpt, mlp, device)
    cache = FeatureCache(args.encoder, 'test')
    data = cache.load_to(device, [0], with_cgnet=True)
    rows = []
    for weights in WEIGHTS:
        prompter = CGNetBBoxPrompter(weights, device, SimpleNamespace(exp_dir='exp', wandb=False, run_name=None))
        for conf in (0.3, 0.5, 0.7):
            sam_m, box_m, sam_o = SegMetrics(), SegMetrics(), ObjectMetrics()
            for i in tqdm(range(len(cache)), desc=f'{weights} conf={conf}', leave=False):
                gt = data['gt'][i].long()
                pr = prompter.get_prompts(data['cgnet'][i:i + 1], conf_threshold=conf)
                emb, interm0 = data['emb'][i:i + 1].float(), data['vit'][0][i:i + 1].float()
                preds, filled = {}, {}
                for cls, _ in CLASSES:
                    boxes = pr[f'{cls.lower()}_bbox_prompts'][0]
                    fill = torch.zeros_like(gt, dtype=torch.bool)
                    if boxes is None:
                        preds[cls] = fill.clone()
                    else:
                        b = boxes[:, 0, :4].float()
                        for x1, y1, x2, y2 in b.round().long().tolist():
                            fill[y1:y2 + 1, x1:x2 + 1] = True
                        preds[cls] = upsample(union_logits(sam_decode(sam, emb, interm0, cls, boxes=to_sam_frame(b))).to(device)) > 0
                    filled[cls] = fill
                    sam_o.update(preds[cls], gt == dict(CLASSES)[cls], cls)
                sam_m.update(preds['TC'], preds['AR'], gt)
                box_m.update(filled['TC'], filled['AR'], gt)
            for output, m in (('own (filled boxes)', box_m), ('sam_bbox', sam_m)):
                rows.append({'weights': weights, 'conf_threshold': conf, 'output': output,
                             **{k: round(v, 4) for k, v in m.compute().items()},
                             **({k: round(v, 4) for k, v in sam_o.compute().items()} if output == 'sam_bbox' else {})})
                print(rows[-1], flush=True)

    out = os.path.join(RESULTS, '03_cgnet_yolo_boxes')
    os.makedirs(out, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(out, f'cgnet_yolo_{args.encoder}.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()
