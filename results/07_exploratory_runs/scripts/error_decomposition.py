"""Where does the existing generator lose IoU: false-positive objects, missed objects, or object shape?"""
import os, sys
ROOT = '/home/worker1/thang/ClimateSAM'
sys.path[:0] = [ROOT, os.path.join(ROOT, 'train_script'), os.path.join(ROOT, 'train_script/official')]
os.chdir(ROOT)
sys.argv = ['x', '--config', 'input_config_mask_prompt']

import cv2
import numpy as np
import torch
from parser_config import parse
from dataset.climatenet import ClimateDataset
from model.prompt_generator import PromptGenerator
from train_mask_prompt_generator import build_climatesam, cache_features, CLASSES

args = parse()
dev = torch.device('cuda')
sam = build_climatesam(args, dev)
val = cache_features(sam, ClimateDataset(data_dir=args.data_dir, train_flag=False, generate_prompt=False), list(range(12)), dev)
gen = PromptGenerator(in_channels=768, fused_channels=128, num_features=12, features_per_block=3).to(dev).eval()
gen.load_state_dict(torch.load('exp/best_weights/best_generator_vit_b_128_generator_128_vit_b_bbox.pth', map_location=dev)['prompt_generator'])

variants = ['as is', 'drop FP objects', 'add missed objects', 'drop FP + add missed', 'GT shape for detected objs']
inter = {(v, c): 0 for v in variants for c, _, _ in CLASSES}
union = dict(inter)
counts = {c: dict(gt=0, hit=0, pred=0, fp=0) for c, _, _ in CLASSES}

with torch.no_grad():
    for i in range(len(val['names'])):
        logit, _ = gen([val['vit'][l][i:i + 1].float() for l in range(12)])
        pred_cls = logit.argmax(1)[0].cpu().numpy()
        gt = val['gt'][i].cpu().numpy()
        for cls, label, _ in CLASSES:
            P, G = (pred_cls == label), (gt == label)
            np_, pl = cv2.connectedComponents(P.astype(np.uint8), connectivity=8)
            ng, gl = cv2.connectedComponents(G.astype(np.uint8), connectivity=8)
            fp = [k for k in range(1, np_) if not G[pl == k].any()]
            missed = [k for k in range(1, ng) if not P[gl == k].any()]
            detected = [k for k in range(1, ng) if P[gl == k].any()]
            c = counts[cls]
            c['gt'] += ng - 1; c['hit'] += len(detected); c['pred'] += np_ - 1; c['fp'] += len(fp)

            no_fp = P & ~np.isin(pl, fp)
            add_missed = P | np.isin(gl, missed)
            # detected GT objects get their exact GT shape; FP objects stay; missed stay missed
            shape = np.isin(pl, fp) | np.isin(gl, detected)
            for v, M in zip(variants, [P, no_fp, add_missed, no_fp | np.isin(gl, missed), shape]):
                inter[(v, cls)] += (M & G).sum()
                union[(v, cls)] += (M | G).sum()

for cls, _, _ in CLASSES:
    c = counts[cls]
    print(f"\n{cls}: {c['gt']} GT objects, {c['hit']} detected ({c['hit'] / c['gt']:.0%} recall); "
          f"{c['pred']} predicted objects, {c['fp']} false positives ({c['fp'] / max(c['pred'], 1):.0%})")
    for v in variants:
        print(f"  {v:>28}: IoU {inter[(v, cls)] / union[(v, cls)]:.2%}")
