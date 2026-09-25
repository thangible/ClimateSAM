"""Upper bound of the mask-prompt route: feed GT masks as SAM mask prompts (class-union vs per-object)."""
import os, sys
ROOT = '/home/worker1/thang/ClimateSAM'
sys.path[:0] = [ROOT, os.path.join(ROOT, 'train_script'), os.path.join(ROOT, 'train_script/official')]
os.chdir(ROOT)
sys.argv = ['x', '--config', 'input_config_mask_prompt']

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from parser_config import parse
from evaluator import StreamSegMetrics
from dataset.climatenet import ClimateDataset
from train_mask_prompt_generator import build_climatesam, cache_features, sam_decode, CLASSES


def decode_boxes(sam, emb, interm0, boxes, cls):
    s, d = sam.prompt_encoder(points=None, boxes=boxes, masks=None)
    _, m = sam.mask_decoder(type=cls, image_embeddings=emb, image_pe=[sam.prompt_encoder.get_dense_pe()], sparse_prompt_embeddings=[s], dense_prompt_embeddings=[d], multimask_output=False, interm_embeddings=[interm0])
    return m[0]


def to_box(o):
    ys, xs = torch.nonzero(o, as_tuple=True)
    return torch.tensor([[xs.min() * 1024 / 1152, ys.min() * 1024 / 768, xs.max() * 1024 / 1152, ys.max() * 1024 / 768]], device=dev).round()


args = parse()
dev = torch.device('cuda')
sam = build_climatesam(args, dev)
val = cache_features(sam, ClimateDataset(data_dir=args.data_dir, train_flag=False, generate_prompt=False), [0], dev)


def to_prompt(m):  # (H, W) {0,1} -> (1, 1, 256, 256), same resize as the dataset's mask prompts
    return (F.interpolate(m[None, None].float(), (256, 256), mode='bilinear') > 0.5).float()


def objects(m, min_px=20):
    n, lab = cv2.connectedComponents(m.cpu().numpy().astype(np.uint8), connectivity=8)
    objs = [torch.from_numpy(lab == k).to(dev) for k in range(1, n) if (lab == k).sum() >= min_px]
    return objs


scales = [5, 10, 20]
modes = ['bbox'] + [f'bbox+{k}_{a}' for k in ('union', 'per_object') for a in scales]
metrics = {(m, c): StreamSegMetrics(['Background', 'Foreground']) for m in modes for c, _, _ in CLASSES}
with torch.no_grad():
    for i in range(len(val['names'])):
        emb, interm0, gt = val['emb'][i:i + 1].float(), val['vit'][0][i:i + 1].float(), val['gt'][i].long()
        for cls, label, _ in CLASSES:
            g = (gt == label)
            prompts = {
                'union': to_prompt(g),
                'per_object': torch.cat([to_prompt(o) for o in objects(g)]) if g.any() else to_prompt(g),
                'empty': torch.zeros(1, 1, 256, 256, device=dev),
            }
            for a in scales:
                prompts[f'union_{a}'] = prompts['union'] * 2 * a - a
                prompts[f'per_object_{a}'] = prompts['per_object'] * 2 * a - a
            for m in modes:
                if m.startswith('bbox'):
                    if not g.any():
                        continue
                    boxes = torch.cat([to_box(o) for o in objects(g)])
                    if m == 'bbox':
                        logits = decode_boxes(sam, emb, interm0, boxes, cls)
                    else:
                        mp = prompts[m[5:]]
                        mp = mp.expand(len(boxes), -1, -1, -1) if mp.shape[0] == 1 else mp
                        s_, d_ = sam.prompt_encoder(points=None, boxes=boxes, masks=mp)
                        logits = sam.mask_decoder(type=cls, image_embeddings=emb, image_pe=[sam.prompt_encoder.get_dense_pe()], sparse_prompt_embeddings=[s_], dense_prompt_embeddings=[d_], multimask_output=False, interm_embeddings=[interm0])[1][0]
                else:
                    logits = sam_decode(sam, emb, interm0, [prompts[m]], cls)[0]  # (N,1,256,256)
                pred = (F.interpolate(logits.float(), gt.shape, mode='nearest') > 0).any(0, keepdim=True)
                metrics[(m, cls)].update([g[None, None].to(torch.uint8)], [pred.to(torch.uint8)], [val['names'][i]])

print()
for m in modes:
    r = {c: metrics[(m, c)].compute()[0]['Mean Foreground IoU'] for c, _, _ in CLASSES}
    print(f"{m:>16}: TC {r['TC']:.2%}  AR {r['AR']:.2%}")
