"""Evaluate the existing PromptGenerator checkpoint with the new prompt scheme (AR logit mask, TC boxes + logit mask)."""
import os, sys
ROOT = '/home/worker1/thang/ClimateSAM'
sys.path[:0] = [ROOT, os.path.join(ROOT, 'train_script'), os.path.join(ROOT, 'train_script/official')]
os.chdir(ROOT)
sys.argv = ['x', '--config', 'input_config_mask_prompt']

import torch
import torch.nn.functional as F
from parser_config import parse
from evaluator import StreamSegMetrics
from dataset.climatenet import ClimateDataset
from model.prompt_generator import PromptGenerator
from train_mask_prompt_generator import build_climatesam, cache_features, sam_decode, boxes_from_mask, CLASSES, EMPTY_LOGIT

args = parse()
dev = torch.device('cuda')
sam = build_climatesam(args, dev)
val = cache_features(sam, ClimateDataset(data_dir=args.data_dir, train_flag=False, generate_prompt=False), list(range(12)), dev)

gen = PromptGenerator(in_channels=768, fused_channels=128, num_features=12, features_per_block=3).to(dev).eval()
gen.load_state_dict(torch.load('exp/best_weights/best_generator_vit_b_128_generator_128_vit_b_bbox.pth', map_location=dev)['prompt_generator'])

scales = [1.0, 2.0]
metrics = {(k, c): StreamSegMetrics(['Background', 'Foreground']) for k in ['gen'] + [f'sam_x{s}' for s in scales] for c, _, _ in CLASSES}
with torch.no_grad():
    for i in range(len(val['names'])):
        emb, interm0, gt = val['emb'][i:i + 1].float(), val['vit'][0][i:i + 1].float(), val['gt'][i].long()
        logit, _ = gen([val['vit'][l][i:i + 1].float() for l in range(12)])  # (1, 3, 768, 1152)
        logp = F.log_softmax(logit.float(), dim=1)
        pred_cls = logit.argmax(1)[0]
        for cls, label, _ in CLASSES:
            g = (gt == label)[None, None].to(torch.uint8)
            metrics[('gen', cls)].update([g], [(pred_cls == label)[None, None].to(torch.uint8)], [val['names'][i]])
            # per-class log-odds, resized to SAM's 256x256 mask-prompt frame
            lo = logp[:, label] - torch.log1p(-logp[:, label].exp().clamp(max=1 - 1e-6))
            lo = F.interpolate(lo[:, None], (256, 256), mode='bilinear')
            for s in scales:
                prompt = (lo * s).clamp(-20, 20)
                if cls == 'AR':
                    out = sam_decode(sam, emb, interm0, 'AR', [prompt])[0][:, 0].max(0).values
                else:
                    boxes = boxes_from_mask(prompt[0, 0] > 0)
                    out = (sam_decode(sam, emb, interm0, 'TC', [prompt.expand(len(boxes), -1, -1, -1)], boxes=[boxes])[0][:, 0].max(0).values
                           if len(boxes) else torch.full((256, 256), EMPTY_LOGIT, device=dev))
                pred = F.interpolate(out[None, None].float(), gt.shape, mode='nearest') > 0
                metrics[(f'sam_x{s}', cls)].update([g], [pred.to(torch.uint8)], [val['names'][i]])

print()
for k in ['gen'] + [f'sam_x{s}' for s in scales]:
    r = {c: metrics[(k, c)].compute()[0]['Mean Foreground IoU'] for c, _, _ in CLASSES}
    print(f"{k:>8}: TC {r['TC']:.2%}  AR {r['AR']:.2%}")
