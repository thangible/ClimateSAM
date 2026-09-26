"""
Re-evaluation of the thesis' Phase-1 models (appendix Table 1 / Tables 4.6, 4.10, 4.12) from their saved checkpoints,
with the thesis' own evaluation pipeline: ClimateDataset(train_flag=False, generate_prompt=True, prompt_type=...)
(= the 61 ClimateNet TEST images, prompts from the ground truth), ClimateSAM.infer / forward, and the metric of
train_adaptation.validate_one_epoch (TC and AR evaluated as two independent binary problems).

Two image paths, as in table413_recheck.py:
  original : ClimateSAM.set_infer_img() -- what train_adaptation.validate_one_epoch does: the first 3 raw channels,
             the learned input adapter is skipped
  fixed    : ClimateSAM.encode_images() -- the 16 channels through the input adapter, as in training
'random' prompts are drawn per image (point or box), so they are evaluated with 3 seeds.
"""
import os
import sys
import csv
import copy
import random
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import RESULTS, DATA_DIR, load_climatesam, seed_everything
from dataset.climatenet import ClimateDataset
from utility import batch_to_cuda

# checkpoint (exp/, no .pth), mlp ratio, adapter, thesis row (appendix Table 1) or None
CHECKPOINTS = [
    ('infused_token_vit_b_1.0_infused_token_vit_b_mlp1_CORRECTED', 1.0, 'linear', 'Infused Token, Linear, 1.0'),
    ('infused_token_vit_b_0.5_infused_token_vit_b_mlp05_CORRECTED', 0.5, 'linear', 'Infused Token, Linear, 0.5'),
    ('infused_token_vit_b_1.0_infused_token_vit_b_mlp1_nonlinear', 1.0, 'nonlinear', 'Infused Token, Nonlinear, 1.0'),
    ('infused_token_vit_b_0.5_infused_token_vit_b_mlp05_nonlinear', 0.5, 'nonlinear', 'Infused Token, Nonlinear, 0.5'),
    ('best_weights/infused_token_vit_b_1.0_infused_token_vit_b_mlp1_CORRECTED_NOSMOOTH', 1.0, 'linear', None),
    ('best_weights/infused_token_vitb_mlp1_best', 1.0, 'linear', None),
    ('infused_token_vit_b_1.0_best_only_bbox', 1.0, 'linear', None),
    ('infused_token_vit_b_0.5_retrain_infused_05_bbox', 0.5, 'linear', None),
]
# appendix Table 1 of the thesis: (TC IoU, AR IoU) per prompt type
THESIS = {
    'Infused Token, Linear, 1.0': {'bbox': (0.7242, 0.6323), 'point': (0.5225, 0.4557), 'random': (0.5951, 0.5234)},
    'Infused Token, Linear, 0.5': {'bbox': (0.7244, 0.6448), 'point': (0.5256, 0.4685), 'random': (0.6056, 0.5362)},
    'Infused Token, Nonlinear, 1.0': {'bbox': (0.6987, 0.4446), 'point': (0.3308, 0.3134), 'random': (0.4468, 0.3605)},
    'Infused Token, Nonlinear, 0.5': {'bbox': (0.7332, 0.5219), 'point': (0.4760, 0.3506), 'random': (0.5718, 0.4213)},
}
PROMPTS = [('bbox', 0), ('point', 0), ('random', 0), ('random', 1), ('random', 2)]


class BinaryIoU:
    """IoU of TC and AR as two independent binary problems, summed over the test set (StreamSegMetrics)."""

    def __init__(self):
        self.c = np.zeros((3, 3))  # rows: TC, AR, BG; cols: tp, fp, fn
        self.per_image = []

    def update(self, tc, ar, gt):
        row = []
        for i, (p, g) in enumerate(((tc, gt == 1), (ar, gt == 2), (~(tc | ar), gt == 0))):
            v = [(p & g).sum().item(), (p & ~g).sum().item(), (~p & g).sum().item()]
            self.c[i] += v
            row += v
        self.per_image.append(row)

    def compute(self):
        iou = self.c[:, 0] / np.maximum(self.c.sum(1), 1)
        return {'TC IoU': iou[0], 'AR IoU': iou[1], 'BG IoU': iou[2], 'Mean IoU': iou.mean(), 'Mean FG IoU': iou[:2].mean()}


def masks(x, like):
    return torch.as_tensor(x).to(like.device).reshape(like.shape) > 0


@torch.no_grad()
def evaluate(sam, prompt_type, seed, device):
    ds = ClimateDataset(data_dir=DATA_DIR, train_flag=False, augmented=False, generate_prompt=True,
                        prompt_type=None if prompt_type == 'random' else prompt_type)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=ClimateDataset.collate_fn)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)  # prompt sampling happens in __getitem__
    res = {'original': BinaryIoU(), 'fixed': BinaryIoU()}
    names = []
    for batch in loader:
        batch = batch_to_cuda(batch, device)  # the thesis' own conversion (missing prompts -> None)
        x = batch['input']
        gts = [g.long() for g in batch['gt_mask']]
        names += list(batch['index_name'])
        prompts = {k: batch[k] for k in ('ar_point_prompts', 'tc_point_prompts', 'ar_bbox_prompts', 'tc_bbox_prompts')}
        sam.set_infer_img(x)
        tc_o, ar_o = sam.infer(**copy.deepcopy(prompts))
        emb, feats, _, size = sam.encode_images(x)
        tc_f, ar_f, _ = sam.forward(image_input=None, image_embeddings=emb, interm_embeddings=feats, ori_img_size=size,
                                    **copy.deepcopy(prompts))
        for b, g in enumerate(gts):
            res['original'].update(masks(tc_o[b], g), masks(ar_o[b], g), g)
            res['fixed'].update(masks(tc_f[b], g), masks(ar_f[b], g), g)
    return res, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', type=int, nargs='*', default=None, help='indices into CHECKPOINTS')
    args = ap.parse_args()
    device = torch.device('cuda')
    out = os.path.join(RESULTS, '08_thesis_recheck')
    os.makedirs(out, exist_ok=True)
    for i, (ckpt, mlp, adapter, row) in enumerate(CHECKPOINTS):
        if args.only is not None and i not in args.only:
            continue
        sam = load_climatesam(ckpt, mlp, device, adapter=adapter)
        rows, per_image = [], {}
        for prompt_type, seed in PROMPTS:
            res, names = evaluate(sam, prompt_type, seed, device)
            for path, m in res.items():
                r = {'checkpoint': ckpt, 'thesis_row': row or '', 'adapter': adapter, 'mlp_ratio': mlp,
                     'prompt': prompt_type, 'seed': seed, 'path': path, **{k: round(v, 4) for k, v in m.compute().items()}}
                if row:
                    r['thesis TC IoU'], r['thesis AR IoU'] = THESIS[row][prompt_type]
                rows.append(r)
                per_image[f'{prompt_type}_s{seed}_{path}'] = m.per_image
                print({k: r[k] for k in ('prompt', 'seed', 'path', 'TC IoU', 'AR IoU')}, flush=True)
        tag = os.path.basename(ckpt)
        with open(os.path.join(out, f'phase1_{tag}.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + (['thesis TC IoU', 'thesis AR IoU'] if not row else []))
            w.writeheader()
            w.writerows(rows)
        np.savez_compressed(os.path.join(out, f'phase1_{tag}_per_image.npz'), names=np.array(names),
                            **{k: np.array(v) for k, v in per_image.items()})
        del sam
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
