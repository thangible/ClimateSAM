"""
Re-run of Table 4.13 (SAM with CG-Net as prompter), with the original prompt pipeline (PromptMaker +
ClimateSAM.infer / forward) and two image paths:

  original : ClimateSAM.set_infer_img() -> first 3 raw channels, the learned input adapter is skipped
             (this is what train_script/official/test_prompt_effect.py does)
  fixed    : ClimateSAM.encode_images() -> 16 channels through the input adapter, as in training

Same 61 test images, same CG-Net checkpoint (exp/cgnet_weight.pth), same prompt settings as the table.
"""
import os
import csv
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import RESULTS, DATA_DIR, SegMetrics, load_climatesam, seed_everything
from dataset.climatenet import ClimateDataset
from model.prompt.prompt_maker import PromptMaker
import prompters

# (prompt types, positive points, negative points, enlarge ratio) -- the rows of Table 4.13
CONFIGS = [
    (('bbox',), 0, 0, 0.0), (('point', 'bbox'), 10, 5, 0.0), (('point', 'bbox'), 15, 10, 0.0),
    (('bbox',), 0, 0, 0.1), (('bbox',), 0, 0, -0.1), (('point',), 5, 10, 0.0), (('point',), 5, 5, 0.0),
    (('point',), 1, 3, 0.0), (('point',), 10, 10, 0.0), (('point',), 1, 1, 0.0), (('point',), 1, 2, 0.0),
    (('point',), 2, 2, 0.0), (('bbox',), 0, 0, 0.2), (('point', 'bbox', 'mask'), 20, 10, 0.0), (('bbox',), 0, 0, 0.3),
    (('bbox',), 0, 0, -0.2), (('point', 'bbox', 'mask'), 10, 5, 0.0), (('bbox',), 0, 0, 0.5), (('mask',), 0, 0, 0.0),
]
ENCODERS = [
    ('infused_token_vit_b_1.0_infused_token_vit_b_mlp1_CORRECTED', 1.0),
    ('best_weights/infused_token_vit_b_1.0_infused_token_vit_b_mlp1_CORRECTED_NOSMOOTH', 1.0),
]
KEYS = ['ar_point_prompts', 'tc_point_prompts', 'ar_bbox_prompts', 'tc_bbox_prompts', 'ar_mask_prompts', 'tc_mask_prompts']


def config_name(cfg):
    types, p, n, e = cfg
    return f"{'+'.join(types)} pos={p} neg={n} enlarge={e}"


def build_prompts(maker, aux, cfg, device):
    """Merge the per-type PromptMaker outputs, as test_prompt_effect.validate_with_combined_prompts does."""
    types, p, n, e = cfg
    merged = {k: [None] * len(aux) for k in KEYS}
    for t in types:
        d = maker.make_prompts(multiclass_mask=aux, prompt_type=t, positive_point_num=max(p, 1), negative_point_num=n,
                               enlarge_ratio=e)
        for k in KEYS:
            if (t == 'point' and 'point' in k) or (t == 'bbox' and 'bbox' in k) or (t == 'mask' and 'mask' in k):
                merged[k] = d[k]
    for k in KEYS:
        merged[k] = [(tuple(x.to(device) for x in v) if isinstance(v, tuple) and v[0] is not None else
                      (v.to(device) if torch.is_tensor(v) else None)) for v in merged[k]]
        if 'point' in k:
            merged[k] = [v if v is not None else None for v in merged[k]]
    return merged


@torch.no_grad()
def main():
    device = torch.device('cuda')
    ds = ClimateDataset(data_dir=DATA_DIR, train_flag=False, augmented=False, generate_prompt=False)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=4, collate_fn=ClimateDataset.collate_fn)
    cgnet = prompters.build('cgnet_finetuned')[0].to(device).eval()
    maker = PromptMaker()
    rows = []
    for ckpt, mlp in ENCODERS:
        sam = load_climatesam(ckpt, mlp, device)
        metrics = {(path, i): SegMetrics() for path in ('original', 'fixed') for i in range(len(CONFIGS))}
        baseline = SegMetrics()
        for step, batch in enumerate(tqdm(loader, desc=ckpt, leave=False)):
            x = batch['input'].to(device)
            aux = cgnet({'cgnet': batch['cgnet_input'].to(device)})['argmax']
            gts = [g.to(device) for g in batch['gt_mask']]
            for b, g in enumerate(gts):
                baseline.update(aux[b] == 1, aux[b] == 2, g)
            emb, feats, _, size = sam.encode_images(x)
            sam.set_infer_img(x)
            for i, cfg in enumerate(CONFIGS):
                seed_everything(1000 * step + i)
                prompts = build_prompts(maker, aux, cfg, device)
                tc_o, ar_o = sam.infer(**copy.deepcopy(prompts))
                tc_f, ar_f, _ = sam.forward(image_input=None, image_embeddings=emb, interm_embeddings=feats,
                                            ori_img_size=size, **copy.deepcopy(prompts))
                for b, g in enumerate(gts):
                    metrics[('original', i)].update(tc_o[b].view_as(g) > 0, ar_o[b].view_as(g) > 0, g)
                    metrics[('fixed', i)].update(tc_f[b].view_as(g) > 0, ar_f[b].view_as(g) > 0, g)
        rows.append({'encoder': ckpt, 'path': '-', 'prompt': 'CG-Net alone (baseline)',
                     **{k: round(v, 4) for k, v in baseline.compute().items()}})
        for i, cfg in enumerate(CONFIGS):
            for path in ('original', 'fixed'):
                rows.append({'encoder': ckpt, 'path': path, 'prompt': config_name(cfg),
                             **{k: round(v, 4) for k, v in metrics[(path, i)].compute().items()}})
                print(rows[-1], flush=True)
        del sam
        torch.cuda.empty_cache()

    out = os.path.join(RESULTS, '02_table_4_13_recheck')
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'table_4_13_recheck.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()
