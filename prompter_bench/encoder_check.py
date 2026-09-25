"""Ground-truth prompt upper bound of every candidate Phase-1 checkpoint (used to pick the frozen SAM for Section 4.3)."""
import os
import csv
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import (RESULTS, DATA_DIR, CLASSES, SegMetrics, load_climatesam, sam_decode, union_logits, upsample,
                    make_prompts, seed_everything)
from dataset.climatenet import ClimateDataset

CANDIDATES = [
    ('best_weights/infused_token_vit_b_0.5_retrain_infused_05_00-005', 0.5),
    ('infused_token_vit_b_0.5_infused_token_vit_b_mlp05_CORRECTED', 0.5),
    ('infused_token_vit_b_0.5_retrain_infused_05_bbox', 0.5),
    ('best_weights/infused_token_vitb_mlp1_best', 1.0),
    ('infused_token_vit_b_1.0_infused_token_vit_b_mlp1_CORRECTED', 1.0),
    ('best_weights/infused_token_vit_b_1.0_infused_token_vit_b_mlp1_CORRECTED_NOSMOOTH', 1.0),
    ('infused_token_vit_b_1.0_best_only_bbox', 1.0),
]
KINDS = ['bbox', 'point', 'mask']


@torch.no_grad()
def main():
    device = torch.device('cuda')
    ds = ClimateDataset(data_dir=DATA_DIR, train_flag=False, augmented=False, generate_prompt=False)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=8, collate_fn=ClimateDataset.collate_fn)
    rows = []
    for ckpt, mlp in CANDIDATES:
        seed_everything(0)
        sam = load_climatesam(ckpt, mlp, device)
        metrics = {k: SegMetrics() for k in KINDS}
        for batch in tqdm(loader, desc=ckpt, leave=False):
            emb, feats, _, _ = sam.encode_images(batch['input'].to(device))
            for b in range(len(emb)):
                gt = batch['gt_mask'][b].to(device)
                for kind in KINDS:
                    preds = {}
                    for cls, label in CLASSES:
                        prompts = make_prompts((gt == label).cpu().numpy().astype('uint8'), kind, device=device)
                        logits = sam_decode(sam, emb[b:b + 1], feats[0][b:b + 1], cls, **prompts) if prompts else None
                        preds[cls] = upsample(union_logits(logits).to(device)) > 0
                    metrics[kind].update(preds['TC'], preds['AR'], gt)
        for kind in KINDS:
            r = metrics[kind].compute()
            rows.append({'checkpoint': ckpt, 'mlp_ratio': mlp, 'gt_prompt': kind, **{k: round(v, 4) for k, v in r.items()}})
            print(rows[-1])
        del sam
        torch.cuda.empty_cache()

    out = os.path.join(RESULTS, '00_setup')
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'encoder_candidates_gt_prompts.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()
