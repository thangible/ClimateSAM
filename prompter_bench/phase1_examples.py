"""Example masks (box prompts from the ground truth) and input-adapter outputs of the Phase-1 checkpoints, for the maps
of results/08_thesis_recheck (original evaluation path without the input adapter vs. with it)."""
import os
import copy
import random

import numpy as np
import torch
import torch.nn.functional as F

from common import RESULTS, DATA_DIR, load_climatesam
from dataset.climatenet import ClimateDataset
from utility import batch_to_cuda

IMAGES = (17, 45)
MODELS = [('linear_1.0', 'infused_token_vit_b_1.0_infused_token_vit_b_mlp1_CORRECTED', 1.0, 'linear'),
          ('nonlinear_1.0', 'infused_token_vit_b_1.0_infused_token_vit_b_mlp1_nonlinear', 1.0, 'nonlinear'),
          ('nonlinear_0.5', 'infused_token_vit_b_0.5_infused_token_vit_b_mlp05_nonlinear', 0.5, 'nonlinear')]


@torch.no_grad()
def main():
    device = torch.device('cuda')
    ds = ClimateDataset(data_dir=DATA_DIR, train_flag=False, augmented=False, generate_prompt=True, prompt_type='bbox')
    out = {}
    for key, ckpt, mlp, adapter in MODELS:
        sam = load_climatesam(ckpt, mlp, device, adapter=adapter)
        for img in IMAGES:
            random.seed(img); np.random.seed(img)
            batch = batch_to_cuda(ClimateDataset.collate_fn([ds[img]]), device)
            x = batch['input']
            gt = batch['gt_mask'][0].long()
            prompts = {k: batch[k] for k in ('ar_point_prompts', 'tc_point_prompts', 'ar_bbox_prompts', 'tc_bbox_prompts')}
            boxes = {c: (batch[f'{c.lower()}_bbox_prompts'][0].reshape(-1, 4).cpu().numpy()
                         if batch[f'{c.lower()}_bbox_prompts'][0] is not None else np.zeros((0, 4))) for c in ('TC', 'AR')}
            sam.set_infer_img(x)
            tc_o, ar_o = sam.infer(**copy.deepcopy(prompts))
            emb, feats, _, size = sam.encode_images(x)
            tc_f, ar_f, _ = sam.forward(image_input=None, image_embeddings=emb, interm_embeddings=feats, ori_img_size=size,
                                        **copy.deepcopy(prompts))
            to = lambda m: (torch.as_tensor(m[0]).reshape(gt.shape) > 0).cpu().numpy()
            out[f'{img}__gt'] = gt.cpu().numpy().astype(np.uint8)
            for c in ('TC', 'AR'):
                out[f'{img}__boxes_{c}'] = boxes[c]
            out[f'{img}__{key}_original_TC'], out[f'{img}__{key}_original_AR'] = to(tc_o), to(ar_o)
            out[f'{img}__{key}_fixed_TC'], out[f'{img}__{key}_fixed_AR'] = to(tc_f), to(ar_f)
            # what the encoder sees: raw first three channels (original path) vs. the adapter output (training path)
            raw = x[0, :3].float().cpu().numpy()
            adapted = sam.input_adapter(x.float())[0].float().cpu().numpy()
            out[f'{img}__raw_rgb'] = raw
            out[f'{img}__{key}_adapter_rgb'] = adapted
        del sam
        torch.cuda.empty_cache()
    np.savez_compressed(os.path.join(RESULTS, '08_thesis_recheck', 'phase1_examples.npz'), **out)
    print('saved', len(out), 'arrays')


if __name__ == '__main__':
    main()
