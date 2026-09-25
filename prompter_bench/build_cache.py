"""Encode the ClimateNet train/test sets once per frozen ClimateSAM checkpoint."""
import sys
import torch
from common import build_feature_cache, load_climatesam

ENCODERS = {
    'infused_mlp1': ('infused_token_vit_b_1.0_infused_token_vit_b_mlp1_CORRECTED', 1.0),
    'infused_mlp05': ('infused_token_vit_b_0.5_infused_token_vit_b_mlp05_CORRECTED', 0.5),
}

if __name__ == '__main__':
    for tag in sys.argv[1:]:
        ckpt, mlp = ENCODERS[tag]
        build_feature_cache(load_climatesam(ckpt, mlp, torch.device('cuda')), tag, torch.device('cuda'))
        print(f'cache {tag} done')
