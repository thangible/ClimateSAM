"""Encode the ClimateNet train/test sets once per frozen ClimateSAM checkpoint."""
import sys
import torch
from common import build_feature_cache, load_climatesam

ENCODERS = {
    'infused_mlp1': ('infused_token_vit_b_1.0_infused_token_vit_b_mlp1_CORRECTED', 1.0),
    'infused_mlp05': ('infused_token_vit_b_0.5_infused_token_vit_b_mlp05_CORRECTED', 0.5),
    # encoders the thesis' own Phase-2 prompters were trained with (test split only, for re-evaluation)
    'infused_mlp1_best': ('best_weights/infused_token_vitb_mlp1_best', 1.0),
    'infused_05_retrain': ('best_weights/infused_token_vit_b_0.5_retrain_infused_05_00-005', 0.5),
}
TEST_ONLY = {'infused_mlp1_best', 'infused_05_retrain'}

if __name__ == '__main__':
    for tag in sys.argv[1:]:
        ckpt, mlp = ENCODERS[tag]
        build_feature_cache(load_climatesam(ckpt, mlp, torch.device('cuda')), tag, torch.device('cuda'),
                            splits=('test',) if tag in TEST_ONLY else ('train', 'test'))
        print(f'cache {tag} done')
