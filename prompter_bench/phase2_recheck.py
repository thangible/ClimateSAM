"""
Re-evaluation of the thesis' own Phase-2 prompter checkpoints (exp/best_weights/best_*.pth) on the benchmark
(prompter mask + every prompt conversion, object metrics, error decomposition, per-image counts).

Each checkpoint is evaluated with the frozen encoder it was trained with (from its wandb config) and with its own
decision rule: 3-class softmax, argmax = prompter mask (train_generator*.py / train_logistic_regression.py).
Dense prompts use the per-class log-odds of the softmax.
"""
import os
import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import ROOT, RESULTS, IMG_H, IMG_W, FeatureCache, load_climatesam
from build_cache import ENCODERS
from train import Batches
from evaluate import evaluate_method
from prompters import refined_tokens
from model.prompt_generator import PromptGenerator as MSF
from model.prompt_generator_token import PromptGenerator as TokenMSF
from model.prompt_generator_token_cgblock import PromptGenerator as TokenCGMSF
from model.prompt_generator_sp_token import PromptGenerator as SPTokenMSF
from model.logistic_regression_prompter import LogisticRegressionPrompter

W = os.path.join(ROOT, 'exp', 'best_weights')
# name -> (class, fused channels, token-gated, encoder tag, checkpoint, wandb run name, prompt type used in training)
USER = {
    'user_msf_Generator': (MSF, 128, False, 'infused_mlp1_best', 'best_generator_vit_b_128_Generator.pth', 'Generator', 'point'),
    'user_msf_bbox_128': (MSF, 128, False, 'infused_05_retrain', 'best_generator_vit_b_128_generator_128_vit_b_bbox.pth',
                          'generator_128_vit_b_bbox', 'bbox'),
    'user_token_FIRST': (TokenMSF, 128, True, 'infused_mlp1_best', 'best_generator_token_vit_b_128_FIRST.pth', 'FIRST', 'point'),
    'user_token_cg_CG': (TokenCGMSF, 128, True, 'infused_mlp1_best', 'best_generator_token_cg_vit_b_128_CG.pth', 'CG', 'point'),
    'user_token_cg_CG_256': (TokenCGMSF, 256, True, 'infused_mlp1_best', 'best_generator_token_cg_vit_b_256_CG_256.pth', 'CG_256', 'bbox'),
    'user_token_cg_CG_256_REAL': (TokenCGMSF, 256, True, 'infused_mlp1_best', 'best_generator_token_cg_vit_b_256_CG_256_REAL.pth',
                                  'CG_256_REAL', 'bbox'),
    'user_sp_token_SP_TOKEN': (SPTokenMSF, 64, True, 'infused_mlp1_best', 'best_generator_token_cg_vit_b_64_SP_TOKEN.pth', 'SP_TOKEN', 'bbox'),
    'user_logreg_Logistic': (LogisticRegressionPrompter, None, False, 'infused_mlp1_best', 'best_logistic_regression_vit_b_Logistic.pth',
                             'Logistic', 'bbox'),
}


class UserPrompter(nn.Module):
    def __init__(self, net, token, climatesam):
        super().__init__()
        self.net, self.token = net, token
        if token:
            t = refined_tokens(climatesam)
            self.register_buffer('ar_refined', t['AR'])
            self.register_buffer('tc_refined', t['TC'])

    def forward(self, batch):
        feats = [batch['vit'][l] for l in range(12)]
        if self.token:
            final = self.net(feats, ar_refined=self.ar_refined, tc_refined=self.tc_refined)[0]
        else:
            final = self.net(feats)[0]
        final = F.interpolate(final.float(), (IMG_H, IMG_W), mode='bilinear', align_corners=False)
        logp = F.log_softmax(final, dim=1)
        odds = torch.stack([logp[:, c] - torch.log1p(-logp[:, c].exp().clamp(max=1 - 1e-6)) for c in (1, 2)], dim=1)
        return {'logits': odds, 'argmax': logp.argmax(1), 'aux': []}


def build(name, climatesam, device):
    cls, fused, token, _, ckpt, _, _ = USER[name]
    state = torch.load(os.path.join(W, ckpt), map_location='cpu')['prompt_generator']
    if fused is None:
        net = cls(in_channels=768, out_channels=3, interpolation_mode='nearest')
        net.load_state_dict(state, strict=True)
    else:  # the channel width is taken from the weights (one wandb config disagrees with its checkpoint)
        for width in (fused, 64, 128, 256):
            net = cls(in_channels=768, fused_channels=width, num_features=12, features_per_block=3)
            try:
                net.load_state_dict(state, strict=True)
                break
            except RuntimeError:
                continue
        else:
            raise RuntimeError(f'{name}: no channel width matches the checkpoint')
        if width != fused:
            print(f'{name}: checkpoint has fused_channels={width} (config said {fused})', flush=True)
    return UserPrompter(net, token, climatesam).to(device).eval()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--names', nargs='+', default=list(USER))
    args = ap.parse_args()
    device = torch.device('cuda')
    by_encoder = {}
    for n in args.names:
        by_encoder.setdefault(USER[n][3], []).append(n)
    for enc, names in by_encoder.items():
        climatesam = load_climatesam(*ENCODERS[enc], device)
        data = Batches(FeatureCache(enc, 'test'), list(range(12)), device)
        out_dir = os.path.join(RESULTS, 'eval', enc)
        for n in names:
            model = build(n, climatesam, device)

            def predict(b, model=model):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    return model(b)['logits']
            evaluate_method(n, predict, data, climatesam, device, out_dir, hard_masks=lambda b, model=model: model(b)['argmax'])
            print('done', n, flush=True)
            del model
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
