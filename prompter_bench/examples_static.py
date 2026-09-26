"""Example masks of the learned static prompts (same test images as evaluate.EXAMPLES), for the qualitative figures."""
import os
import numpy as np
import torch

from common import RESULTS, CLASSES, FeatureCache, load_climatesam, upsample
from build_cache import ENCODERS
from evaluate import EXAMPLES
from train_learned_prompt import StaticPrompts, decode


@torch.no_grad()
def main(encoder='infused_mlp1', run='learned_prompt_k4_s0'):
    device = torch.device('cuda')
    sam = load_climatesam(*ENCODERS[encoder], device)
    tokens = torch.load(os.path.join(RESULTS, 'runs', encoder, run, 'best.pth'))['tokens'].to(device)
    prompts = StaticPrompts(tokens[0, 0], tokens.shape[1]).to(device)
    prompts.tokens.data.copy_(tokens)
    data = FeatureCache(encoder, 'test').load_to(device, [0])
    out = {}
    for i in EXAMPLES:
        emb, interm0 = data['emb'][i:i + 1].float(), data['vit'][0][i:i + 1].float()
        out[f'{i}__gt'] = data['gt'][i].cpu().numpy()
        for k, (cls, _) in enumerate(CLASSES):
            out[f'{i}__sam_static_{cls}'] = (upsample(decode(sam, prompts, emb, interm0, k, cls)[0].float()) > 0).cpu().numpy()
    np.savez_compressed(os.path.join(RESULTS, 'eval', encoder, f'{run}_examples.npz'), **out)


if __name__ == '__main__':
    main()
