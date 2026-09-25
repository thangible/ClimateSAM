"""
Every prompter behind one interface: forward(batch) -> dict with
    'logits': (B, 2, h, w) TC / AR logits (any resolution, upsampled for loss and evaluation)
    'aux':    list of (B, 2, h, w) deep-supervision logits (multi-scale fusion only)

All learned prompters use two sigmoid channels (TC, AR). The original multi-scale generators had a third
"background" channel that no loss ever touched but that still took part in the final argmax; here it is removed.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import ROOT, VIT_DIM, NUM_LAYERS
from model.prompt_generator import PromptGenerator as MultiScaleFusion
from model.prompt_generator_token import PromptGenerator as TokenGatedMultiScaleFusion
from model.mask_prompt_generator import MaskPromptGenerator
from model.prompt.cgnet_module import CGNetModule


class LogisticRegression(nn.Module):
    """Pixel-wise linear classifier on one ViT layer (the thesis prompter used layer 0)."""

    def __init__(self, vit_dim, layer):
        super().__init__()
        self.layer = layer
        self.classifier = nn.Conv2d(vit_dim, 2, kernel_size=1)

    def forward(self, batch):
        return {'logits': self.classifier(batch['vit'][self.layer].permute(0, 3, 1, 2)), 'aux': []}


class MSF(nn.Module):
    """Multi-scale fusion generator (thesis Figure 3.9) with a 2-channel head."""

    def __init__(self, vit_dim, num_layers, fused_channels=128):
        super().__init__()
        self.num_layers = num_layers
        self.net = MultiScaleFusion(in_channels=vit_dim, fused_channels=fused_channels, out_channels=2,
                                    num_features=num_layers, features_per_block=num_layers // 4)

    def forward(self, batch):
        logits, aux = self.net([batch['vit'][l] for l in range(self.num_layers)])
        return {'logits': logits, 'aux': aux}


class MSFToken(nn.Module):
    """
    Multi-scale fusion + token gate: the decoder's refined TC / AR HQ tokens gate the fused features of two
    binary heads (these heads produce the prompts). The multiclass head is kept as an auxiliary output.
    """

    def __init__(self, vit_dim, num_layers, refined_tokens, fused_channels=128):
        super().__init__()
        self.num_layers = num_layers
        self.net = TokenGatedMultiScaleFusion(in_channels=vit_dim, fused_channels=fused_channels, out_channels=2,
                                              num_features=num_layers, features_per_block=num_layers // 4)
        self.register_buffer('ar_refined', refined_tokens['AR'])
        self.register_buffer('tc_refined', refined_tokens['TC'])

    def forward(self, batch):
        multiclass, aux, ar, tc = self.net([batch['vit'][l] for l in range(self.num_layers)],
                                           ar_refined=self.ar_refined, tc_refined=self.tc_refined)
        return {'logits': torch.cat([tc, ar], dim=1), 'aux': aux + [multiclass]}


class MaskPrompt(nn.Module):
    """Small head on the SAM neck embedding + two ViT layers (model/mask_prompt_generator.py)."""

    def __init__(self, vit_dim, layers, channels=128, num_blocks=3):
        super().__init__()
        self.layers = layers
        self.net = MaskPromptGenerator(vit_dim=vit_dim, num_vit_feats=len(layers), channels=channels, num_blocks=num_blocks)

    def forward(self, batch):
        return {'logits': self.net(batch['emb'], [batch['vit'][l] for l in self.layers]), 'aux': []}


class CGNet(nn.Module):
    """
    ClimateNet CG-Net on the raw TMQ/U850/V850/PSL fields (3-class softmax -> per-class log-odds).
    batch_stats: the official checkpoint's BatchNorm running statistics do not match its weights (with them it
    predicts background everywhere), so it is run with batch statistics (momentum 0: nothing is updated).
    """

    def __init__(self, weights, batch_stats=False):
        super().__init__()
        self.net = CGNetModule(classes=3, channels=4)
        self.net.load_state_dict(torch.load(weights, map_location='cpu'))
        self.bns = [m for m in self.net.modules() if isinstance(m, nn.BatchNorm2d)] if batch_stats else []
        for bn in self.bns:
            bn.momentum = 0.0

    def forward(self, batch):
        for bn in self.bns:
            bn.train()
        logp = F.log_softmax(self.net(batch['cgnet']).float(), dim=1)
        # log-odds of "class c" vs "not class c"
        odds = torch.stack([logp[:, c] - torch.log1p(-logp[:, c].exp().clamp(max=1 - 1e-6)) for c in (1, 2)], dim=1)
        return {'logits': odds, 'aux': [], 'argmax': logp.argmax(1)}


def refined_tokens(climatesam):
    dec = climatesam.mask_decoder
    with torch.no_grad():
        return {'AR': dec.hf_mlp_ar(dec.hf_token_ar.weight).detach(), 'TC': dec.hf_mlp_tc(dec.hf_token_tc.weight).detach()}


def build(arch, sam_type='vit_b', climatesam=None):
    """Returns (model, ViT layers the model reads)."""
    D, L = VIT_DIM[sam_type], NUM_LAYERS[sam_type]
    if arch == 'logreg_l0':
        return LogisticRegression(D, 0), [0]
    if arch == 'logreg_last':
        return LogisticRegression(D, L - 1), [L - 1]
    if arch == 'msf':
        return MSF(D, L), list(range(L))
    if arch == 'msf_token':
        return MSFToken(D, L, refined_tokens(climatesam)), list(range(L))
    if arch == 'mpg':
        layers = [L // 2 - 1, L - 1]
        return MaskPrompt(D, layers), layers
    if arch == 'cgnet_official':
        return CGNet(os.path.join(ROOT, 'pretrained', 'weights_cgnet.pth'), batch_stats=True), []
    if arch == 'cgnet_finetuned':
        return CGNet(os.path.join(ROOT, 'exp', 'cgnet_weight.pth')), []
    raise ValueError(arch)
