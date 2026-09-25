import math
import torch
import torch.nn as nn
from .layer_module import LayerNorm2d


class ConvNeXtBlock(nn.Module):
    """Depthwise 3x3 (optionally dilated) + pointwise MLP, with a residual connection."""

    def __init__(self, channels: int, dilation: int = 1, mlp_ratio: int = 4):
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, groups=channels)
        self.norm = LayerNorm2d(channels)
        self.pwconv1 = nn.Conv2d(channels, channels * mlp_ratio, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(channels * mlp_ratio, channels, kernel_size=1)

    def forward(self, x):
        return x + self.pwconv2(self.act(self.pwconv1(self.norm(self.dwconv(x)))))


class MaskPromptGenerator(nn.Module):
    """
    Minimal automatic prompter for ClimateSAM.

    Works on the frozen encoder features at their native 64x64 resolution and outputs
    one 256x256 logit map per class (channel 0 = TC, channel 1 = AR). The sigmoid of this
    map is fed to SAM's prompt encoder as a dense mask prompt, which keeps the whole
    prompt -> mask path differentiable so the generator can be trained on SAM's output.
    """

    def __init__(self,
                 embed_dim: int = 256,
                 vit_dim: int = 768,
                 num_vit_feats: int = 2,
                 channels: int = 128,
                 num_blocks: int = 3,
                 num_classes: int = 2,
                 prior_prob: float = 0.05):
        super().__init__()

        self.embed_proj = nn.Conv2d(embed_dim, channels, kernel_size=1)
        self.vit_proj = nn.Conv2d(vit_dim * num_vit_feats, channels, kernel_size=1)

        # growing dilation widens the receptive field for long, ribbon-like ARs
        self.blocks = nn.Sequential(*[ConvNeXtBlock(channels, dilation=2 ** i) for i in range(num_blocks)])

        # 64x64 -> 256x256 (SAM's mask-prompt resolution)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(channels, channels // 2, kernel_size=2, stride=2),
            LayerNorm2d(channels // 2),
            nn.GELU(),
            nn.ConvTranspose2d(channels // 2, channels // 4, kernel_size=2, stride=2),
            nn.GELU(),
        )
        self.head = nn.Conv2d(channels // 4, num_classes, kernel_size=3, padding=1)

        # start from a low foreground prior so the initial mask prompts are nearly empty
        nn.init.constant_(self.head.bias, -math.log((1 - prior_prob) / prior_prob))

    def forward(self, image_embeddings, vit_feats):
        """
        Args:
            image_embeddings: (B, 256, 64, 64) SAM neck output
            vit_feats: list of (B, 64, 64, vit_dim) intermediate ViT features
        Returns:
            (B, num_classes, 256, 256) logits
        """
        vit = torch.cat([f.permute(0, 3, 1, 2) for f in vit_feats], dim=1)
        x = self.embed_proj(image_embeddings) + self.vit_proj(vit)
        x = self.blocks(x)
        return self.head(self.upsample(x))
