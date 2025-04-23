import torch
from torch import nn
from .layer_module import Adapter

class SAMImageEncodeWrapper(nn.Module):

    def __init__(self, ori_sam, fix: bool = True):
        super(SAMImageEncodeWrapper, self).__init__()
        self.sam_img_encoder = ori_sam.image_encoder
        if fix:
            for name, param in self.sam_img_encoder.named_parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.sam_img_encoder(x)
        return x
      

class ClimateSAMImageEncoder(SAMImageEncodeWrapper):

    def __init__(
            self, ori_sam, hq_token: torch.Tensor, fix: bool = True,
    ):
        super(ClimateSAMImageEncoder, self).__init__(ori_sam=ori_sam, fix=True)
        
        self.hq_token = hq_token

        total_p_layer = len(self.sam_img_encoder.blocks)
        prompt_dim = self.sam_img_encoder.pos_embed.shape[-1]
        self.hq_token_proj = nn.Sequential(
            *[Adapter(hq_token.size(-1), prompt_dim, mlp_ratio=0.25) for _ in range(total_p_layer)]
        )


    def forward(self, x):
        x = self.sam_img_encoder.patch_embed(x)
        if self.sam_img_encoder.pos_embed is not None:
            x = x + self.sam_img_encoder.pos_embed

        hq_prompt_tokens = []
        for i in range(0, len(self.hq_token_proj)):
            hq_prompt_tokens.append(self.hq_token_proj[i](self.hq_token).unsqueeze(0))

        interm_embeddings = []
        for i, blk in enumerate(self.sam_img_encoder.blocks):
            x = blk(x, hq_prompt_tokens[i])
            interm_embeddings.append(x)

        x = self.sam_img_encoder.neck(x.permute(0, 3, 1, 2))
        return x, interm_embeddings