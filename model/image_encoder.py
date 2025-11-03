import torch
import torch.utils.checkpoint as checkpoint
from torch import nn
from .layer_module  import Adapter

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
            self, 
            ori_sam,
            hq_token_ar: torch.Tensor, 
            hq_token_tc: torch.Tensor,
            fix: bool = True, 
            mlp_ratio=0.25,
            use_checkpoint: bool = True  # Add this parameter
    ):
        super(ClimateSAMImageEncoder, self).__init__(ori_sam=ori_sam, fix=True)

        self.use_checkpoint = use_checkpoint
        self.hq_token_ar = hq_token_ar
        self.hq_token_tc = hq_token_tc

        total_p_layer = len(self.sam_img_encoder.blocks)
        prompt_dim = self.sam_img_encoder.pos_embed.shape[-1]
        self.hq_token_proj_ar = nn.Sequential(
            *[Adapter(hq_token_ar.size(-1), prompt_dim//2, mlp_ratio=mlp_ratio) for _ in range(total_p_layer)]
        )
        self.hq_token_proj_tc = nn.Sequential(
            *[Adapter(hq_token_tc.size(-1), prompt_dim//2, mlp_ratio=mlp_ratio) for _ in range(total_p_layer)]
        )

    def _checkpoint_block(self, block, x, hq_prompt_tokens):
        """Wrapper function for checkpointing SAM blocks"""
        def custom_forward(x_input, prompt_tokens):
            return block(x_input, prompt_tokens)
        
        return checkpoint.checkpoint(
            custom_forward, 
            x, 
            hq_prompt_tokens, 
            use_reentrant=False
        )

    def forward(self, x):
        x = self.sam_img_encoder.patch_embed(x)
        if self.sam_img_encoder.pos_embed is not None:
            x = x + self.sam_img_encoder.pos_embed

        hq_prompt_tokens_ar = []
        hq_prompt_tokens_tc = []
        for i in range(0, len(self.hq_token_proj_ar)):
            hq_prompt_tokens_ar.append(self.hq_token_proj_ar[i](self.hq_token_ar).unsqueeze(0))
        for i in range(0, len(self.hq_token_proj_tc)):
            hq_prompt_tokens_tc.append(self.hq_token_proj_tc[i](self.hq_token_tc).unsqueeze(0))

        interm_embeddings = []
        for i, blk in enumerate(self.sam_img_encoder.blocks):
            hq_prompt_tokens = torch.cat((hq_prompt_tokens_ar[i], hq_prompt_tokens_tc[i]), dim=-1)
            
            # Use gradient checkpointing only during training
            if self.use_checkpoint and self.training:
                x = self._checkpoint_block(blk, x, hq_prompt_tokens)
            else:
                x = blk(x, hq_prompt_tokens)
            
            interm_embeddings.append(x)

        x = self.sam_img_encoder.neck(x.permute(0, 3, 1, 2))
        return x, interm_embeddings
    
    def train(self, mode: bool = True):
        # Set the entire module to training/eval mode
        super().train(mode)
        if mode:
            # training: turn the modules of original SAM mask decoder to eval mode
            for n, c in self.named_children():
                if n in ['sam_img_encoder']:
                    c.eval()
                    for name, param in c.named_parameters():
                        param.requires_grad = False
                else:
                    c.train()
                    for name, param in c.named_parameters():
                        param.requires_grad = True
        else:
            # eval:
            for module in self.children():
                module.train(mode)
                for name, param in module.named_parameters():
                    param.requires_grad = False
