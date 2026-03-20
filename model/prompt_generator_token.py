import torch
import torch.nn as nn
import torch.nn.functional as F
from .layer_module import LayerNorm2d

class PromptGenerator(nn.Module):
    def __init__(self, pool_size: tuple = (2, 2),
                 fused_channels: int = 128,
                 in_channels: int = 768,
                 out_channels: int = 3,
                 num_features: int = 12,
                 features_per_block: int = 3,
                 transformer_dim: int = 256):
        super(PromptGenerator, self).__init__()  
        
        # keep fused_channels accessible for token gating
        self.fused_channels = fused_channels
        
        # New: Project transformer tokens into the fusion space and produce a sigmoid gate
        self.token_projection = nn.Sequential(
            nn.Linear(transformer_dim, fused_channels),
            nn.ReLU(),
            nn.Linear(fused_channels, fused_channels),
            nn.Sigmoid()
        )
        
        # Calculate number of blocks based on total features and features per block
        self.num_blocks = num_features // features_per_block
        self.features_per_block = features_per_block
        
        if num_features % features_per_block != 0:
            raise ValueError(f"num_features ({num_features}) must be divisible by features_per_block ({features_per_block})")
             
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        # Create blocks based on calculated number
        self.input_reduction = nn.ModuleList()
        self.block_feature_upsamplers = nn.ModuleList()
        self.block_fuse_convs = nn.ModuleList()
        self.block_out_trans = nn.ModuleList()
        
        # Output convolutions for multi-level supervision
        self.multilevel_mask_convs = nn.ModuleList()
        for _ in range(self.num_blocks):
            self.multilevel_mask_convs.append(
                nn.Conv2d(fused_channels, out_channels, kernel_size=3, padding=1)
            )
        
        for block_idx in range(self.num_blocks):
            self.input_reduction.append(
                nn.Sequential(
                nn.Conv2d(in_channels, fused_channels, kernel_size=1, padding=0, bias=False),
                nn.ReLU(),
                )
            )
            num_layers = block_idx + 1
            # Each input feature is assumed to have fused_channels channels.
            # Upsample each feature (by a factor of 2 each ConvTranspose2d)
            self.block_feature_upsamplers.append(
                nn.Sequential(*[
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                    for i in range(num_layers)
                ])
            )
            # self.block_feature_upsamplers.append(
            #     nn.Sequential(*[
            #         nn.ConvTranspose2d(fused_channels, fused_channels, kernel_size=2, stride=2)
            #         for i in range(num_layers)
            #     ])
            # )
            # Fuse the three features:
            # The concatenation will have features_per_block*fused_channels channels.
            self.block_fuse_convs.append(
                nn.Sequential(
                    nn.Conv2d(features_per_block * fused_channels, fused_channels, kernel_size=1),
                    # LayerNorm2d(fused_channels),
                    # nn.ReLU(),
                    nn.Conv2d(fused_channels, fused_channels, kernel_size=3, padding=1, stride=2),
                    LayerNorm2d(fused_channels),
                    nn.ReLU()
                )
            )
            # After fusion, if this is not the first block, we will concatenate with the previous block's
            # upsampled output. That doubles the channels from fused_channels to 2*fused_channels.
            up_in_channels = fused_channels if block_idx == 0 else fused_channels * 2
            # Extra upsampling: always upsample by a factor of 2.
            self.block_out_trans.append(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            )

        self.neck = nn.Sequential(
            nn.Conv2d(self.num_blocks * fused_channels, fused_channels, kernel_size=1, padding=0),  # fused_channelsx1024x1024
            nn.ReLU()
        )
        
        self.multiclass_mask_conv = nn.Sequential(
            nn.Conv2d(fused_channels, fused_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            # Output channels set to 3 for classes 0, 1, and 2
            # Sigmoid is removed to output logits for CrossEntropyLoss
            nn.Conv2d(fused_channels, out_channels, kernel_size=3, padding=1), # Use out_channels=3
        )
        
        # self.mask1_conv =  nn.Sequential(
        #     nn.Conv2d(fused_channels, fused_channels, kernel_size=4, stride=2, groups=2, padding=1),  # 2x518x518
        #     nn.Conv2d(fused_channels, 1, kernel_size=4, stride=2,  padding=1)  # 1x256x256
        #     )
        
        # self.mask2_conv = nn.Sequential(
        #     nn.Conv2d(fused_channels, fused_channels, kernel_size=4, stride=2, groups=2, padding=1),  # 2x518x518
        #     nn.Conv2d(fused_channels, 1, kernel_size=4, stride=2, padding=1)  # 1x256x256
        #     )
    

    def forward(self, feat_list, ar_token_weight=None, tc_token_weight=None, mode='AR'):
        """
        Args:
            feat_list: list of feature maps, each of shape (B, 64, 64, 768)
            ar_token_weight: tensor (1, transformer_dim)
            tc_token_weight: tensor (1, transformer_dim)
            mode: 'AR' or 'TC' to decide which token to use
        Returns:
            multiclass_mask, intermediate_masks
        """
        # Permute feature maps from (B, 64, 64, 768) to (B, 768, 64, 64)
        feat_list = [f.permute(0, 3, 1, 2) for f in feat_list]
        intermediate_masks = []
        reversed_feats = feat_list[::-1]
        # Prepare token gate
        token = None
        feat_batch = reversed_feats[0].shape[0]
        if ar_token_weight is not None and tc_token_weight is not None:
            token = ar_token_weight if mode == 'AR' else tc_token_weight
            # ensure token has batch dimension
            if token.dim() == 1:
                token = token.unsqueeze(0)
            # Project token to fused space -> (T, fused_channels)
            gate = self.token_projection(token)  # (T, C)
            gate = gate.view(gate.size(0), self.fused_channels, 1, 1)  # (T, C, 1, 1)
            # If single token provided, expand to batch size for broadcasting
            if gate.size(0) == 1 and feat_batch > 1:
                gate = gate.expand(feat_batch, self.fused_channels, 1, 1)
            # If token batch matches feature batch, keep as-is. Otherwise, fallback to first token expanded.
            if gate.size(0) != feat_batch and gate.size(0) != 1:
                gate = gate[0:1].expand(feat_batch, self.fused_channels, 1, 1)
            gate = gate.to(reversed_feats[0].device)
        else:
            # fallback: no gating (all ones)
            gate = torch.ones(feat_batch, self.fused_channels, 1, 1, 
                  device=reversed_feats[0].device, 
                  dtype=reversed_feats[0].dtype)

        prev_up = None
        for block_idx in range(self.num_blocks):
            # Get group of features for this block.
            start_idx = block_idx * self.features_per_block
            end_idx = (block_idx + 1) * self.features_per_block
            group = reversed_feats[start_idx:end_idx]
            # reduce the input channels to fused_channels.
            reduced_group = [self.input_reduction[block_idx](f) for f in group]

            # Apply Token Gating to each reduced feature map in the group
            gated_group = [f * gate for f in reduced_group]

            # Upsample each feature using the corresponding block upsampler.
            upsampled_group = [self.block_feature_upsamplers[block_idx](f) for f in gated_group]
            # Concatenate along the channel dimension.
            group_concat = torch.cat(upsampled_group, dim=1)  # shape: (B, 3*256, H, W)
            # Fuse the concatenated features.
            fused = self.block_fuse_convs[block_idx](group_concat)  # shape: (B, 256, H', W')
            # ----------------------------------------------------
            # 1. Compute intermediate mask (logits)
            # Use the fused feature for the output convolution for this level
            intermediate_logit = self.multilevel_mask_convs[block_idx](fused)
            # No interpolation here; it will be done on the loss side
            intermediate_masks.append(intermediate_logit)

            # For blocks after the first, concatenate with the previous block’s upsampled output.
            if prev_up is not None:
                fused = torch.cat([prev_up, fused], dim=1)  # shape: (B, 512, H', W')
            # Upsample fused result to feed next block.
            up = self.block_out_trans[block_idx](fused)
            prev_up = up

        # The final fused feature is the output from the last block.
        neck_out = self.neck(prev_up)
        # Compute the multi-class mask (logits)
        multiclass_mask = self.multiclass_mask_conv(neck_out)

        # Interpolate the mask (B, 3, H, W)
        multiclass_mask = F.interpolate(multiclass_mask, size=(768, 1152), mode='bilinear', align_corners=False)

        return multiclass_mask, intermediate_masks

