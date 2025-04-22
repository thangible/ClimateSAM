import torch
import torch.nn as nn
import torch.nn.functional as F
from module import LayerNorm2d

class PromptGenerator(nn.Module):
    def __init__(self, pool_size: tuple = (2, 2)):
        super(PromptGenerator, self).__init__()
        fused_channels = 128
        in_channels = 768
        out_channels = 2
        
        
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        
        # Create four blocks.
        # Each block processes a group of 3 features.
        # - block_feature_upsamplers: Upsample each individual feature in the group.
        #   The number of ConvTranspose2d layers increases by block.
        # - block_fuse_convs: Fuse the three upsampled features via a Conv2d.
        #   For blocks 2 and 4, we downsample spatially (stride=2) during fusion.
        # - block_out_trans: After fusion (and possible concatenation with previous block output), 
        #   upsample the fused feature for the next block.
        self.input_reduction = nn.ModuleList()
        self.block_feature_upsamplers = nn.ModuleList()
        self.block_fuse_convs = nn.ModuleList()
        self.block_out_trans = nn.ModuleList()
        
        for block_idx in range(4):
            self.input_reduction.append(
                nn.Conv2d(in_channels, fused_channels, kernel_size=1, padding=0, bias=False)
            )
            num_layers = block_idx + 1
            # Each input feature is assumed to have fused_channels channels.
            # Upsample each feature (by a factor of 2 each ConvTranspose2d)
            self.block_feature_upsamplers.append(
                nn.Sequential(*[
                    nn.ConvTranspose2d(fused_channels, fused_channels, kernel_size=2, stride=2)
                    for i in range(num_layers)
                ])
            )
            # Fuse the three features:
            # The concatenation will have 3*fused_channels channels.
            self.block_fuse_convs.append(
                nn.Conv2d(3 * fused_channels, fused_channels, kernel_size=3, padding=1, stride=2)
            )
            # After fusion, if this is not the first block, we will concatenate with the previous block's
            # upsampled output. That doubles the channels from fused_channels to 2*fused_channels.
            up_in_channels = fused_channels if block_idx == 0 else fused_channels * 2
            # Extra upsampling: always upsample by a factor of 2.
            self.block_out_trans.append(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            )
        
        # Update box_mlp to accept 4 pooled features.
        # mlp_in_dim = 4 * fused_channels * pool_size[0] * pool_size[1]
        # self.box_mlp = nn.Sequential(
        #     nn.Linear(mlp_in_dim, fused_channels),
        #     nn.ReLU(),
        #     nn.Linear(fused_channels, out_channels)
        # )
        
        # self.neck = nn.Sequential(
        #     # nn.Conv2d(
        #     #     fused_channels,
        #     #     out_channels,
        #     #     kernel_size=1,
        #     #     padding=0,
        #     #     bias=False,
        #     # ),
        #     LayerNorm2d(out_channels)            
        # )

    def forward(self, feat_list):
        """
        Args:
            feat_list: list of 12 feature maps, each of shape (B, 64, 64, 768)
        Returns:
            fused_feats: final upsampled feature from last block.
            box_out: output from box_mlp.
        """
        # Reverse the feature list and process in groups of 3.
        # Permute feature maps from (B, 64, 64, 768) to (B, 768, 64, 64)
        feat_list = [f.permute(0, 3, 1, 2) for f in feat_list]
        reversed_feats = feat_list[::-1]
        # box_queries_list = []
        prev_up = None
        for block_idx in range(4):
            
            # Get group of 3 features for this block.
            group = reversed_feats[block_idx*3:(block_idx+1)*3]
            # reduce the input channels to fused_channels.
            reduced_group = [self.input_reduction[block_idx](f) for f in group]
            # Upsample each feature using the corresponding block upsampler.
            upsampled_group = [self.block_feature_upsamplers[block_idx](f) for f in reduced_group]
            # Concatenate along the channel dimension.
            group_concat = torch.cat(upsampled_group, dim=1)  # shape: (B, 3*256, H, W)
            # Fuse the concatenated features.
            fused = self.block_fuse_convs[block_idx](group_concat)  # shape: (B, 256, H', W')
            # Save the fused feature for later pooling.
            # box_queries_list.append(fused)
            # For blocks after the first, concatenate with the previous block’s upsampled output.
            if prev_up is not None:
                fused = torch.cat([prev_up, fused], dim=1)  # shape: (B, 512, H', W')
            # Upsample fused result to feed next block.
            up = self.block_out_trans[block_idx](fused)
            prev_up = up

        # Process each saved fused feature through the adaptive pool and flatten.
        # pooled = [self.pool(feat).reshape(feat.shape[0], -1) for feat in box_queries_list]
        # concat_token = torch.cat(pooled, dim=1)
        # classifier_tokens = self.box_mlp(concat_token)
        
        # The final fused feature is the output from the last block.
        aux_mask = self.neck(prev_up)
        return aux_mask