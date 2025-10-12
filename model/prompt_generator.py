import torch
import torch.nn as nn
import torch.nn.functional as F
from .layer_module import LayerNorm2d

class PromptGenerator(nn.Module):
    def __init__(self, pool_size: tuple = (2, 2),
                 fused_channels: int = 128,
                 in_channels: int = 768,
                 out_channels: int = 2,
                 num_features: int = 12,
                 features_per_block: int = 3):
        super(PromptGenerator, self).__init__()  
        
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
        
        self.mask1_conv = nn.Sequential(
            nn.Conv2d(fused_channels, fused_channels, kernel_size=3, padding=1),  # maintain spatial size
            nn.ReLU(),
            # nn.ConvTranspose2d(fused_channels, fused_channels, kernel_size=2, stride=2),  # 2x upsampling
            nn.Conv2d(fused_channels, 1, kernel_size=3, padding=1),  # 1x1024x1024
            nn.Sigmoid()
        )
        
        self.mask2_conv = nn.Sequential(
            nn.Conv2d(fused_channels, fused_channels, kernel_size=3, padding=1),  # maintain spatial size
            nn.ReLU(),
            # nn.ConvTranspose2d(fused_channels, fused_channels, kernel_size=2, stride=2),  # 2x upsampling
            nn.Conv2d(fused_channels, 1, kernel_size=3, padding=1),  # 1x1024x1024
            nn.Sigmoid()
        )
        
        # self.mask1_conv =  nn.Sequential(
        #     nn.Conv2d(fused_channels, fused_channels, kernel_size=4, stride=2, groups=2, padding=1),  # 2x518x518
        #     nn.Conv2d(fused_channels, 1, kernel_size=4, stride=2,  padding=1)  # 1x256x256
        #     )
        
        # self.mask2_conv = nn.Sequential(
        #     nn.Conv2d(fused_channels, fused_channels, kernel_size=4, stride=2, groups=2, padding=1),  # 2x518x518
        #     nn.Conv2d(fused_channels, 1, kernel_size=4, stride=2, padding=1)  # 1x256x256
        #     )

    def forward(self, feat_list):
        """
        Args:
            feat_list: list of feature maps, each of shape (B, 64, 64, 768)
        Returns:
            tc_mask, ar_mask: output masks
        """
        # Reverse the feature list and process in groups.
        # Permute feature maps from (B, 64, 64, 768) to (B, 768, 64, 64)
        feat_list = [f.permute(0, 3, 1, 2) for f in feat_list]
        reversed_feats = feat_list[::-1]
        # box_queries_list = []
        prev_up = None
        for block_idx in range(self.num_blocks):
            
            # Get group of features for this block.
            start_idx = block_idx * self.features_per_block
            end_idx = (block_idx + 1) * self.features_per_block
            group = reversed_feats[start_idx:end_idx]
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
        neck_out = self.neck(prev_up)
        tc_mask = self.mask1_conv(neck_out)
        ar_mask = self.mask2_conv(neck_out)
        
        tc_mask = F.interpolate(tc_mask, size=(768, 1152), mode='bilinear', align_corners=False)
        ar_mask = F.interpolate(ar_mask, size=(768, 1152), mode='bilinear', align_corners=False)
        
        return tc_mask.squeeze(), ar_mask.squeeze()

        