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
                 features_per_block: int = 3):
        super(PromptGenerator, self).__init__()  
        
        self.num_blocks = num_features // features_per_block
        self.features_per_block = features_per_block
        
        if num_features % features_per_block != 0:
            raise ValueError(f"num_features ({num_features}) must be divisible by features_per_block ({features_per_block})")
             
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        
        self.input_reduction = nn.ModuleList()
        self.shallow_upsamplers = nn.ModuleList()
        self.up_trans = nn.ModuleList()
        self.fuse_convs = nn.ModuleList()
        self.multilevel_mask_convs = nn.ModuleList()
        
        for block_idx in range(self.num_blocks):
            # 1. Reduce incoming channels to fused_channels
            self.input_reduction.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, fused_channels, kernel_size=1, bias=False),
                    nn.ReLU(),
                )
            )
            
            # 2. Setup structural upsampling and fusion
            if block_idx == 0:
                # The deepest block fuses directly without upsampling
                self.fuse_convs.append(
                    nn.Sequential(
                        nn.Conv2d(features_per_block * fused_channels, fused_channels, kernel_size=3, padding=1),
                        LayerNorm2d(fused_channels),
                        nn.ReLU()
                    )
                )
                self.shallow_upsamplers.append(nn.Identity())
            else:
                # Shallower blocks require upsampling to match the spatial dimensions
                # of the accumulated deeper features
                layers = []
                in_ch = features_per_block * fused_channels
                for i in range(block_idx):
                    layers.append(nn.ConvTranspose2d(in_ch if i == 0 else fused_channels,
                                                     fused_channels, kernel_size=2, stride=2))
                    layers.append(nn.ReLU())
                self.shallow_upsamplers.append(nn.Sequential(*layers))
                
                # Transpose convolution to double the size of the previous accumulated features
                self.up_trans.append(
                    nn.ConvTranspose2d(fused_channels, fused_channels, kernel_size=2, stride=2)
                )
                
                # Fuse the concatenated features
                self.fuse_convs.append(
                    nn.Sequential(
                        nn.Conv2d(fused_channels * 2, fused_channels, kernel_size=3, padding=1),
                        LayerNorm2d(fused_channels),
                        nn.ReLU()
                    )
                )
                
            # 3. Deep supervision layer for current block
            self.multilevel_mask_convs.append(
                nn.Conv2d(fused_channels, out_channels, kernel_size=3, padding=1)
            )

        self.neck = nn.Sequential(
            nn.Conv2d(fused_channels, fused_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.multiclass_mask_conv = nn.Sequential(
            nn.Conv2d(fused_channels, fused_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(fused_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, feat_list):
        # Permute feature maps from (B, 64, 64, 768) to (B, 768, 64, 64)
        feat_list = [f.permute(0, 3, 1, 2) for f in feat_list]
        reversed_feats = feat_list[::-1]
        
        intermediate_masks = []
        prev_fused = None
        
        for block_idx in range(self.num_blocks):
            start_idx = block_idx * self.features_per_block
            end_idx = (block_idx + 1) * self.features_per_block
            group = reversed_feats[start_idx:end_idx]
            
            reduced_group = [self.input_reduction[block_idx](f) for f in group]
            group_concat = torch.cat(reduced_group, dim=1) 
            
            if block_idx == 0:
                fused = self.fuse_convs[block_idx](group_concat)
            else:
                # Upsample the current shallower group to match current spatial scale
                current_upsampled = self.shallow_upsamplers[block_idx](group_concat)
                
                # Upsample the accumulated deeper features
                prev_upsampled = self.up_trans[block_idx - 1](prev_fused)
                
                # Concatenate and fuse
                concat_fused = torch.cat([prev_upsampled, current_upsampled], dim=1)
                fused = self.fuse_convs[block_idx](concat_fused)
                
            # Deep supervision calculation
            intermediate_logit = self.multilevel_mask_convs[block_idx](fused)
            intermediate_masks.append(intermediate_logit)
            
            prev_fused = fused
            
        neck_out = self.neck(prev_fused)
        multiclass_mask = self.multiclass_mask_conv(neck_out)

        # Final interpolation to exact target size
        multiclass_mask = F.interpolate(multiclass_mask, size=(768, 1152), mode='bilinear', align_corners=False)
        
        return multiclass_mask, intermediate_masks