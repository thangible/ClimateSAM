import torch
import torch.nn as nn
import torch.nn.functional as F
from .layer_module import LayerNorm2d
from prompt.cgnet_module import ContextGuidedBlock, ConvBNPReLU

class PromptGenerator(nn.Module):
    def __init__(self, 
                 pool_size: tuple = (2, 2),
                 fused_channels: int = 128,
                 in_channels: int = 768,
                 out_channels: int = 3,
                 num_features: int = 12,
                 features_per_block: int = 3,
                 transformer_dim: int = 256):
        super(PromptGenerator, self).__init__()  
        
        self.fused_channels = fused_channels
        self.num_blocks = num_features // features_per_block
        self.features_per_block = features_per_block
        
        if num_features % features_per_block != 0:
            raise ValueError(f"num_features ({num_features}) must be divisible by features_per_block ({features_per_block})")
             
        # Project refined (MLP) tokens (dim = 32) into class-specific gates
        # These refined tokens are the 'intelligence' from the Mask Decoder [cite: 246, 249]
        refined_dim = transformer_dim // 8
        self.ar_gate_proj = nn.Sequential(
            nn.Linear(refined_dim, fused_channels),
            nn.Sigmoid()
        )
        self.tc_gate_proj = nn.Sequential(
            nn.Linear(refined_dim, fused_channels),
            nn.Sigmoid()
        )

        # Final binary heads for automated prompt generation 
        self.ar_head = nn.Conv2d(fused_channels, 1, kernel_size=1)
        self.tc_head = nn.Conv2d(fused_channels, 1, kernel_size=1)
        
        # Encoder blocks for multi-scale fusion
        self.input_reduction = nn.ModuleList()
        self.block_feature_upsamplers = nn.ModuleList()
        self.block_fuse_convs = nn.ModuleList()
        self.block_out_trans = nn.ModuleList()
        
        # Multi-level supervision to address ribbon-like and circular structures [cite: 258, 487]
        self.multilevel_mask_convs = nn.ModuleList()
        for _ in range(self.num_blocks):
            self.multilevel_mask_convs.append(
                nn.Conv2d(fused_channels, out_channels, kernel_size=3, padding=1)
            )
        
        for block_idx in range(self.num_blocks):
            self.input_reduction.append(
                ConvBNPReLU(in_channels, fused_channels, kSize=1, stride=1)
            )
            num_layers = block_idx + 1
            self.block_feature_upsamplers.append(
                nn.Sequential(*[
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                    for _ in range(num_layers)
                ])
            )
            self.block_fuse_convs.append(
                nn.Sequential(
                    # Using standard conv here to transition dimensions
                    nn.Conv2d(features_per_block * fused_channels, fused_channels, kernel_size=1),
                    # We can use ConvBNPReLU for the spatial downsampling
                    ConvBNPReLU(fused_channels, fused_channels, kSize=3, stride=2),
                    LayerNorm2d(fused_channels)
                )
            )
            self.block_out_trans.append(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            )

        self.neck = nn.Sequential(
            ConvBNPReLU(self.num_blocks * fused_channels, fused_channels, kSize=1, stride=1),
            ContextGuidedBlock(fused_channels, fused_channels, dilation_rate=2, reduction=16, add=True),
            ContextGuidedBlock(fused_channels, fused_channels, dilation_rate=4, reduction=16, add=True)
        )
        
        self.multiclass_mask_conv = nn.Sequential(
            ContextGuidedBlock(fused_channels, fused_channels, dilation_rate=2, reduction=16, add=True),
            # Final output must remain plain Conv2d for raw logits
            nn.Conv2d(fused_channels, out_channels, kernel_size=3, padding=1), 
        )

    def forward(self, feat_list, ar_refined=None, tc_refined=None):
        # Permute feature maps from (B, 64, 64, 768) to (B, 768, 64, 64) [cite: 154]
        feat_list = [f.permute(0, 3, 1, 2) for f in feat_list]
        intermediate_masks = []
        reversed_feats = feat_list[::-1]

        prev_up = None
        for block_idx in range(self.num_blocks):
            start_idx = block_idx * self.features_per_block
            end_idx = (block_idx + 1) * self.features_per_block
            group = reversed_feats[start_idx:end_idx]
            
            # 1. Feature reduction and multi-scale fusion [cite: 323, 480]
            reduced_group = [self.input_reduction[block_idx](f) for f in group]
            upsampled_group = [self.block_feature_upsamplers[block_idx](f) for f in reduced_group]
            
            group_concat = torch.cat(upsampled_group, dim=1)
            fused = self.block_fuse_convs[block_idx](group_concat)
            
            # 2. Intermediate auxiliary logits [cite: 322, 487]
            intermediate_logit = self.multilevel_mask_convs[block_idx](fused)
            intermediate_masks.append(intermediate_logit)

            if prev_up is not None:
                fused = torch.cat([prev_up, fused], dim=1)
            
            up = self.block_out_trans[block_idx](fused)
            prev_up = up

        # 3. Global neck features
        neck_out = self.neck(prev_up)
        
        # 4. Multiclass Prediction (Logits for classes 0, 1, 2) [cite: 48, 553]
        multiclass_mask = self.multiclass_mask_conv(neck_out)
        multiclass_mask = F.interpolate(multiclass_mask, size=(768, 1152), mode='bilinear', align_corners=False)
        
        # 5. Class-specific binary heads using refined task tokens [cite: 249, 362]
        ar_mask, tc_mask = None, None
        if ar_refined is not None and tc_refined is not None:
            # Broadcast gates to match batch size [cite: 308]
            b = neck_out.shape[0]
            ar_gate = self.ar_gate_proj(ar_refined).view(-1, self.fused_channels, 1, 1).expand(b, -1, -1, -1)
            tc_gate = self.tc_gate_proj(tc_refined).view(-1, self.fused_channels, 1, 1).expand(b, -1, -1, -1)

            # Task-specific feature highlighting
            ar_mask = self.ar_head(neck_out * ar_gate)
            tc_mask = self.tc_head(neck_out * tc_gate)

            ar_mask = F.interpolate(ar_mask, size=(768, 1152), mode='bilinear', align_corners=False)
            tc_mask = F.interpolate(tc_mask, size=(768, 1152), mode='bilinear', align_corners=False)

        return multiclass_mask, intermediate_masks, ar_mask, tc_mask

