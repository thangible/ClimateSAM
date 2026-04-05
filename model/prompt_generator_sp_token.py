import torch
import torch.nn as nn
import torch.nn.functional as F
from .layer_module import LayerNorm2d

class PromptGenerator(nn.Module):
    def __init__(self,
                 in_channels: int = 768,
                 fused_channels: int = 128,
                 out_channels: int = 3,
                 num_features: int = 12,
                 features_per_block: int = 3,
                 transformer_dim: int = 256):
        super(PromptGenerator, self).__init__()  
        
        self.fused_channels = fused_channels
        self.num_blocks = num_features // features_per_block
        self.features_per_block = features_per_block
        
        # 1. SHARED INPUT REDUCTION
        self.shared_input_reduction = nn.Sequential(
            nn.Conv2d(in_channels, fused_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )
        
        # 2. SHARED BLOCK FUSION
        self.shared_block_fuse = nn.Sequential(
            nn.Conv2d(features_per_block * fused_channels, fused_channels, kernel_size=1),
            nn.Conv2d(fused_channels, fused_channels, kernel_size=3, padding=1, stride=2, groups=fused_channels),
            nn.Conv2d(fused_channels, fused_channels, kernel_size=1),
            LayerNorm2d(fused_channels),
            nn.ReLU(inplace=True)
        )
        
        # 3. SHARED MULTILEVEL MASK
        self.shared_multilevel_mask = nn.Conv2d(fused_channels, out_channels, kernel_size=3, padding=1)

        # 4. TOKEN PROJECTION LOGIC
        refined_dim = transformer_dim // 8
        self.ar_gate_proj = nn.Sequential(
            nn.Linear(refined_dim, fused_channels),
            nn.Sigmoid()
        )
        self.tc_gate_proj = nn.Sequential(
            nn.Linear(refined_dim, fused_channels),
            nn.Sigmoid()
        )

        # 5. CLASS-SPECIFIC BINARY HEADS
        self.ar_head = nn.Conv2d(fused_channels, 1, kernel_size=1)
        self.tc_head = nn.Conv2d(fused_channels, 1, kernel_size=1)

        # Neck processes the accumulated channels
        self.neck = nn.Sequential(
            nn.Conv2d(self.num_blocks * fused_channels, fused_channels, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        
        self.multiclass_mask_conv = nn.Sequential(
            nn.Conv2d(fused_channels, fused_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fused_channels, out_channels, kernel_size=3, padding=1),
        )

    def _build_gate(self, refined_token, proj_layer, batch_size, dtype, device):
        gate = proj_layer(refined_token)

        # Accept [C], [1, C], or [B, C] token layouts and align to image batch.
        if gate.dim() == 1:
            gate = gate.unsqueeze(0)

        if gate.shape[0] == 1:
            gate = gate.expand(batch_size, -1)
        elif gate.shape[0] != batch_size:
            raise ValueError(
                f"Token batch mismatch: got {gate.shape[0]} gates for image batch {batch_size}."
            )

        return gate.to(device=device, dtype=dtype).unsqueeze(-1).unsqueeze(-1)

    def forward(self, feat_list, ar_refined=None, tc_refined=None):
        # Ensure we only process the expected number of hierarchical features
        feat_list = feat_list[-self.num_blocks * self.features_per_block:]
        
        # Permute feature maps from (B, 64, 64, 768) to (B, 768, 64, 64)
        feat_list = [f.permute(0, 3, 1, 2) for f in feat_list]
        reversed_feats = feat_list[::-1]
        
        intermediate_masks = []
        prev_up = None
        
        for block_idx in range(self.num_blocks):
            # Get group of features for this block
            start_idx = block_idx * self.features_per_block
            end_idx = (block_idx + 1) * self.features_per_block
            group = reversed_feats[start_idx:end_idx]
            
            # Apply shared reduction
            reduced_group = [self.shared_input_reduction(f) for f in group]
            
            # Upsample each feature using F.interpolate
            num_layers = block_idx + 1
            upsampled_group = []
            for f in reduced_group:
                upsampled_f = F.interpolate(f, scale_factor=2**num_layers, mode='nearest')
                upsampled_group.append(upsampled_f)
                
            # Concatenate along the channel dimension
            group_concat = torch.cat(upsampled_group, dim=1)
            
            # Fuse the concatenated features
            fused = self.shared_block_fuse(group_concat)
            
            # Compute intermediate mask using shared layer
            intermediate_logit = self.shared_multilevel_mask(fused)
            intermediate_masks.append(intermediate_logit)

            # Accumulate channels with previous blocks
            if prev_up is not None:
                fused = torch.cat([prev_up, fused], dim=1)
                
            # Upsample accumulated fused result to feed to the next block
            prev_up = F.interpolate(fused, scale_factor=2, mode='nearest')

        # The final accumulated feature is the output from the last block
        neck_out = self.neck(prev_up)
        
        # Compute the multi-class mask
        multiclass_mask = self.multiclass_mask_conv(neck_out)
        multiclass_mask = F.interpolate(multiclass_mask, size=(768, 1152), mode='nearest')

        # Class-specific binary heads using refined task tokens
        ar_mask, tc_mask = None, None
        if ar_refined is not None and tc_refined is not None:
            b = neck_out.shape[0]
            
            # Project tokens to gates and reshape for spatial broadcasting
            ar_gate = self._build_gate(
                ar_refined,
                self.ar_gate_proj,
                b,
                dtype=neck_out.dtype,
                device=neck_out.device,
            )
            tc_gate = self._build_gate(
                tc_refined,
                self.tc_gate_proj,
                b,
                dtype=neck_out.dtype,
                device=neck_out.device,
            )

            # Task-specific feature highlighting and head projection
            ar_mask = self.ar_head(neck_out * ar_gate)
            tc_mask = self.tc_head(neck_out * tc_gate)

            ar_mask = F.interpolate(ar_mask, size=(768, 1152), mode='nearest')
            tc_mask = F.interpolate(tc_mask, size=(768, 1152), mode='nearest')
        
        return multiclass_mask, intermediate_masks, ar_mask, tc_mask