import torch
import torch.nn as nn
import torch.nn.functional as F
from .layer_module import LayerNorm2d

class PromptGenerator(nn.Module):
    def __init__(self, 
                 in_channels: int = 768,
                 fused_channels: int = 128,
                 out_channels: int = 3,
                 num_features: int = 5):
        super(PromptGenerator, self).__init__()  
        
        self.num_features = num_features
        
        self.input_reduction = nn.ModuleList()
        self.shallow_upsamplers = nn.ModuleList()
        self.up_trans = nn.ModuleList()
        self.fuse_convs = nn.ModuleList()
        self.multilevel_mask_convs = nn.ModuleList()
        
        for i in range(self.num_features):
            # Reduce incoming channels to a consistent fused_channels dimension
            self.input_reduction.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, fused_channels, kernel_size=1, bias=False),
                    nn.ReLU(inplace=True),
                )
            )
            
            # Create the learned upsampling pathway for the incoming shallower features.
            # The higher the level (i), the more times it needs to be upsampled to match the main branch.
            shallow_layers = []
            for _ in range(i):
                shallow_layers.append(
                    nn.ConvTranspose2d(fused_channels, fused_channels, kernel_size=2, stride=2)
                )
                shallow_layers.append(nn.ReLU(inplace=True))
                
            if len(shallow_layers) > 0:
                self.shallow_upsamplers.append(nn.Sequential(*shallow_layers))
            else:
                self.shallow_upsamplers.append(nn.Identity())
            
            if i == 0:
                # Deepest level fusion
                self.fuse_convs.append(
                    nn.Sequential(
                        nn.Conv2d(fused_channels, fused_channels, kernel_size=3, padding=1),
                        LayerNorm2d(fused_channels),
                        nn.ReLU(inplace=True)
                    )
                )
            else:
                # Transpose convolution to double the size of the accumulated main branch
                self.up_trans.append(
                    nn.ConvTranspose2d(fused_channels, fused_channels, kernel_size=2, stride=2)
                )
                
                # Fuse the concatenated features
                self.fuse_convs.append(
                    nn.Sequential(
                        nn.Conv2d(fused_channels * 2, fused_channels, kernel_size=3, padding=1),
                        LayerNorm2d(fused_channels),
                        nn.ReLU(inplace=True)
                    )
                )
                
            # Deep supervision layer applied at every hierarchical level
            self.multilevel_mask_convs.append(
                nn.Conv2d(fused_channels, out_channels, kernel_size=3, padding=1)
            )

        self.neck = nn.Sequential(
            nn.Conv2d(fused_channels, fused_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.multiclass_mask_conv = nn.Sequential(
            nn.Conv2d(fused_channels, fused_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fused_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, feat_list):
        """
        Args:
            feat_list: list of feature maps from the image encoder.
        Returns:
            multiclass_mask: final high resolution mask.
            intermediate_masks: list of masks for deep supervision.
        """
        # Ensure we only process the expected number of hierarchical features
        feat_list = feat_list[-self.num_features:] 
        
        # Permute feature maps from (B, H, W, C) to (B, C, H, W)
        feat_list = [f.permute(0, 3, 1, 2) for f in feat_list]
        
        # Reverse the list to start processing from the deepest feature map
        reversed_feats = feat_list[::-1]
        
        intermediate_masks = []
        accumulated_feat = None
        
        for i in range(self.num_features):
            current_feat = reversed_feats[i]
            
            # Reduce channel dimensionality
            reduced_feat = self.input_reduction[i](current_feat)
            
            if i == 0:
                # Process the deepest level
                fused = self.fuse_convs[i](reduced_feat)
            else:
                # Upsample the accumulated features from the previous deeper level
                upsampled_accumulated = self.up_trans[i - 1](accumulated_feat)
                
                # Upsample the incoming shallower feature using the learned ConvTranspose2d layers
                aligned_reduced_feat = self.shallow_upsamplers[i](reduced_feat)
                
                # Concatenate the upsampled features with the aligned shallower features
                concat_feat = torch.cat([upsampled_accumulated, aligned_reduced_feat], dim=1)
                
                # Fuse the concatenated representation
                fused = self.fuse_convs[i](concat_feat)
                
            # Update the accumulated feature for the next iteration
            accumulated_feat = fused
            
            # Generate the deep supervision mask for the current level
            intermediate_logit = self.multilevel_mask_convs[i](accumulated_feat)
            intermediate_masks.append(intermediate_logit)
            
        # Final neck and output generation
        neck_out = self.neck(accumulated_feat)
        multiclass_mask = self.multiclass_mask_conv(neck_out)

        # Final interpolation to the target ground truth resolution is still standard practice 
        # (as the GT mask is rarely the exact downsampled multiple of the ViT output)
        multiclass_mask = F.interpolate(multiclass_mask, size=(768, 1152), mode='bilinear', align_corners=False)
        
        return multiclass_mask, intermediate_masks