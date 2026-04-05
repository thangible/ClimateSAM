import torch
import torch.nn as nn
import torch.nn.functional as F
from .layer_module import LayerNorm2d

class PromptGenerator(nn.Module):
    def __init__(self, 
                 in_channels: int = 768,
                 fused_channels: int = 128,
                 out_channels: int = 3,
                 num_features: int = 12):
        super(PromptGenerator, self).__init__()  
        
        self.num_features = num_features
        
        self.input_reduction = nn.ModuleList()
        self.up_trans = nn.ModuleList()
        self.fuse_convs = nn.ModuleList()
        self.multilevel_mask_convs = nn.ModuleList()
        
        # METHOD 1: Single Shared Upsampling Block
        # This replaces the 4.3M parameter combinatorial module list
        self.shared_shallow_up = nn.Sequential(
            nn.ConvTranspose2d(fused_channels, fused_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        
        for i in range(self.num_features):
            # Reduce incoming channels to a consistent fused_channels dimension
            self.input_reduction.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, fused_channels, kernel_size=1, bias=False),
                    nn.ReLU(inplace=True),
                )
            )
            
            if i == 0:
                # Deepest level fusion
                # self.fuse_convs.append(
                #     nn.Sequential(
                #         nn.Conv2d(fused_channels, fused_channels, kernel_size=3, padding=1),
                #         LayerNorm2d(fused_channels),
                #         nn.ReLU(inplace=True)
                #     )
                # )
                self.fuse_convs.append(
                    nn.Sequential(
                        # Depthwise (groups = in_channels)
                        nn.Conv2d(fused_channels * 2, fused_channels * 2, kernel_size=3, padding=1, groups=fused_channels * 2),
                        # Pointwise
                        nn.Conv2d(fused_channels * 2, fused_channels, kernel_size=1),
                        LayerNorm2d(fused_channels),
                        nn.ReLU(inplace=True)
                    )
                )
            else:
                # Transpose convolution to double the size of the accumulated main branch
                self.up_trans.append(
                    nn.ConvTranspose2d(fused_channels, fused_channels, kernel_size=2, stride=2)
                )
                
                # # Fuse the concatenated features
                # self.fuse_convs.append(
                #     nn.Sequential(
                #         nn.Conv2d(fused_channels * 2, fused_channels, kernel_size=3, padding=1),
                #         LayerNorm2d(fused_channels),
                #         nn.ReLU(inplace=True)
                #     )
                # )
                self.fuse_convs.append(
                    nn.Sequential(
                        # Depthwise (groups = in_channels)
                        nn.Conv2d(fused_channels * 2, fused_channels * 2, kernel_size=3, padding=1, groups=fused_channels * 2),
                        # Pointwise
                        nn.Conv2d(fused_channels * 2, fused_channels, kernel_size=1),
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
                
                # METHOD 1 APPLIED:
                # Iteratively apply the shared upsampler to align the incoming 
                # shallower feature with the accumulated deep features.
                aligned_reduced_feat = reduced_feat
                for _ in range(i):
                    aligned_reduced_feat = self.shared_shallow_up(aligned_reduced_feat)
                
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

        # Final interpolation to the target ground truth resolution
        multiclass_mask = F.interpolate(multiclass_mask, size=(768, 1152), mode='bilinear', align_corners=False)
        
        return multiclass_mask, intermediate_masks