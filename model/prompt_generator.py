import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class PromptGenerator(nn.Module):
    def __init__(self, 
                 in_channels_list,  # list of channels from each level (e.g., [768, 768, ..., 768])
                 fused_channels: int,  # channels after fusion (N)
                 pool_size: tuple = (2, 2)
                ):
        super(PromptGenerator, self).__init__()
        self.conv_adapters = nn.ModuleList([
            nn.Conv2d(in_channels, fused_channels, kernel_size=3, padding=1)
            for in_channels in in_channels_list
        ])
        
        # Define transpose convolution upsamplers for each feature level.
        # Here we assume all features are upsampled from a smaller spatial size to the target size.
        # You may adjust kernel_size/stride as appropriate for your resolution differences.
        self.upsamplers = nn.ModuleList()
        for i in range(len(in_channels_list)):
            if i < 3:
                num_layers = 1
            elif i < 6:
                num_layers = 2
            elif i < 9:
                num_layers = 3
            else:
                num_layers = 4
            conv_layers = [nn.ConvTranspose2d(fused_channels, fused_channels, kernel_size=2, stride=2)
                        for _ in range(num_layers)]
            
            self.upsamplers.append(nn.Sequential(*conv_layers))

        self.fuse_conv = nn.Conv2d(len(in_channels_list)*fused_channels, fused_channels, kernel_size=3, padding=1)
        
        self.box_mlp = nn.Sequential(
            nn.Linear(len(in_channels_list)*fused_channels*pool_size[0]*pool_size[1], fused_channels),
            nn.ReLU(),
            nn.Linear(fused_channels, fused_channels)
        )
        self.pool = nn.AdaptiveAvgPool2d(pool_size)

    def forward(self, feat_list):
        processed_feats = []
        box_queries = []
        upsampled_feats = []
        for feat, conv, upsampler in zip(feat_list, self.conv_adapters, self.upsamplers):
            feat = feat.permute(0, 3, 1, 2)
            feat = conv(feat)  # (B, fused_channels, H, W)
            processed_feats.append(feat)
            
            # Pool for box query.
            box_feat = self.pool(feat)  # (B, fused_channels, pool_size[0], pool_size[1])
            box_queries.append(box_feat.reshape(feat.shape[0], -1))
            
            # Upsample using transpose convolution.
            up_feat = upsampler(feat)  # Adjust spatial dims accordingly.
            upsampled_feats.append(up_feat)
        
        # Alternatively, if feature levels don't match target size, you may want to further adjust sizes.
        # Here we assume upsamplers produce features of the same spatial size.
        fused_feats = torch.cat(upsampled_feats, dim=1)  # (B, len(feats)*fused_channels, H_target, W_target)
        fused_feats = self.fuse_conv(fused_feats)  # (B, fused_channels, H_target, W_target)
        
        box_concat = torch.cat(box_queries, dim=1)  # (B, len(feats)*fused_channels*pool_size[0]*pool_size[1])
        box_out = self.box_mlp(box_concat)  # (B, fused_channels)
        
        return fused_feats, box_out


# import torch
# import torch.nn as nn

# class PromptGenerator(nn.Module):
#     def __init__(self, depth: int = 12, in_channels: int = 256):
#         super(PromptGenerator, self).__init__()
        
#         self.depth = depth  # Total number of embeddings
        
#         # Branch blocks: each branch processes a feature map with i ConvTranspose2d layers
#         # i = 1 for the second deepest, i = 2 for the third deepest, etc.
#         self.branch_blocks = nn.ModuleList()
#         for i in range(1, depth):
#             layers = []
#             for _ in range(i):
#                 layers.append(
#                     nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=1, padding=0)
#                 )
#             self.branch_blocks.append(nn.Sequential(*layers))
        
#         # Fusion blocks: after concatenation of a processed branch (in_channels) and the current fused feature (in_channels),
#         # use a ConvTranspose2d to fuse and upsample the result back to in_channels.
#         self.fusion_convs = nn.ModuleList()
#         for _ in range(1, depth):
#             self.fusion_convs.append(
#                 nn.ConvTranspose2d(in_channels * 2, in_channels, kernel_size=4, stride=2, padding=1)
#             )
    
#     def forward(self, interm_embeddings: list):
#         # Assume interm_embeddings is a list of tensors.
#         # The deepest feature map (lowest resolution) is the last element.
#         x = interm_embeddings.pop()  # Deepest feature map
#         x = self.branch_blocks[0](x)  # Process the deepest feature map
        
        
#         # Iteratively fuse remaining feature maps (from deep to shallow)
#         for i in range(1, self.depth):
#             # Get the next highest feature map
#             current = interm_embeddings.pop()
            
#             # Process the current feature map using i ConvTranspose2d layers
#             processed_current = self.branch_blocks[i](current)
            
#             # Concatenate the processed current feature map with the previously fused feature
#             fused = torch.cat([processed_current, x], dim=1)
            
#             # Fuse and upsample the concatenated tensor
#             x = self.fusion_convs[i - 1](fused)
        
#         return x
    
    
    
# import torch
# import torch.nn as nn

# class PromptGenerator(nn.Module):
#     def __init__(self, depth: int, in_channels: int = 256):

#         super(PromptGenerator, self).__init__()
        
#         # List of layers (ConvTranspose2d layers)
#         self.conv_transpose_layers = nn.ModuleList()
#         self.total_layers = depth -1
        
#         # Create ConvTranspose2d layers based on depth
#         for total_layer_number in range(self.total_layers, -1, -1): # first element goes through dept -1 ConvTranspose2d layer
#             # Each ConvTranspose2d layer corresponds to the i-th index
#             # where the i-th tensor will go through i+1 ConvTranspose2d layers.
#             transposed_layers = []
#             for i in range(total_layer_number):
#                 stride = 2
#                 transposed_layers.append(
#                     nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=stride, padding=1)
#                 )
#             transposed_layers.append(
#                 nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
#             )
#             self.conv_transpose_layers.append(nn.Sequential(*transposed_layers))
        
#         # Example with self.total_layers = 5: 
#         # Layer 0: Contains 5 ConvTranspose2d layers followed by 1 Conv2d layer.
#         # Layer 1: Contains 4 ConvTranspose2d layers followed by 1 Conv2d layer.
#         # Layer 2: Contains 3 ConvTranspose2d layers followed by 1 Conv2d layer.
#         # Layer 3: Contains 2 ConvTranspose2d layers followed by 1 Conv2d layer.
#         # Layer 4: Contains 1 ConvTranspose2d layer followed by 1 Conv2d layer.
#         # Layer 5: Contains only 1 Conv2d layer.
                
#         # AdaptiveAvgPool2d layer
#         self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
#         # Pointwise Conv to generate binary mask
#         self.pointwise_conv = nn.Conv2d(256, 256, kernel_size=1)

#     def forward(self, interm_embeddings: list):
        
#         image_embeddings = interm_embeddings.pop() # Pop the last tensor from the list
#         processed_list = []
        
#         # Process each tensor in the list of inputs
#         for i, x in enumerate(interm_embeddings):
#             # Pass through the corresponding ConvTranspose2d layers
#             x = self.conv_transpose_layers[i](x)
            
#             # Apply AdaptiveAvgPool2d
#             x = self.adaptive_avg_pool(x)
            
#             # Append to the processed list
#             processed_list.append(x)
        
#         # Concatenate the results along the channel dimension (dim=1)
#         concatenated = torch.cat(processed_list, dim=1)
        
#         # Apply pointwise convolution to generate binary mask
#         mask = self.pointwise_conv(concatenated)
        
#         return mask
