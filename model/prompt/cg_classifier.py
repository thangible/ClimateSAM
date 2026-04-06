import os
import torch
import torch.nn as nn
from .cgnet_module import CGNetModule

class CGNetPatchClassifier(nn.Module):
    def __init__(self, pretrained_weights_path, device, num_classes=3, in_channels=4):
        """
        Adapts a pretrained CGNet segmentation model into an image patch classifier.
        """
        super().__init__()
        
        # 1. Initialize the base CGNet architecture
        self.backbone = CGNetModule(classes=num_classes, channels=in_channels)
        
        # 2. Load the pretrained weights
        if pretrained_weights_path and os.path.exists(pretrained_weights_path):
            pretrained_dict = torch.load(pretrained_weights_path, map_location=device)
            model_dict = self.backbone.state_dict()
            
            # Filter out any mismatched keys (robust loading)
            filtered_dict = {
                k: v for k, v in pretrained_dict.items() 
                if k in model_dict and v.size() == model_dict[k].size()
            }
            model_dict.update(filtered_dict)
            self.backbone.load_state_dict(model_dict)
            print(f"✓ Loaded {len(filtered_dict)}/{len(model_dict)} matching layers into CGNet Classifier Backbone.")
        else:
            print(f"Warning: Pretrained weights not found at {pretrained_weights_path}! Using random init.")

        # 3. Add Global Average Pooling (GAP)
        # This crushes the spatial dimensions (H, W) down to a single pixel (1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # 1. Forward pass through the pretrained CGNet
        # Input shape: (B, 4, 128, 128) -> Output shape: (B, 3, 128, 128)
        spatial_logits = self.backbone(x)
        
        # 2. Pool the spatial dimensions to get an average "vote" per class
        # Output shape: (B, 3, 1, 1)
        pooled_logits = self.global_pool(spatial_logits)
        
        # 3. Flatten to standard classifier output shape
        # Output shape: (B, 3) -> [Background_Score, TC_Score, AR_Score]
        logits = torch.flatten(pooled_logits, 1)
        
        return logits