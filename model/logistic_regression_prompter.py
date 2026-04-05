import torch
import torch.nn as nn
import torch.nn.functional as F


class LogisticRegressionPrompter(nn.Module):
    """
    A logistic regression-based prompter that performs pixel-wise classification.
    
    Takes image embeddings from a transformer (e.g., ViT-B: 256 channels at 64x64),
    performs pixel-wise linear classification to create a coarse mask,
    interpolates it to the nearest neighbor, and outputs a multiclass mask.
    
    This approach is much simpler than PromptGenerator but still effective for
    generating prompts for SAM.
    """
    
    def __init__(self, 
                 in_channels: int = 768,
                 out_channels: int = 3,
                 interpolation_mode: str = 'nearest'):
        """
        Args:
            in_channels: Number of input channels from transformer embedding (e.g., 768 for ViT-B)
            out_channels: Number of output classes (default: 3 for background, TC, AR)
            interpolation_mode: Interpolation mode for upsampling ('nearest', 'bilinear', etc.)
        """
        super(LogisticRegressionPrompter, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.interpolation_mode = interpolation_mode
        
        # Pixel-wise classifier: Linear layer applied to each spatial position
        # Input: (B, C, H, W) -> reshape to (B*H*W, C) -> classify -> reshape back
        self.classifier = nn.Linear(in_channels, out_channels)
        
    def forward(self, feat_list):
        """
        Args:
            feat_list: List of feature maps from transformer encoder.
                      Each feature map has shape (B, H, W, C) where C is in_channels.
                      
        Returns:
            multiclass_mask: Output mask with shape (B, out_channels, H, W) containing logits
            intermediate_masks: List of intermediate predictions (empty for logistic regression)
        """
        # Use the first (and typically only) feature map
        if isinstance(feat_list, list) and len(feat_list) > 0:
            x = feat_list[0]
        else:
            x = feat_list
        
        # x shape: (B, H, W, C)
        b, h, w, c = x.shape
        
        # Reshape to (B*H*W, C) for pixel-wise classification
        x = x.reshape(-1, c)
        
        # Apply linear classifier
        logits = self.classifier(x)  # (B*H*W, out_channels)
        
        # Reshape back to (B, H, W, out_channels)
        logits = logits.reshape(b, h, w, self.out_channels)
        
        # Permute to (B, out_channels, H, W) for consistency with other modules
        logits = logits.permute(0, 3, 1, 2)  # (B, out_channels, H, W)
        
        # Interpolate to target size (768, 1152) using nearest neighbor
        multiclass_mask = F.interpolate(
            logits, 
            size=(768, 1152), 
            mode=self.interpolation_mode, 
            align_corners=False if self.interpolation_mode != 'nearest' else None
        )
        
        # No intermediate masks for logistic regression (return empty list)
        intermediate_masks = []
        
        return multiclass_mask, intermediate_masks
