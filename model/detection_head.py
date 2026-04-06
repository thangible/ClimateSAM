import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenGatedDetectionHead(nn.Module):
    def __init__(self, embedding_dim=256, num_classes=2):
        super().__init__()
        # Project refined tokens from SAM's decoder into channel-wise gates
        self.ar_gate = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Sigmoid()
        )
        self.tc_gate = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Sigmoid()
        )
        
        # Detection Neck: Processing gated embeddings
        self.conv_block = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embedding_dim, embedding_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(embedding_dim // 2),
            nn.ReLU(inplace=True),
            # Output: 1 (conf) + 4 (bbox) + num_classes (TC/AR)
            nn.Conv2d(embedding_dim // 2, 1 + 4 + num_classes, kernel_size=1)
        )

    def forward(self, image_embeddings, ar_token, tc_token):
        """
        image_embeddings: [B, 256, 64, 64] from SAM image encoder
        ar_token, tc_token: [B, 256] from SAM mask decoder
        """
        # Generate class-specific channel weights
        ar_w = self.ar_gate(ar_token).unsqueeze(-1).unsqueeze(-1)
        tc_w = self.tc_gate(tc_token).unsqueeze(-1).unsqueeze(-1)
        
        # Apply gating: Emphasize features known to the class tokens
        gated_features = image_embeddings * (ar_w + tc_w)
        
        # Grid-based prediction (YOLO-style)
        return self.conv_block(gated_features)