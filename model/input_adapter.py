import torch
from torch import nn

class ClimateInputAdapter(nn.Module):
    def __init__(self, in_channels: int = 16, out_channels: int = 3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, out_channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid() # Squashes to 0-1 range
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x * 255.0 # Scale directly to pseudo-RGB range