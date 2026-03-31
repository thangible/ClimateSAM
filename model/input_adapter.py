import torch
from torch import nn

class ClimateInputAdapter(nn.Module):
    def __init__(self, in_channels=16, out_channels=3):
        super().__init__()
        
        # Standard architecture defined in your approach
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, out_channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid() 
        )
        
        self.initialize_weights()

    def initialize_weights(self):
        # Indices based on your ClimateDataset.variables list
        # TMQ=0, U850=1, V850=2, PSL=7
        diagnostic_indices = [0, 1, 2, 7] 
        
        first_conv = self.layers[0]
        with torch.no_grad():
            # Small random initialization for non-diagnostic channels
            nn.init.normal_(first_conv.weight, mean=0.0, std=0.02)
            nn.init.constant_(first_conv.bias, 0.0)
            
            # Set diagnostic channels to 1.0 to prioritize them initially
            for out_ch in range(first_conv.weight.shape[0]):
                for in_ch in diagnostic_indices:
                    # For a 3x3 kernel, [1, 1] is the center pixel
                    first_conv.weight[out_ch, in_ch, 1, 1] = 1.0

    def forward(self, x):
        # Pass through layers and scale to pseudo-RGB range (0-255)
        return self.layers(x) * 255.0