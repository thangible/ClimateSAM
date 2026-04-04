import torch
from torch import nn


class ClimateInputAdapter(nn.Module):
    """Base class for climate input adapters.
    
    Diagnostic variables: TMQ (0), U850 (1), V850 (2), PSL (7)
    """
    
    DIAGNOSTIC_INDICES = [0, 1, 2, 7]
    
    def __init__(self, in_channels=16, out_channels=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
    
    def initialize_weights(self):
        """Override in subclasses to implement specific weight initialization."""
        raise NotImplementedError("Subclasses must implement initialize_weights()")
    
    def forward(self, x):
        """Override in subclasses to implement specific forward pass."""
        raise NotImplementedError("Subclasses must implement forward()")


class LinearClimateInputAdapter(ClimateInputAdapter):
    """Linear adapter using a single 1x1 convolution layer (default)."""
    
    def __init__(self, in_channels=16, out_channels=3, hidden_dim=32):
        super().__init__(in_channels, out_channels)
        
        self.input_adapt = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        )
        
        self.initialize_weights()

    def initialize_weights(self):
        first_conv = self.input_adapt[0]

        with torch.no_grad():
            # Initialize with small normal distribution
            nn.init.normal_(first_conv.weight, mean=0.0, std=0.05)
            nn.init.constant_(first_conv.bias, 0.0)
            
            # Priority Injection for diagnostic variables
            first_conv.weight[0, 0, 0, 0] = 1.0
            first_conv.weight[1, 1, 0, 0] = 1.0
            first_conv.weight[2, 2, 0, 0] = 1.0

    def forward(self, x):
        x = self.input_adapt(x)
        x = torch.clamp(x, 0.0, 255.0)
        return x


class NonlinearClimateInputAdapter(ClimateInputAdapter):
    """Non-linear adapter with multiple convolutional layers and batch normalization."""
    
    def __init__(self, in_channels=16, out_channels=3):
        super().__init__(in_channels, out_channels)
        
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
        first_conv = self.layers[0]
        with torch.no_grad():
            # Initialize with small normal distribution
            nn.init.normal_(first_conv.weight, mean=0.0, std=0.05)
            nn.init.constant_(first_conv.bias, 0.0)
            
            # Priority Injection for diagnostic variables
            first_conv.weight[0, 0, 0, 0] = 1.0
            first_conv.weight[1, 1, 0, 0] = 1.0
            first_conv.weight[2, 2, 0, 0] = 1.0

    def forward(self, x):
        return self.layers(x) * 255.0
    
    
    
# import torch
# from torch import nn

# class ClimateInputAdapter(nn.Module):
#     def __init__(self, in_channels=16, out_channels=3, hidden_dim=32):
#         super().__init__()
        
#         self.input_adapt = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
#         )
        
#         self.initialize_weights()

#     def initialize_weights(self):
#         # diagnostic variables: TMQ (0), U850 (1), V850 (2), PSL (7)
#         diagnostic_indices = [0, 1, 2, 7] 

#         first_conv = self.input_adapt[0]

#         with torch.no_grad():
#             # Initialize with Kaiming Uniform to keep the signal variance stable
#             nn.init.kaiming_uniform_(first_conv.weight, nonlinearity='linear')
            
#             # Priority Injection
#             for out_ch in range(first_conv.weight.shape[0]):
#                 # Set TMQ (Index 0) to a higher priority
#                 first_conv.weight[out_ch, 0, 0, 0] = 1.0 
                
#                 # Set U850 and V850 (Indices 1 and 2) to 1.0
#                 first_conv.weight[out_ch, 1, 0, 0] = 1.0
#                 first_conv.weight[out_ch, 2, 0, 0] = 1.0
#                 # Optional: Set PSL (Index 7) to 1.0 to help with TC centers
#                 first_conv.weight[out_ch, 7, 0, 0] = 0.5

#     def forward(self, x):
#         # 1. Non-linear point-wise transformation
#         x = self.input_adapt(x)
        
#         # 2. Rescale to 0-255 range
#         # We shift and scale, then clamp only at the very end to prevent overflow.
#         # This keeps the internal "subtle" changes from becoming 0 or 1.
#         # x = x * 127.5 + 127.5
        
#         return torch.clamp(x, 0.0, 255.0)