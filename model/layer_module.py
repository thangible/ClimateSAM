import torch
from typing import List, Union, Type

import torch.nn as nn


import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x



class DConvAdapter(nn.Module):
    def __init__(self, embed_dim, reduction=4):
        super(DConvAdapter, self).__init__()
        self.fc_down = nn.Linear(embed_dim, embed_dim // reduction)
        self.gelu = nn.GELU()
        # A 3D depth-wise conv can be simulated by setting groups equal to the number of channels.
        # Here we treat the spatial dims and the reduced channels as a 3D volume.
        self.dconv = nn.Conv3d(
            in_channels=1,  # We insert a dummy spatial dimension.
            out_channels=1,
            kernel_size=(3, 3, 3),
            padding=1,
            groups=1,
        )
        self.fc_up = nn.Linear(embed_dim // reduction, embed_dim)

    def forward(self, x):
        # x has shape (B, H, W, C)
        shortcut = x
        B, H, W, C = x.shape
        # FC Down
        x = self.fc_down(x)  # (B, H, W, C//reduction)
        x = self.gelu(x)
        # Prepare for 3D depth-wise conv:
        # Reshape: treat (H,W) as a 2D grid and add a dummy channel dimension for 3d conv.
        x = x.unsqueeze(1)  # (B, 1, H, W, C//reduction)
        # Apply 3D depth-wise conv over the (H, W, C//reduction) dimensions.
        x = self.dconv(x)
        x = x.squeeze(1)    # Back to (B, H, W, C//reduction)
        # FC Up and add skip connection.
        x = self.fc_up(x)   # (B, H, W, C)
        return x + shortcut
    

class Adapter(nn.Module):
    """
    Adapted from
    https://github.com/NiFangBaAGe/Explicit-Visual-Prompt/blob/601ae9985f04264c0411aa3722822f70134fa488/models/mmseg/models/backbones/vit_adaptformer.py#L295
    """
    def __init__(
            self,
            in_features: int,
            out_features: int = None,
            mlp_ratio: Union[List[float], float] = 0.25,
            add_last_layer: bool = True
    ):
        super().__init__()
        if add_last_layer and (out_features is None):
            raise ValueError

        if not isinstance(mlp_ratio, List):
            mlp_ratio = [mlp_ratio]

        layer_list, input_dim = [], in_features
        for m_r in mlp_ratio:
            output_dim = int(in_features * m_r)
            layer_list.extend(
                [
                    nn.Linear(input_dim, output_dim),
                    torch.nn.GELU()
                ]
            )
            input_dim = output_dim

        if add_last_layer:
            layer_list.append(nn.Linear(input_dim, out_features))
        self.adapter = nn.Sequential(*layer_list)


    def forward(self, x):
        x = self.adapter(x)
        return x


class MLP(nn.Module):
    """
    Adapted from
    https://github.com/SysCV/sam-hq/blob/322488826bda616798901c6280d13a9a90444ae7/train/train.py#L43
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

