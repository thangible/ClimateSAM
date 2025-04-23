import torch
import pytest


# Pseudocode:
# 1. Import necessary modules: torch, pytest and the function/class to test via absolute import.
# 2. Define a test function to verify the __init__ method of PromptGenerator:
#    a. Create an instance of PromptGenerator.
#    b. Assert the block_feature_upsamplers, block_fuse_convs, block_out_trans ModuleLists all have length 4.
#    c. Assert that each element of these ModuleLists is of the expected type (e.g., nn.Sequential for upsamplers,
#       and nn.Conv2d for fuse_convs, nn.ConvTranspose2d for out_trans).
#    d. Assert the box_mlp attribute is an instance of nn.Sequential.
# 3. Optionally, run the tests via pytest's main block.

import torch.nn as nn
from prompt_generator import PromptGenerator


def test_promptgenerator_forward():
    B = 3
    # Create 12 dummy feature maps with shape (B, 64, 64, 768)
    dummy_feats = [torch.rand(B, 64, 64, 768) for _ in range(12)]
    pg = PromptGenerator()
    
    fused_feats = pg(dummy_feats)
    
    # Expected output shapes:
    # fused_feats should be (B, 768, 1024, 1024) based on the upsampling in 4 blocks:
    #   Block0: 64 -> 128 -> fuse (stride=2) -> 64 -> out_trans -> 128
    #   Block1: 64 -> 256 -> fuse -> 128 -> out_trans -> 256
    #   Block2: 64 -> 512 -> fuse -> 256 -> out_trans -> 512
    #   Block3: 64 -> 1024 -> fuse -> 512 -> out_trans -> 1024
    # box_out is produced by box_mlp with output shape (B, 768)
    expected_fused_shape = (B, 2, 256, 256)
    # expected_box_shape = (B, 2)
    
    assert fused_feats.shape == expected_fused_shape, \
        f"Expected fused_feats shape {expected_fused_shape}, got {fused_feats.shape}"


if __name__ == '__main__':
    pytest.main([__file__])