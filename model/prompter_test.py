import torch
import pytest
from prompter import PromptGenerator

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



def test_promptgenerator_forward():
    B = 2
    # Create 12 dummy feature maps with shape (B, 64, 64, 768)
    dummy_feats = [torch.rand(B, 64, 64, 768) for _ in range(12)]
    pg = PromptGenerator()
    
    fused_feats, box_out = pg(dummy_feats)
    
    # Expected output shapes:
    # fused_feats should be (B, 768, 1024, 1024) based on the upsampling in 4 blocks:
    #   Block0: 64 -> 128 -> fuse (stride=2) -> 64 -> out_trans -> 128
    #   Block1: 64 -> 256 -> fuse -> 128 -> out_trans -> 256
    #   Block2: 64 -> 512 -> fuse -> 256 -> out_trans -> 512
    #   Block3: 64 -> 1024 -> fuse -> 512 -> out_trans -> 1024
    # box_out is produced by box_mlp with output shape (B, 768)
    expected_fused_shape = (B, 768, 1024, 1024)
    expected_box_shape = (B, 768)
    
    assert fused_feats.shape == expected_fused_shape, \
        f"Expected fused_feats shape {expected_fused_shape}, got {fused_feats.shape}"
    assert box_out.shape == expected_box_shape, \
        f"Expected box_out shape {expected_box_shape}, got {box_out.shape}"



def test_promptgenerator_init():
    # Create an instance of PromptGenerator with default parameters.
    pg = PromptGenerator()
    
    # Check that block_feature_upsamplers has 4 modules, each should be a nn.Sequential.
    assert len(pg.block_feature_upsamplers) == 4, "Expected 4 feature upsampler blocks"
    for seq in pg.block_feature_upsamplers:
        assert isinstance(seq, nn.Sequential), "Each feature upsampler should be an instance of nn.Sequential"
        # Check that the sequential module contains nn.ConvTranspose2d layers.
        for layer in seq:
            assert isinstance(layer, nn.ConvTranspose2d), "Layers in feature upsampler should be ConvTranspose2d"
    
    # Check that block_fuse_convs has 4 modules and each module is a nn.Conv2d.
    assert len(pg.block_fuse_convs) == 4, "Expected 4 fuse conv blocks"
    for conv in pg.block_fuse_convs:
        assert isinstance(conv, nn.Conv2d), "Each fuse conv block should be an instance of nn.Conv2d"
    
    # Check that block_out_trans has 4 modules and each module is a nn.ConvTranspose2d.
    assert len(pg.block_out_trans) == 4, "Expected 4 output trans blocks"
    for conv_t in pg.block_out_trans:
        assert isinstance(conv_t, nn.ConvTranspose2d), "Each output trans block should be an instance of nn.ConvTranspose2d"
    
    # Check that box_mlp is an instance of nn.Sequential.
    assert isinstance(pg.box_mlp, nn.Sequential), "box_mlp should be an instance of nn.Sequential"
    
    # Additionally, verify the dimensions of the first linear layer in box_mlp
    fused_channels = 768
    pool_size = (2, 2)
    mlp_in_dim = 4 * fused_channels * pool_size[0] * pool_size[1]
    first_linear = pg.box_mlp[0]
    assert isinstance(first_linear, nn.Linear), "First layer of box_mlp should be a Linear layer"
    assert first_linear.in_features == mlp_in_dim, "The input dimension of the first linear layer is incorrect"
    
if __name__ == '__main__':
    pytest.main([__file__])