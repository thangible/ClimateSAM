import torch
from climatesam import ClimateSAM

def test_mask_shapes():
    # Initialize the model using one of the supported types.
    model = ClimateSAM(model_type='vit_b')
    model.eval()

    # Create a dummy input with 16 channels and the same spatial dimensions as SAM expects.
    # Use the SAM image size from the model for consistency.
    batch_size = 1
    channels = 16
    height, width = model.sam_img_size # 1024, 1024
    dummy_input = torch.randn(batch_size, channels, height, width) #

    # Run a forward pass
    outputs = model(dummy_input)
    
    # Assuming forward() returns (tc_mask, ar_mask)
    tc_mask, ar_mask = outputs

    # Print the shapes
    print("tc_mask shape:", tc_mask.shape)
    print("ar_mask shape:", ar_mask.shape)

    # Check that the two masks have the same shape
    assert tc_mask.shape == ar_mask.shape, "Mismatch: tc_mask and ar_mask must have the same shape!"

if __name__ == "__main__":
    test_mask_shapes()