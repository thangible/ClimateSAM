import torch
from climatesam import ClimateSAM

def test_mask_shapes():
    # Initialize the model using one of the supported types.
    model = ClimateSAM(model_type='vit_b')
    model.eval()

    # Create a dummy input with 16 channels and the same spatial dimensions as SAM expects.
    # Use the SAM image size from the model for consistency.
    batch_size = 3
    channels = 16
    height, width = model.sam_img_size # 1024, 1024
    dummy_input = torch.randn(batch_size, channels, height, width) #

    # Run a forward pass
    outputs = model(dummy_input)
    # tc_dense_embeddings_list, ar_dense_embeddings_list = outputs[0], outputs[1]
    
    # Assuming forward() returns (tc_mask, ar_mask)
    # print(f"TC Dense Embeddings Shape: {tc_dense_embeddings_list[0].shape}")

if __name__ == "__main__":
    test_mask_shapes()