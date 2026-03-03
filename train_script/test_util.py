import torch
import numpy as np
import pytest
from train_util import extract_point_and_bbox_prompts_from_climatenet_mask



def create_mask(shape, ar_rect=None, tc_rect=None):
    """
    Create a mask with optional rectangular regions:
    - ar_rect and tc_rect are tuples (x1, y1, x2, y2) where the region is set to 2 (AR) or 1 (TC)
    """
    mask = np.zeros(shape, dtype=np.uint8)
    if ar_rect:
        x1, y1, x2, y2 = ar_rect
        mask[y1:y2+1, x1:x2+1] = 2
    if tc_rect:
        x1, y1, x2, y2 = tc_rect
        mask[y1:y2+1, x1:x2+1] = 1
    return mask

def test_extract_prompts():
    # Set threshold such that the rectangle areas satisfy the conditions (>50)
    shape = (100, 100)
    # Image 1: both AR and TC regions present
    # Create an AR region from (10,10) to (20,20) => 11x11 = 121 pixels
    # Create a TC region from (30,30) to (40,40) => 11x11 = 121 pixels
    ar_rect1 = (10, 10, 20, 20)
    tc_rect1 = (30, 30, 40, 40)
    mask1 = create_mask(shape, ar_rect=ar_rect1, tc_rect=tc_rect1)

    # Image 2: only TC region present (AR remains zero)
    tc_rect2 = (15, 15, 25, 25)  # 11x11 = 121 pixels
    mask2 = create_mask(shape, tc_rect=tc_rect2)

    # Create batch tensor with shape (B, H, W)
    masks_np = np.stack([mask1, mask2], axis=0)
    masks_tensor = torch.from_numpy(masks_np)

    # Call the function under test
    ar_prompts, tc_prompts, ar_bboxes, tc_bboxes = extract_point_and_bbox_prompts_from_climatenet_mask(masks_tensor, threshold=50)

    # print(f"AR Prompts: {ar_prompts}")
    # print(f"TC Prompts: {tc_prompts}")
    # print(f"AR Bounding Boxes: {ar_bboxes}")
    # print(f"TC Bounding Boxes: {tc_bboxes}")
    
    # Check that output lists have a length equal to the batch size
    assert len(ar_prompts) == 2
    assert len(tc_prompts) == 2
    assert len(ar_bboxes) == 2
    assert len(tc_bboxes) == 2

    # For image 1, both AR and TC prompts should be found
    assert ar_prompts[0] is not None
    assert tc_prompts[0] is not None
    # And bounding boxes should be non-empty (since both regions exceed the threshold)
    assert len(ar_bboxes[0]) > 0
    assert len(tc_bboxes[0]) > 0

    # For image 2, only TC prompt should be present (AR is not set)
    assert ar_prompts[1] is None
    assert tc_prompts[1] is not None
    # In this branch, AR bounding boxes should be empty whereas TC bounding boxes are not.
    # (Depending on implementation, ar_bboxes[1] might be an empty list/array.)
    assert (ar_bboxes[1] is None) or (len(ar_bboxes[1]) == 0)
    assert len(tc_bboxes[1]) > 0


if __name__ == '__main__':
    pytest.main([__file__])