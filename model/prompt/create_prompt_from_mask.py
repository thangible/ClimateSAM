import cv2  # ensure cv2 is imported
import numpy as np
import torch
import torch.nn.functional as F

def extract_point_and_bbox_prompts_from_pred_masks(preds: torch.Tensor, device=None, connectivity=8, threshold=50, prompt_type='point', centroid_ratio=0.1):
    """
    Given a prediction tensor (B, 1, H, W), extract prompt information for AR and TC classes.
    Assumes AR class is labeled as 2 and TC as 1.
    Returns a list (length=B) of dictionaries containing the prompt outputs.
    """
    prompt_list = []
    batch = preds.shape[0]
    for i in range(batch):
        # Convert the single-sample tensor to a NumPy array (H x W) for processing
        mask_np = preds[i].squeeze(0).cpu().numpy()  
        # Create binary masks for each class
        ar_mask = (mask_np == 2).astype(np.uint8)
        tc_mask = (mask_np == 1).astype(np.uint8)
        background_mask = (mask_np == 0).astype(np.uint8)
        
        # Extract prompts for each class
        
        ar_object_masks, ar_positive_points, ar_bboxes, ar_noisy_masks = get_prompts_from_binary_mask(ar_mask, connectivity, threshold, centroid_ratio, prompt_type, num_points=5)
        tc_object_masks, tc_positive_points, tc_bboxes, tc_noisy_masks = get_prompts_from_binary_mask(tc_mask, connectivity, threshold, centroid_ratio, prompt_type, num_points=5)
        # background_points = get_negative_point_prompts(background_mask, num_points=5)
        
       
        ar_point_count = ar_positive_points.shape[0] if ar_positive_points is not None else 0
        tc_point_count = tc_positive_points.shape[0] if tc_positive_points is not None else 0
        # background_point_count = background_points.shape[0] if background_points is not None else 0
        
        ar_positive_point_labels = torch.ones(ar_point_count, dtype=torch.float32).unsqueeze(1) if ar_point_count > 0 else None
        # ar_negative_point_labels = torch.zeros(background_point_count + tc_point_count, dtype=torch.float32).unsqueeze(1) if background_point_count > 0 else None
        
        tc_positive_point_labels = torch.ones(tc_point_count, dtype=torch.float32).unsqueeze(1) if tc_point_count > 0 else None
        # tc_negative_point_labels = torch.zeros(background_point_count + ar_point_count, dtype=torch.float32).unsqueeze(1) if background_point_count > 0 else None
        

        
        ar_point_prompts = (ar_positive_points, ar_positive_point_labels)
        tc_point_prompts = (tc_positive_points, tc_positive_point_labels)
       

        prompt_dict = {
            'ar_point_prompts': ar_point_prompts,
            'tc_point_prompts': tc_point_prompts,
            'ar_bbox_prompts': ar_bboxes,
            'tc_bbox_prompts': tc_bboxes,
            'ar_mask_prompts': ar_noisy_masks,
            'tc_mask_prompts': tc_noisy_masks,
            'ar_object_masks': ar_object_masks,
            'tc_object_masks': tc_object_masks,
        }
        prompt_list.append(prompt_dict)
        
    # Convert the list of dictionaries to a dictionary of lists
    prompt_dict = {key: [d[key] if d[key] is not None else None for d in prompt_list] for key in prompt_list[0]}
    # Move the tensors back to the specified device if provided
    # if device is not None:
    #     for key in prompt_dict:
    #         if prompt_dict[key] is not None:
    #             prompt_dict[key] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in prompt_dict[key]]
    return prompt_dict


def get_prompts_from_binary_mask(binary_mask, connectivity=8, threshold=50, centroid_ratio=0.1, prompt_type='point', num_points=5):
    """
    Process a binary mask (H x W, np.array) using connected components.
    Returns object_masks, points/bboxes, and (if prompt_type=='mask') noisy masks.
    """
    # Compute connected components using cv2
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8), connectivity=connectivity)
    
    if num_labels - 1 == 0:  # no objects found (background excluded)
        return None, None, None, None

    object_masks_list = []
    bboxes = [] if prompt_type == 'bbox' else None
    points = [] if prompt_type == 'point' else None

    for obj_index in range(1, num_labels):  # skip background label 0
        stat = stats[obj_index]
        area_in_pixels = stat[4]
        if area_in_pixels >= threshold:
            left, top, width, height, _ = stat
            
            # Create binary mask for the current object
            object_mask = (labels == obj_index).astype(np.uint8)
            object_masks_list.append(object_mask)
            
            if prompt_type == 'bbox':
                right = left + width - 1
                bottom = top + height - 1
                bounding_box = [left, top, right, bottom]
                bboxes.append([bounding_box])
            
            if prompt_type == 'point':
                object_centroid = centroids[obj_index]
                object_points = get_positive_point_prompts(mode='new', object_mask=object_mask, object_centroid=object_centroid, centroid_ratio=centroid_ratio, num_points=5)
                points.append(object_points)

        else:
            return None, None, None, None  # If any object is below threshold, return None
    
    object_masks = torch.from_numpy(np.stack(object_masks_list, axis=0)).to(torch.float32).unsqueeze(1) if object_masks_list else None
    noisy_masks = make_noisy_mask_on_objects(object_masks) if prompt_type == 'mask' else None
    points = torch.from_numpy(np.stack(points, axis=0)).to(torch.float32) if prompt_type == 'point' else None
    bboxes = torch.from_numpy(np.stack(bboxes, axis=0)).to(torch.float32) if prompt_type == 'bbox' else None

    return object_masks, points, bboxes, noisy_masks

def get_positive_point_prompts(object_mask, object_centroid, centroid_ratio, num_points, mode = 'old'):
    if mode == 'old':
        object_points = np.argwhere(object_mask)
        object_centroid = object_centroid[::-1]  # (x, y)
        random_idx = np.random.randint(len(object_points))
        random_point = object_points[random_idx]  
        # With probability based on centroid_ratio, select the centroid
        chosen_point = random_point if np.random.rand() > centroid_ratio else object_centroid
        chosen_point = chosen_point[::-1]
        return [chosen_point]
    
    if mode == 'new':
        object_points = np.argwhere(object_mask)
        extra_points = np.random.choice(object_points, size=num_points, replace=False)
        extra_points = [pt[::-1] for pt in extra_points]
        extra_points.append(object_centroid)
        return extra_points
        
        
# def get_negative_point_prompts(background_mask, num_points):
#     background_points = np.argwhere(background_mask)
#     if len(background_points) == 0:
#         return None
#     extra_points = np.random.choice(background_points, size=num_points, replace=False)
#     extra_points = [pt[::-1] for pt in extra_points]
#     return extra_points
        
        
def make_noisy_mask_on_objects(object_masks, scale_factor: int = 8, noisy_mask_threshold: float = 0.5, h=256, w=256):
    """
    Add noise to the input object masks. Based on Mask Transfiner.
    """
    def get_incoherent_mask(input_masks, h, w):
        mask = input_masks.float()
        mask_small = F.interpolate(mask, (h // scale_factor, w // scale_factor), mode='bilinear', align_corners=False)
        mask_recover = F.interpolate(mask_small, (h, w), mode='bilinear', align_corners=False)
        mask_residue = (mask - mask_recover).abs()
        mask_residue = (mask_residue >= 0.01).float()
        return mask_residue

    if object_masks.ndim() == 3:
        object_masks = object_masks.unsqueeze(1)

    o_m_resized = F.interpolate(object_masks.float(), (h, w), mode='bilinear', align_corners=False)
    mask_noise = torch.randn(o_m_resized.shape) * 1.0
    inc_masks = get_incoherent_mask(o_m_resized, h, w)
    o_m_noisy = ((o_m_resized + mask_noise * inc_masks) > noisy_mask_threshold).float()
    
    return o_m_noisy