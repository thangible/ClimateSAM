from train_parser import parse
import os
import random
import numpy as np
import torch
import os
import random
from typing import List, Tuple, Union, Optional

import numpy as np
import torch
from GPUtil import getGPUs, GPU
from packaging.version import parse as V
import cv2
import torch.nn.functional as F
from torch import Tensor

def extract_point_and_bbox_prompts_from_climatenet_mask(mask: np.array, device = None, connectivity = 8, threshold = 50, prompt_type = 'point', centroid_ratio = 0.1):
    
    # get mask of ar and tc
    ar_mask = mask == 2
    tc_mask = mask == 1
    

    ar_object_masks, ar_positive_points, ar_bboxes, ar_noisy_masks = get_prompts_from_binary_mask(ar_mask, connectivity, threshold, centroid_ratio=centroid_ratio, prompt_type=prompt_type)
    tc_object_masks, tc_positive_points, tc_bboxes, tc_noisy_masks = get_prompts_from_binary_mask(tc_mask, connectivity, threshold, centroid_ratio=centroid_ratio, prompt_type=prompt_type)
    
    # SQUEEZE 

    ar_point_count = ar_positive_points.shape[0] if ar_positive_points is not None else 0
    tc_point_count = tc_positive_points.shape[0] if tc_positive_points is not None else 0
    
    ar_point_labels = torch.ones(ar_point_count, dtype=torch.float32).unsqueeze(1) if ar_point_count > 0 else None
    tc_point_labels = torch.ones(tc_point_count, dtype=torch.float32).unsqueeze(1) if tc_point_count > 0 else None
    
    ar_point_prompts = (ar_positive_points, ar_point_labels)
    tc_point_prompts = (tc_positive_points, tc_point_labels)
    
    # print(f"ar_point_prompts shape: {ar_point_prompts[0].shape}") if ar_point_prompts[0] is not None else None
    # print(f"tc_point_prompts shape: {tc_point_prompts[0].shape}") if tc_point_prompts[0] is not None else None
    # print(f"ar_bbox_prompts shape: {ar_bboxes.shape}") if ar_bboxes is not None else None
    # print(f"tc_bbox_prompts shape: {tc_bboxes.shape}") if tc_bboxes is not None else None
    # print(f"ar_noisy_masks_in_torch shape: {ar_noisy_masks_in_torch.shape}") if ar_noisy_masks_in_torch is not None else None
    # print(f"tc_noisy_masks_in_torch shape: {tc_noisy_masks_in_torch.shape}") if tc_noisy_masks_in_torch is not None else None
    # print(f"ar_object_masks shape: {ar_object_masks.shape}") if ar_object_masks is not None else None
    # print(f"tc_object_masks shape: {tc_object_masks.shape}") if tc_object_masks is not None else None


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
    
    return prompt_dict


def get_prompts_from_binary_mask(binary_mask, connectivity = 8, threshold = 50, centroid_ratio = 0.1, prompt_type = 'point'):

    # Convert the PyTorch tensor to a NumPy array
    masks = binary_mask
    object_count, integrated_label, per_object_stats, per_label_centroids = cv2.connectedComponentsWithStats(
        image=masks.astype(np.uint8), connectivity=connectivity)
    
    if object_count - 1 == 0: # no objects found
        return None, None, None, None
    
    # Create a list to store the object masks
    object_masks = []
    bboxes = [] if prompt_type == 'bbox' else None
    points = [] if prompt_type == 'point' else None
    
    
    for obj_index in range(1, object_count): # loop through each object ignore background
        # stat[0] = leftmost pixel
        # stat[1] = topmost pixel
        # stat[2] = width
        # stat[3] = height
        # stat[4] = area
        stats = per_object_stats[obj_index]
        area_in_pixels = stats[4]
        if area_in_pixels >= threshold:
            # Extract infos
            leftmost_pixel = stats[0]
            topmost_pixel = stats[1]
            width = stats[2]
            height = stats[3]
            
            # OBJECT MASK
            object_mask = (integrated_label == obj_index).astype(np.uint8) # binary mask of object
            object_masks.append(object_mask)
            
            # BBOX PROMPT
            if prompt_type == 'bbox':
                rightmost_pixel = leftmost_pixel + width - 1
                bottommost_pixel = topmost_pixel + height - 1
            
                #Bounding box prompt for sam
                bounding_box = [leftmost_pixel, topmost_pixel, rightmost_pixel, bottommost_pixel]
                bboxes.append([bounding_box]) 
            
            # POINT PROMPT
            if prompt_type == 'point':
                object_points = np.argwhere(object_mask) # get the coordinates of the object points
                object_centroid = per_label_centroids[obj_index] # get the centroid of the object
                random_point = object_points[np.random.randint(len(object_points))] # get a random point from the object
                if object_centroid in object_points:
                    chosen_point = random_point if np.random.rand() > centroid_ratio else object_centroid
                else:
                    chosen_point = random_point
                # Append to list
                points.append([chosen_point])
    
    object_masks = torch.from_numpy(np.stack(object_masks, axis = 0)).to(torch.float32).unsqueeze(1)
    noisy_masks = make_noisy_mask_on_objects(object_masks) if prompt_type == 'noisy_mask' else None
    points = torch.from_numpy(np.stack(points, axis = 0)).to(torch.float32) if prompt_type == 'point' else None
    bboxes = torch.from_numpy(np.stack(bboxes, axis = 0)).to(torch.float32) if prompt_type == 'bbox' else None
    
    return object_masks, points, bboxes, noisy_masks



def make_noisy_mask_on_objects(object_masks, scale_factor: int = 8, noisy_mask_threshold: float = 0.5, h = 256, w = 256):
    """
        Add noise to mask input
        From Mask Transfiner https://github.com/SysCV/transfiner
    """
    def get_incoherent_mask(input_masks, h, w):
        mask = input_masks.float()
        h, w = input_masks.shape[-2:]

        mask_small = F.interpolate(mask, (h // scale_factor, w // scale_factor), mode='bilinear')
        mask_recover = F.interpolate(mask_small, (h, w), mode='bilinear')
        mask_residue = (mask - mask_recover).abs()
        mask_residue = (mask_residue >= 0.01).float()
        return mask_residue
    
    if object_masks.dim() == 3:
        object_masks = object_masks.unsqueeze(1)

    o_m_resized = F.interpolate(object_masks.float(), (h, w), mode='bilinear')
    mask_noise = torch.randn(o_m_resized.shape) * 1.0
    inc_masks = get_incoherent_mask(o_m_resized, h, w)
    o_m_noisy = ((o_m_resized + mask_noise * inc_masks) > noisy_mask_threshold).float()
    
    return o_m_noisy
