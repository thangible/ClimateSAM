import random
import numpy as np
import torch
import cv2
import torch.nn.functional as F


class PromptMaker:
    def __init__(self, device=None, connectivity=8, threshold=20, prompt_type='point', centroid_ratio=0.1, positive_point_num =5, negative_point_num=5):
        """
        """

        self.device = device
        self.connectivity = connectivity
        self.threshold = threshold
        self.prompt_type = prompt_type
        self.centroid_ratio = centroid_ratio
        self.positive_point_num = positive_point_num
        self.negative_point_num = negative_point_num

    def make_prompts(self, masks: torch.Tensor):
        batch_size = len(masks)
        prompt_list = []
        
        
        #if prompt_type is a list, randomly choose one for each sample
        if isinstance(self.prompt_type, (list, tuple)):
            if len(self.prompt_type) == 0:
                raise ValueError("prompt_type list must be non-empty")
            prompt_type = random.choice(self.prompt_type)
        else:
            prompt_type = self.prompt_type
            
        for i in range(batch_size):
            mask_np = masks[i].squeeze(0).cpu().numpy() 
            ar_mask = (mask_np == 2).astype(np.uint8)
            tc_mask = (mask_np == 1).astype(np.uint8)   
            # if sum(tc_mask.flatten()) == 0:
            #     import matplotlib.pyplot as plt
            #     fig, ax = plt.subplots(figsize=(6, 6))
            #     ax.imshow(mask_np, cmap='gray', vmin=0, vmax=2)
            #     ax.set_title('Mask with No TC Objects Found')
            if prompt_type == 'point':
                ar_point_prompts, ar_object_masks = make_point_prompts(ar_mask, connectivity=self.connectivity, threshold=self.threshold, num_positive_points=self.positive_point_num, num_negative_points=self.negative_point_num, erode_size= 5)
                tc_point_prompts, tc_object_masks = make_point_prompts(tc_mask, connectivity= self.connectivity, threshold=self.threshold, num_positive_points=min(1, self.positive_point_num//3), num_negative_points=min(1, self.negative_point_num//3), erode_size=1)
                ar_bbox_prompts = None
                tc_bbox_prompts = None
                ar_noisy_masks = None
                tc_noisy_masks = None
            elif prompt_type == 'bbox':
                ar_bbox_prompts, ar_object_masks = make_bbox_prompts(ar_mask, self.connectivity, self.threshold)
                tc_bbox_prompts, tc_object_masks = make_bbox_prompts(tc_mask, self.connectivity, self.threshold)
                ar_point_prompts = (None, None)
                tc_point_prompts = (None, None)
                ar_noisy_masks = None
                tc_noisy_masks = None
            elif prompt_type == 'mask':
                ar_noisy_masks, ar_object_masks =  make_noisy_mask_on_objects(ar_mask, self.connectivity, self.threshold)
                tc_noisy_masks, tc_object_masks =  make_noisy_mask_on_objects(tc_mask, self.connectivity, self.threshold)
                ar_point_prompts = (None, None)
                tc_point_prompts = (None, None)
                ar_bbox_prompts = None
                tc_bbox_prompts = None
            prompt_dict = {
            'ar_point_prompts': ar_point_prompts,
            'tc_point_prompts': tc_point_prompts,
            'ar_bbox_prompts': ar_bbox_prompts,
            'tc_bbox_prompts': tc_bbox_prompts,
            'ar_mask_prompts': ar_noisy_masks,
            'tc_mask_prompts': tc_noisy_masks,
            'ar_object_masks': ar_object_masks,
            'tc_object_masks': tc_object_masks,
        }
            prompt_list.append(prompt_dict)
        prompt_dict = {key: [d[key] if d[key] is not None else None for d in prompt_list] for key in prompt_list[0]}
        return prompt_dict
                
def make_bbox_prompts(binary_mask, connectivity, threshold=20):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8), connectivity=connectivity)
    object_masks_list = []
    bboxes = []
    for obj_index in range(1, num_labels):  # skip background label 0
        stat = stats[obj_index]
        area_in_pixels = stat[4]
        if area_in_pixels >= threshold:
            left, top, width, height, _ = stat
            # Create binary mask for the current object
            object_mask = (labels == obj_index).astype(np.uint8)
            object_masks_list.append(object_mask)
            right = left + width - 1
            bottom = top + height - 1
            bounding_box = [left, top, right, bottom]
            bboxes.append([bounding_box])

    if len(object_masks_list) == 0:
        return None, None
    
    bboxes_prompts = torch.from_numpy(np.stack(bboxes, axis=0)).to(torch.float32)
    object_masks = torch.from_numpy(np.stack(object_masks_list, axis=0)).to(torch.float32).unsqueeze(1) 
    
    return bboxes_prompts, object_masks

def make_point_prompts(binary_mask, connectivity, threshold=20, num_positive_points=5, num_negative_points=5, erode_size=1, dilate_size=15):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8), connectivity=connectivity)
    object_masks_list = []
    positive_points_list = []
    negative_points_list = []
    for obj_index in range(1, num_labels):
        stat = stats[obj_index]
        area_in_pixels = stat[4]
        if area_in_pixels >= threshold:
            # print(area_in_pixels)
            erode_size = max(1, int(np.sqrt(area_in_pixels) * 0.2))
            dilate_size = max(15, int(np.sqrt(area_in_pixels) * 0.1))
            object_masks = (labels == obj_index).astype(np.uint8)
            object_masks_list.append(object_masks)
            object_centroid = centroids[obj_index]
            #POSITIVE POINTS
            positive_points = make_positive_point_prompts(object_mask=object_masks, object_centroid=object_centroid, num_points=num_positive_points, erode_size=erode_size)
            positive_points_list.append(positive_points)
            #NEGATIVE POINTS
            negative_points = make_negative_point_prompts(object_mask=object_masks, object_centroid=object_centroid, num_points=num_negative_points, dilate_size=dilate_size)
            negative_points_list.append(negative_points)
            
            
    object_masks = torch.from_numpy(np.stack(object_masks_list, axis=0)).to(torch.float32).unsqueeze(1) if object_masks_list else None
    
    
    # if len(positive_points_list) == 0:
    #     import matplotlib.pyplot as plt
    #     fig, ax = plt.subplots(figsize=(6, 6))
    #     ax.imshow(binary_mask, cmap='gray', vmin=0, vmax=1)
    #     ax.set_title('Binary Mask with No Objects Found')
    #     ax.axis('off')
    #     plt.tight_layout()
    #     plt.show()
    
    # if len(negative_points_list) == 0:
    #     import matplotlib.pyplot as plt
    #     fig, ax = plt.subplots(figsize=(6, 6))
    #     ax.imshow(binary_mask, cmap='gray', vmin=0, vmax=1)
    #     ax.set_title('Binary Mask with No Objects Found for Negative Points')
    #     ax.axis('off')
    #     plt.tight_layout()
    #     plt.show()
        
    if len(positive_points_list) == 0 or len(negative_points_list) == 0 or len(object_masks_list) == 0:
        return (None, None), None
        
    positive_point_coords = torch.from_numpy(np.stack(positive_points_list, axis=0)).to(torch.float32)
    negative_point_coords = torch.from_numpy(np.stack(negative_points_list, axis=0)).to(torch.float32)
    
    point_coords = torch.cat([positive_point_coords, negative_point_coords], dim=1)
    positive_point_labels = torch.ones(positive_point_coords.shape[:-1], dtype=torch.float32)
    negative_point_labels = torch.zeros(negative_point_coords.shape[:-1], dtype=torch.float32)
    
    point_labels = torch.cat([positive_point_labels, negative_point_labels], dim=1)
    point_prompts = (point_coords, point_labels)
    
    
    return point_prompts, object_masks

            
    
            
            
def make_negative_point_prompts(object_mask, object_centroid, num_points=5, dilate_size=15):
    #dilate it
    
    dilated_mask_small = cv2.dilate(object_mask.astype(np.uint8), kernel=np.ones((dilate_size, dilate_size), np.uint8), iterations=1)
    dilated_mask_big = cv2.dilate(object_mask.astype(np.uint8), kernel=np.ones((int(dilate_size*1.5), int(dilate_size*1.5)), np.uint8), iterations=1)
    negative_region = dilated_mask_big - dilated_mask_small
    region_points = np.argwhere(negative_region)
    selected_indices = np.random.choice(len(region_points), size=num_points, replace=False)
    negative_points = region_points[selected_indices]
    negative_points = [pt[::-1] for pt in negative_points]
    import matplotlib.pyplot as plt

    # # visualize object mask, dilated mask and negative region side-by-side
    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # axes[0].imshow(object_mask, cmap='gray')
    # axes[0].set_title('Object Mask')
    # axes[0].axis('off')

    # axes[1].imshow(dilated_mask_big, cmap='gray')
    # axes[1].set_title(f'Dilated Mask, dilated_size: {dilate_size}')
    # axes[1].axis('off')

    # axes[2].imshow(negative_region, cmap='gray')
    # axes[2].set_title('Negative Region')
    # axes[2].axis('off')

    # plt.tight_layout()
    # plt.show()
    return negative_points
    
    
def make_positive_point_prompts(object_mask, object_centroid, num_points=5, erode_size=1):
    eroded_mask = cv2.erode(object_mask.astype(np.uint8), kernel=np.ones((erode_size,erode_size), np.uint8), iterations=1)
    object_points = np.argwhere(eroded_mask)
    selected_indices = np.random.choice(len(object_points), size=num_points, replace=False)
    positive_points = object_points[selected_indices]
    positive_points = [pt[::-1] for pt in positive_points]
    positive_points.append(object_centroid)
    
    
    # import matplotlib.pyplot as plt

    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # axes[0].imshow(object_mask, cmap='gray')
    # axes[0].set_title('Object Mask')
    # axes[0].axis('off')

    # axes[1].imshow(eroded_mask, cmap='gray')
    # axes[1].set_title(f'Eroded Mask (erode_size={erode_size})')
    # axes[1].axis('off')

    # plt.tight_layout()
    # plt.show()
    return positive_points

def make_noisy_mask_on_objects(binary_mask, connectivity, threshold, scale_factor: int = 8, noisy_mask_threshold: float = 0.5, h=256, w=256):
    """
    Add noise to the input object masks. Based on Mask Transfiner.
    """
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8), connectivity=connectivity)
    object_masks_list = []
    for obj_index in range(1, num_labels):
        stat = stats[obj_index]
        area_in_pixels = stat[4]
        if area_in_pixels >= threshold:
            object_mask = (labels == obj_index).astype(np.uint8)
            object_masks_list.append(object_mask)
    
    if len(object_masks_list) == 0:
        return None, None        
    
    object_masks = torch.from_numpy(np.stack(object_masks_list, axis = 0)).to(torch.float32).unsqueeze(1)

    def get_incoherent_mask(input_masks, h, w):
        mask = input_masks.float()
        mask_small = F.interpolate(mask, (h // scale_factor, w // scale_factor), mode='bilinear', align_corners=False)
        mask_recover = F.interpolate(mask_small, (h, w), mode='bilinear', align_corners=False)
        mask_residue = (mask - mask_recover).abs()
        mask_residue = (mask_residue >= 0.01).float()
        return mask_residue

    if object_masks.dim() == 3:
        object_masks = object_masks.unsqueeze(1)

    o_m_resized = F.interpolate(object_masks.float(), (h, w), mode='bilinear', align_corners=False)
    mask_noise = torch.randn(o_m_resized.shape) * 1.0
    inc_masks = get_incoherent_mask(o_m_resized, h, w)
    o_m_noisy = ((o_m_resized + mask_noise * inc_masks) > noisy_mask_threshold).float()
    
    return o_m_noisy, object_masks