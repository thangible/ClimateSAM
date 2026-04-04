import random
import numpy as np
import torch
import cv2
import torch.nn.functional as F


class PromptMaker:
    def __init__(self, device=None, connectivity=8, threshold=20, prompt_type='point', centroid_ratio=0.1, positive_point_num =5, negative_point_num =1):
        """
        """

        self.device = device
        self.connectivity = connectivity
        self.threshold = threshold
        self.prompt_type = prompt_type
        self.centroid_ratio = centroid_ratio
        self.positive_point_num = positive_point_num
        self.negative_point_num = negative_point_num

    @torch.no_grad()
    def make_prompts(self, multiclass_mask: torch.Tensor = None, ar_mask: torch.Tensor = None, tc_mask: torch.Tensor = None, prompt_type = None, enlarge_ratio=0.1, positive_point_num=None, negative_point_num=None, centroid_ratio=None):
        """
        make_prompts now supports two calling conventions for backward compatibility:
        - Pass a multiclass_mask tensor (B, H, W) or (B,1,H,W) where values are {0,1,2}
        - OR pass ar_mask and tc_mask tensors (each B,1,H,W or B,H,W) with probabilities/logits or binary values.

        The function will prefer ar_mask/tc_mask when both are provided.
        """
        if positive_point_num is not None:
            self.positive_point_num = positive_point_num
        if negative_point_num is not None:
            self.negative_point_num = negative_point_num
        if centroid_ratio is not None:
            self.centroid_ratio = centroid_ratio
        
        # Determine batch size from available inputs
        if ar_mask is not None and tc_mask is not None:
            batch_size = ar_mask.shape[0]
        elif multiclass_mask is not None:
            batch_size = len(multiclass_mask)
        else:
            raise ValueError("make_prompts requires either multiclass_mask or both ar_mask and tc_mask")

        prompt_list = []

        if prompt_type is None:
            prompt_type = self.prompt_type
        else:
            prompt_type = prompt_type

        for i in range(batch_size):
            # If binary masks provided, use them directly (threshold at 0.5 if floats)
            if ar_mask is not None and tc_mask is not None:
                # Extract per-sample masks and ensure 2D arrays (H, W)
                a = ar_mask[i]
                t = tc_mask[i]
                # a/t may have shape (1, H, W) or (H, W) or (1,1,H,W)
                # squeeze all singleton dims to get (H, W)
                if isinstance(a, torch.Tensor):
                    a = a.squeeze().detach().cpu().numpy()
                else:
                    a = np.asarray(a).squeeze()
                if isinstance(t, torch.Tensor):
                    t = t.squeeze().detach().cpu().numpy()
                else:
                    t = np.asarray(t).squeeze()

                # Convert to binary uint8 2D masks
                a_np = (a > 0.5).astype(np.uint8)
                t_np = (t > 0.5).astype(np.uint8)
                ar_mask_np = a_np
                tc_mask_np = t_np
            else:
                # Backward-compatible: derive binary masks from multiclass mask
                m = multiclass_mask[i]
                if isinstance(m, torch.Tensor):
                    mask_np = m.squeeze().cpu().numpy()
                else:
                    mask_np = np.asarray(m).squeeze()
                ar_mask_np = (mask_np == 2).astype(np.uint8)
                tc_mask_np = (mask_np == 1).astype(np.uint8)

            if prompt_type == 'point':
                ar_point_prompts, ar_object_masks = make_point_prompts(ar_mask_np, connectivity=self.connectivity, threshold=self.threshold, num_positive_points=self.positive_point_num, num_negative_points=self.negative_point_num, erode_size= 5)
                tc_point_prompts, tc_object_masks = make_point_prompts(tc_mask_np, connectivity= self.connectivity, threshold=self.threshold, num_positive_points=min(1, self.positive_point_num//3), num_negative_points=min(1, self.negative_point_num//3), erode_size=1)
                ar_bbox_prompts = None
                tc_bbox_prompts = None
                ar_noisy_masks = None
                tc_noisy_masks = None
            elif prompt_type == 'bbox':
                ar_bbox_prompts, ar_object_masks = make_bbox_prompts(ar_mask_np, self.connectivity, self.threshold, enlarge_ratio=enlarge_ratio)
                tc_bbox_prompts, tc_object_masks = make_bbox_prompts(tc_mask_np, self.connectivity, self.threshold, enlarge_ratio=enlarge_ratio)
                ar_point_prompts = (None, None)
                tc_point_prompts = (None, None)
                ar_noisy_masks = None
                tc_noisy_masks = None
            elif prompt_type == 'mask':
                ar_noisy_masks, ar_object_masks =  make_noisy_mask_on_objects(ar_mask_np, self.connectivity, self.threshold)
                tc_noisy_masks, tc_object_masks =  make_noisy_mask_on_objects(tc_mask_np, self.connectivity, self.threshold)
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
                
def make_bbox_prompts(binary_mask, connectivity, threshold=20, enlarge_ratio=0.5):
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
            # enlarge bbox by 10% (clipped to image bounds)
            h_img, w_img = binary_mask.shape
            pad_w = int(round(enlarge_ratio * width))
            pad_h = int(round(enlarge_ratio * height))
            nleft = max(0, left - pad_w)
            ntop = max(0, top - pad_h)
            nright = min(w_img - 1, right + pad_w)
            nbottom = min(h_img - 1, bottom + pad_h)
            bounding_box = [nleft, ntop, nright, nbottom]
            bboxes.append([bounding_box])

    if len(object_masks_list) == 0:
        return None, None
    
    bboxes_prompts = torch.from_numpy(np.stack(bboxes, axis=0)).to(torch.float32)
    object_masks = torch.from_numpy(np.stack(object_masks_list, axis=0)).to(torch.float32).unsqueeze(1) 
    
    return bboxes_prompts, object_masks

def make_point_prompts(binary_mask, connectivity, threshold=20, num_positive_points=5, num_negative_points=0, erode_size=1, dilate_size=15):
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
            if num_negative_points > 0:
                negative_points = make_negative_point_prompts(object_mask=object_masks, object_centroid=object_centroid, num_points=num_negative_points, dilate_size=dilate_size)
                negative_points_list.append(negative_points)

    object_masks = torch.from_numpy(np.stack(object_masks_list, axis=0)).to(torch.float32).unsqueeze(1) if object_masks_list else None
    
    if len(positive_points_list) == 0 or len(object_masks_list) == 0:
        return None, None
        
    # all entries in positive_points_list / negative_points_list are now arrays with shape (N_pos,2) and (N_neg,2)
    positive_point_coords = torch.from_numpy(np.stack(positive_points_list, axis=0)).to(torch.float32)
    positive_point_labels = torch.ones(positive_point_coords.shape[:-1], dtype=torch.float32)
    if num_negative_points > 0:
        negative_point_coords = torch.from_numpy(np.stack(negative_points_list, axis=0)).to(torch.float32)
        negative_point_labels = torch.zeros(negative_point_coords.shape[:-1], dtype=torch.float32)
        point_coords = torch.cat([positive_point_coords, negative_point_coords], dim=1)
        point_labels = torch.cat([positive_point_labels, negative_point_labels], dim=1)
    else:
        point_coords = positive_point_coords
        point_labels = positive_point_labels

    point_prompts = (point_coords, point_labels)
    
    
    return point_prompts, object_masks

            
    
            
def make_negative_point_prompts(object_mask, object_centroid, num_points=5, dilate_size=15):
    #dilate it
    
    dilated_mask_small = cv2.dilate(object_mask.astype(np.uint8), kernel=np.ones((dilate_size, dilate_size), np.uint8), iterations=1)
    dilated_mask_big = cv2.dilate(object_mask.astype(np.uint8), kernel=np.ones((int(dilate_size*1.5), int(dilate_size*1.5)), np.uint8), iterations=1)
    negative_region = dilated_mask_big - dilated_mask_small
    region_points = np.argwhere(negative_region)

    # if negative_region empty -> fall back to background points outside the object
    if len(region_points) == 0:
        bg_points = np.argwhere(object_mask == 0)
        if len(bg_points) == 0:
            # totally degenerate case: return empty array shape (0,2)
            return np.zeros((0, 2), dtype=float)
        replace = num_points > len(bg_points)
        selected = np.random.choice(len(bg_points), size=num_points, replace=replace)
        negative_points = bg_points[selected][:, ::-1].astype(float)  # (x, y)
        return negative_points

    replace = num_points > len(region_points)
    selected_indices = np.random.choice(len(region_points), size=num_points, replace=replace)
    negative_points = region_points[selected_indices][:, ::-1].astype(float)  # ensure shape (num_points, 2) and (x,y)
    return negative_points
    
    
def make_positive_point_prompts(object_mask, object_centroid, num_points=5, erode_size=1):
    eroded_mask = cv2.erode(object_mask.astype(np.uint8), kernel=np.ones((erode_size,erode_size), np.uint8), iterations=1)
    
    if num_points == 1:
        return np.array([object_centroid], dtype=float)
    
    else:
        
        object_points = np.argwhere(eroded_mask)
        
        if len(object_points) == 0:
            object_points = np.argwhere(object_mask)  # fallback to non-eroded mask points if erosion removed all points
            

        # if len(object_points) == 0:
        #     # fallback: use centroid repeated to satisfy requested number
        #     centroid_arr = np.array(object_centroid, dtype=float).reshape(1, 2)
        #     positive_points = np.tile(centroid_arr, (num_points + 1, 1))  # include centroid + requested points
        #     return positive_points

        replace = num_points > len(object_points)  - 1
        selected_indices = np.random.choice(len(object_points), size=num_points, replace=replace)
        positive_points = object_points[selected_indices][:, ::-1].astype(float)  # (num_points, 2) in (x,y)
        centroid_arr = np.array(object_centroid, dtype=float).reshape(1, 2)
        positive_points = np.vstack([positive_points, centroid_arr])  # ensure last entry is centroid
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
    
    # import matplotlib.pyplot as plt

    # o_m_noisy_np = o_m_noisy.detach().cpu().numpy()
    # o_m_resized_np = o_m_resized.detach().cpu().numpy()
    # num_objs = o_m_noisy_np.shape[0]

    # if num_objs > 0:
    #     cols = 2
    #     rows = num_objs
    #     fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, max(3, rows) * 3))
    #     axes = np.atleast_2d(axes)
    #     for i in range(num_objs):
    #         axes[i, 0].imshow(o_m_resized_np[i, 0], cmap='gray', vmin=0, vmax=1)
    #         axes[i, 0].set_title(f'Original #{i}')
    #         axes[i, 0].axis('off')

    #         axes[i, 1].imshow(o_m_noisy_np[i, 0], cmap='gray', vmin=0, vmax=1)
    #         axes[i, 1].set_title(f'Noisy #{i}')
    #         axes[i, 1].axis('off')

    #     plt.tight_layout()
    #     plt.show()
    
    return o_m_noisy, object_masks




def distance_transform(mask: np.ndarray, threshold: float = 0.5):
    """
    Compute the distance transform of a binary mask.
    """
    mask_np = (mask > threshold).astype(np.uint8)
    dist_transform = cv2.distanceTransform(mask_np, cv2.DIST_L2, 5)
    dist_output = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)
    
    return dist_output