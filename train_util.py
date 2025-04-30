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
import torch


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


import cv2
import torch
import numpy as np




def extract_point_and_bbox_prompts_from_climatenet_mask(masks, device = None, connectivity = 8, threshold = 50) -> Tuple[List[Union[Tuple[torch.Tensor, torch.Tensor],
                                                                                                                     None]], 
                                                                                                          List[Union[Tuple[torch.Tensor,torch.Tensor], None]],
                                                                                                          List[Union[torch.Tensor, None]], List[Union[torch.Tensor, None]]]:
    # Convert the PyTorch tensor to a NumPy array
    masks = masks.cpu().numpy() if isinstance(masks, torch.Tensor) else masks
    
    
    ar_masks = [ (mask == 2) for mask in masks ]
    tc_masks = [ (mask == 1)  for mask in masks ]
    #noisy_masks 
    ar_point_prompts, tc_point_prompts, ar_bbox_prompts, tc_bbox_prompts = [], [], [], []

    
    for index in range(len(ar_masks)):
        ar_mask = ar_masks[index]
        tc_mask = tc_masks[index]
        
        ar_positive_points, ar_bboxes = get_point_and_bbox_from_binary_mask(ar_mask, connectivity, threshold)
        tc_positive_points, tc_bboxes = get_point_and_bbox_from_binary_mask(tc_mask, connectivity, threshold)

        
        # print("AR Positive Points:", ar_positive_points)
        # print("AR Bounding Boxes:", ar_bboxes)
        # print("TC Positive Points:", tc_positive_points)
        # print("TC Bounding Boxes:", tc_bboxes)
        # print("AR Positive Points Shape:", ar_positive_points.shape if ar_positive_points is not None else "None")
        # print("AR Bounding Boxes Length:", len(ar_bboxes) if ar_bboxes is not None else "None")
        # print("TC Positive Points Shape:", tc_positive_points.shape if tc_positive_points is not None else "None")
        # print("TC Bounding Boxes Length:", len(tc_bboxes) if tc_bboxes is not None else "None")
        
        ar_point_count = ar_positive_points.shape[1] if ar_positive_points is not None else 0
        tc_point_count = tc_positive_points.shape[1] if tc_positive_points is not None else 0
        
        if ar_positive_points is None and tc_positive_points is None: # IN CASE OF NO PROMPTS
            current_ar_point_prompt = (None, None)
            current_tc_point_prompts = (None, None)
        
        elif ar_positive_points is None: # IN CASE OF TC ONLY
            # create tc_point_labels a tensor of ones with the same length as tc_positive_points
            tc_point_labels = torch.ones(tc_point_count, dtype=torch.float32).unsqueeze(0)
            current_tc_point_prompts: Tuple[torch.Tensor, torch.Tensor] = (tc_positive_points, tc_point_labels)
            current_ar_point_prompt = None
            
        elif tc_positive_points is None: # IN CASE OF AR ONLY
            ar_point_labels = torch.ones(ar_point_count, dtype=torch.float32).unsqueeze(0)

            current_ar_point_prompt: Tuple[torch.Tensor, torch.Tensor] = (ar_positive_points,       ar_point_labels)
            current_tc_point_prompts = None
            
        else: # IN CASE OF BOTH AR AND TC
            ar_point_labels: torch.Tensor = torch.concat([
                torch.ones(ar_point_count, dtype=torch.float32).unsqueeze(0), 
                torch.zeros(tc_point_count, dtype=torch.float32).unsqueeze(0)
            ], axis = 1)
            
            
            
            tc_point_labels: torch.Tensor = torch.concat([
                torch.ones(tc_point_count, dtype=torch.float32).unsqueeze(0), 
                torch.zeros(ar_point_count, dtype=torch.float32).unsqueeze(0)
            ], axis = 1)
            
            ar_points: torch.Tensor =  torch.concat((ar_positive_points, tc_positive_points), axis=1)
            tc_points: torch.Tensor = torch.concat((tc_positive_points, ar_positive_points), axis=1)
            
            
            current_ar_point_prompt: Tuple[torch.Tensor, torch.Tensor] = (ar_points, ar_point_labels)
            current_tc_point_prompts: Tuple[torch.Tensor, torch.Tensor] = (tc_points, tc_point_labels)
            
            # APPEND TO PROMPT LIST
        ar_point_prompts.append(current_ar_point_prompt)
        tc_point_prompts.append(current_tc_point_prompts)
        ar_bbox_prompts.append(ar_bboxes)
        tc_bbox_prompts.append(tc_bboxes)
         # Delete temporary variables that are no longer needed
        del ar_mask, tc_mask, ar_positive_points, ar_bboxes, tc_positive_points, tc_bboxes


    # print("AR Point Prompts:", ar_point_prompts, "AR Point Prompts Length:", len(ar_point_prompts))
    # print("TC Point Prompts:", tc_point_prompts, "TC Point Prompts Length:", len(tc_point_prompts))
    # print("AR Bounding Boxes:", ar_bbox_prompts, "AR Bounding Boxes Length:", len(ar_bbox_prompts))
    # print("TC Bounding Boxes:", tc_bbox_prompts, "TC Bounding Boxes Length:", len(tc_bbox_prompts))
    # ar_point_prompts = postprocess_prompts(ar_point_prompts)
    # tc_point_prompts = postprocess_prompts(tc_point_prompts)
    if device is not None:
        ar_point_prompts = [(prompt[0].to(device), prompt[1].to(device)) if prompt is not None else None for prompt in ar_point_prompts]
        tc_point_prompts = [(prompt[0].to(device), prompt[1].to(device)) if prompt is not None else None for prompt in tc_point_prompts]
        ar_bbox_prompts = [prompt.to(device) if prompt is not None else None for prompt in ar_bbox_prompts]
        tc_bbox_prompts = [prompt.to(device) if prompt is not None else None for prompt in tc_bbox_prompts]
        
    del ar_masks, tc_masks, masks
    
    # Optionally force garbage collection
    import gc
    gc.collect()
    
    return ar_point_prompts, tc_point_prompts, ar_bbox_prompts, tc_bbox_prompts
    

def postprocess_prompts(prompts: List[Optional[Tuple[torch.Tensor, torch.Tensor]]]) -> Tuple[Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]:
    """
    Merges a list of prompts (each is a tuple of (points, labels) or None)
    into a tuple of two lists:
      - list of points (torch.Tensor) if available, else None,
      - list of labels (torch.Tensor) if available, else None.
    """
    all_points: List[Optional[torch.Tensor]] = []
    all_labels: List[Optional[torch.Tensor]]= []
    
    for item in prompts:
        if item is not None:
            points, labels = item
            all_points.append(points)
            all_labels.append(labels)
        else:
            all_points.append(None)
            all_labels.append(None)
            
    points = np.array(all_points, dtype = object) if all_points else None
    labels = np.array(all_labels, dtype = object) if all_labels else None
    return (points, labels)


def get_point_and_bbox_from_binary_mask(binary_mask, connectivity = 8, threshold = 50) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[None, None]]:

    # Convert the PyTorch tensor to a NumPy array
    masks = binary_mask.cpu().numpy() if isinstance(binary_mask, torch.Tensor) else binary_mask
    object_count, integrated_label, per_object_stats, per_label_centroids = cv2.connectedComponentsWithStats(
        image=masks.astype(np.uint8), connectivity=connectivity)
    
    if object_count - 1 == 0: # no objects found
        return None, None
    
    bboxes = []
    points = []
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
            
            rightmost_pixel = leftmost_pixel + width - 1
            bottommost_pixel = topmost_pixel + height - 1
            
            #Bounding box prompt for sam
            bounding_box = [leftmost_pixel, topmost_pixel, rightmost_pixel, bottommost_pixel]
            
            #Point prompts for sam
            # Create a mask for the current object
            object_mask = (integrated_label == obj_index).astype(np.uint8) # binary mask of object
            object_points = np.argwhere(object_mask) # get the coordinates of the object points
            object_centroid = per_label_centroids[obj_index] # get the centroid of the object
            random_point = object_points[np.random.randint(len(object_points))] # get a random point from the object
            if object_centroid in object_points:
                chosen_point = random_point if np.random.rand() < 0.5 else object_centroid
            else:
                chosen_point = random_point
            # Append to list

            bboxes.append(bounding_box)
            points.append(chosen_point)
                
    torch_points = torch.from_numpy(np.array(points)).to(torch.float32)
    torch_bboxes = torch.tensor(bboxes, dtype=torch.float32)
    
    torch_points = torch_points.unsqueeze(0) # anfoderung von prompt_encoder.py
    torch_bboxes = torch_bboxes.unsqueeze(0) # anfoderung von prompt_encoder.py
    return torch_points,torch_bboxes

# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# import torch
# import numpy as np

def plot_with_projection(image, ar_pred, tc_pred, ar_gt, tc_gt, save_path, use_projection=True, epoch=None, title = None):
    ar_gt_color = 'red'
    tc_gt_color = 'green'
    ar_pred_color = 'orange'
    tc_pred_color = 'blue'
    
    # PREPROCESSING
    ar_pred = ar_pred.squeeze() if ar_pred is not None else None
    tc_pred = tc_pred.squeeze() if tc_pred is not None else None
    ar_gt = ar_gt.squeeze() if ar_gt is not None else None
    tc_gt = tc_gt.squeeze() if tc_gt is not None else None
    # height, width = tc_gt.shape[1], image.shape[2]
    # Convert tensors to numpy arrays
    image_np = image.cpu().numpy() if torch.is_tensor(image) else image
    # print("Image shape:", image_np.shape)
    if image_np.shape[0] == 3:
        image_np = np.squeeze(image_np)
        image_np = image_np.transpose(1, 2, 0)
    # Normalize image data to [0, 1] range for imshow
    min_val = image_np.min()
    max_val = image_np.max()
    image_np = (image_np - min_val) / (max_val - min_val)
    # image_np = np.clip(image_np, 0, 1)
    # Resize the image to 1024x1024 using interpolation
    image_np = cv2.resize(image_np, (1152, 768), interpolation=cv2.INTER_LINEAR)
    longitudes = np.linspace(-180, 180, image_np.shape[1])
    latitudes = np.linspace(-90, 90, image_np.shape[0])
    
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()} if use_projection else {})
    
    # Plot the RGB image
    ax.imshow(image_np, origin='upper', extent=[-180, 180, -90, 90] if use_projection else None, alpha=0.5)
    ax.add_feature(cfeature.COASTLINE, edgecolor='black')
    ar_gt_line = plt.Line2D([0], [0], color=ar_gt_color, linewidth=1, label='AR Ground Truth')
    tc_gt_line = plt.Line2D([0], [0], color=tc_gt_color, linewidth=1, label='TC Ground Truth')
    # Plot the mask contours
    ax.contour(longitudes, latitudes, ar_gt, colors=ar_gt_color, linewidths=1, levels=[0.5], transform=ccrs.PlateCarree() if use_projection else None)
    ax.contour(longitudes, latitudes, tc_gt, colors=tc_gt_color, linewidths=1, levels=[0.5], transform=ccrs.PlateCarree() if use_projection else None)
    
    
    
    
    # PREPROCESSING
    ar_gt = ar_gt.cpu().numpy().squeeze() if torch.is_tensor(ar_gt) else ar_gt.squeeze() 
    tc_gt = tc_gt.cpu().numpy().squeeze() if torch.is_tensor(tc_gt) else tc_gt.squeeze()
    if ar_pred is not None:
        ar_pred = ar_pred.detach().cpu().numpy().squeeze() if torch.is_tensor(ar_pred) else ar_pred.squeeze()
        ax.contour(longitudes, latitudes, ar_pred, colors=ar_pred_color, linewidths=1, levels=[0.5], transform=ccrs.PlateCarree() if use_projection else None)
        ac_pred_line = plt.Line2D([0], [0], color=ar_pred_color, linewidth=1, label='AC Prediction')

        
    if tc_pred is not None:
        tc_pred = tc_pred.detach().cpu().numpy().squeeze() if torch.is_tensor(tc_pred) else tc_pred.squeeze()
        ax.contour(longitudes, latitudes, tc_pred, colors=tc_pred_color, linewidths=1, levels=[0.5], transform=ccrs.PlateCarree() if use_projection else None)
        tc_red_line = plt.Line2D([0], [0], color=tc_pred_color, linewidth=1, label='TC Prediction')


    handles = [ar_gt_line, ac_pred_line, tc_gt_line, tc_red_line]
    
    plt.legend(handles=handles, loc='upper right')

    # Add title and labels
    if title is None:
        title = f'World projection with RGB  - Epoch {epoch}'
        
    plt.title(title)
    # Save the plot to a numpy array
    fig.canvas.draw()
    plot_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    plot_array = plot_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    
    return plot_array, title


def calculate_focal_loss(inputs: torch.Tensor, targets: torch.Tensor, gamma: float = 5, alpha: float = 0.75):
    """
    Compute the Focal Loss for binary classification.
    
    Args:
        inputs: A float tensor of arbitrary shape. These are the raw logits.
        targets: A float tensor with the same shape as inputs.
                 Contains binary labels (0 for negative, 1 for positive).
        gamma: Focusing parameter that reduces the loss contribution from easy examples. Default is 2.0.
        alpha: Balancing parameter to balance the importance of positive/negative examples. Default is 0.25.
    
    Returns:
        A scalar focal loss value.
    """
    # Apply sigmoid to get probabilities
    p = inputs.sigmoid()
    # Compute p_t, which is p if target is 1 and (1 - p) otherwise
    p_t = p * targets + (1 - p) * (1 - targets)
    
    # Compute the alpha factor according to targets
    alpha_factor = alpha * targets + (1 - alpha) * (1 - targets)
    # Compute focal weight
    focal_weight = alpha_factor * (1 - p_t).pow(gamma)
    
    # Compute the focal loss
    loss = -focal_weight * torch.log(p_t.clamp(min=1e-8))
    return loss.mean()

def calculate_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    assert inputs.size(0) == targets.size(0)
    inputs = inputs.sigmoid()
    inputs, targets = inputs.flatten(1), targets.flatten(1)

    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()


########### SET UP  ############
def worker_init_fn(worker_id: int, base_seed: int, same_worker_seed: bool = True):
    """
    Set random seed for each worker in DataLoader to ensure the reproducibility.

    """
    seed = base_seed if same_worker_seed else base_seed + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_idle_gpu(gpu_num: int = 1, id_only: bool = True) -> List[GPU]:
    """

    find idle GPUs for distributed learning.

    """
    sorted_gpus = sorted(getGPUs(), key=lambda g: g.memoryUtil)
    if len(sorted_gpus) < gpu_num:
        raise RuntimeError(
            f"Your machine doesn't have enough GPUs ({len(sorted_gpus)}) as you specified ({gpu_num})!")
    sorted_gpus = sorted_gpus[:gpu_num]

    if id_only:
        return [gpu.id for gpu in sorted_gpus]
    else:
        return sorted_gpus



def batch_to_cuda(batch, device):
    for key in batch.keys():
        if key == 'input':
            # input is already a single tensor (B, C, H, W)
            # batch[key] = torch.from_numpy(batch[key])
            batch[key] = batch[key].to(device=device, dtype=torch.float32)
        
        elif key in ["gt_masks"]:
            batch[key] = [
                torch.from_numpy(item).to(device=device, dtype=torch.float32)
                if isinstance(item, np.ndarray)
                else item.to(device=device, dtype=torch.float32) if item is not None else None
                for item in batch[key]
            ]
        elif key in ["ar_point_prompts", "tc_point_prompts", "ar_bbox_prompts", "tc_bbox_prompts"]:
            batch[key] = [
                (torch.from_numpy(item[0]).to(device=device, dtype=torch.float32),
                 torch.from_numpy(item[1]).to(device=device, dtype=torch.float32))
                if isinstance(item, tuple) and item is not None
                else item.to(device=device, dtype=torch.float32) if item is not None else None
                for item in batch[key]
            ]
    return batch


def get_idle_port() -> str:
    """
    find an idle port to used for distributed learning

    """
    pscmd = "netstat -ntl |grep -v Active| grep -v Proto|awk '{print $4}'|awk -F: '{print $NF}'"
    procs = os.popen(pscmd).read()
    procarr = procs.split("\n")
    tt = str(random.randint(15000, 30000))
    if tt not in procarr:
        return tt
    else:
        return get_idle_port()


def set_randomness():
    random.seed(3407)
    np.random.seed(3407)
    torch.manual_seed(3407)
    torch.cuda.manual_seed(3407)
    os.environ['PYTHONHASHSEED'] = str(3407)

    # For more details about 'CUBLAS_WORKSPACE_CONFIG',
    # please refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    if V(torch.version.cuda) >= V("10.2"):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(mode=True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('medium')
