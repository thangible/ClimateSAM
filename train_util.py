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



def plot_with_projection(image, ar_pred, tc_pred, ar_gt, tc_gt, save_path, use_projection=True, epoch=None, title = None):
    ar_gt_color = 'red'
    tc_gt_color = 'green'
    ar_pred_color = 'orange'
    tc_pred_color = 'blue'
    alpha_gt = 0.5
    alpha_pred = 0.3
    
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
    ax.gridlines(x_inline=True)
    # Plot the RGB image
    ax.imshow(image_np, origin='upper', extent=[-180, 180, -90, 90] if use_projection else None, alpha=0.5)
    ax.set_global()
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE, edgecolor='black')
    
    ar_gt_line = plt.Line2D([0], [0], color=ar_gt_color, linewidth=1, label='AR Ground Truth')
    tc_gt_line = plt.Line2D([0], [0], color=tc_gt_color, linewidth=1, label='TC Ground Truth')
    # Plot the mask contours
    ax.contourf(longitudes, latitudes, ar_gt, colors=ar_gt_color, levels=[0.5, 1.0], alpha = alpha_gt, transform=ccrs.PlateCarree() if use_projection else None)
    ax.contourf(longitudes, latitudes, tc_gt, colors=tc_gt_color, levels=[0.5, 1.0], alpha = alpha_gt, transform=ccrs.PlateCarree() if use_projection else None)
    
    # PREPROCESSING
    ar_gt = ar_gt.cpu().numpy().squeeze() if torch.is_tensor(ar_gt) else ar_gt.squeeze() 
    tc_gt = tc_gt.cpu().numpy().squeeze() if torch.is_tensor(tc_gt) else tc_gt.squeeze()
    if ar_pred is not None:
        ar_pred = ar_pred.detach().cpu().numpy().squeeze() if torch.is_tensor(ar_pred) else ar_pred.squeeze()
        if ar_pred.ndim == 3:
            for pred in ar_pred:
                ax.contourf(longitudes, latitudes, pred, colors=ar_pred_color, levels=[0.5, 1.0], alpha = alpha_pred, transform=ccrs.PlateCarree() if use_projection else None)
        ac_pred_line = plt.Line2D([0], [0], color=ar_pred_color, linewidth=1, label='AC Prediction')

        
    if tc_pred is not None:
        tc_pred = tc_pred.detach().cpu().numpy().squeeze() if torch.is_tensor(tc_pred) else tc_pred.squeeze()
        if tc_pred.ndim == 3:
            for pred in tc_pred:
                ax.contourf(longitudes, latitudes, pred, colors=tc_pred_color, levels=[0.5, 1.0], alpha = alpha_pred, transform=ccrs.PlateCarree() if use_projection else None)
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
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
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
        
        elif key in ["gt_masks", "ar_object_masks", "tc_object_masks"]:
            batch[key] = [
                torch.from_numpy(item).to(device=device, dtype=torch.float32)
                if isinstance(item, np.ndarray)
                else item.to(device=device, dtype=torch.float32) if item is not None else None
                for item in batch[key]
            ]
        elif key in ["ar_bbox_prompts", "tc_bbox_prompts", "ar_mask_prompts", "tc_mask_prompts"]:
            batch[key] = [
                item.to(device=device, dtype=torch.float32) if item is not None else None
                for item in batch[key]
            ]
        elif key in ["ar_point_prompts", "tc_point_prompts"]:
            # points, labels = zip(*batch[key])
            batch[key] = [
                (item[0].to(device=device, dtype=torch.float32),
                 item[1].to(device=device, dtype=torch.float32))
                if item[0] is not None
                else None
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
