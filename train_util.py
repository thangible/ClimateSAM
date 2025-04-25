from train_parser import parse
import os
import random
import numpy as np
import torch
import os
import random
from typing import List

import numpy as np
import torch
from GPUtil import getGPUs, GPU
from packaging.version import parse as V

import cv2

import cv2
import torch
import numpy as np




def extract_point_and_bbox_prompts_from_climatenet_mask(masks, connectivity = 8, threshold = 50):
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
        
        if len(ar_positive_points) == 0 and len(tc_positive_points) == 0:
            pass
        
        elif len(ar_positive_points) == 0:
            tc_point_labels = np.ones(len(tc_positive_points))
            current_tc_point_prompts = (tc_positive_points, tc_point_labels)
            tc_point_prompts.append(current_tc_point_prompts)
            tc_bbox_prompts.append(np.array(tc_bboxes))
        elif len(tc_positive_points) == 0:
            ar_point_labels = np.ones(len(ar_positive_points))
            current_ar_point_prompt = (ar_positive_points, ar_point_labels)
            ar_point_prompts.append(current_ar_point_prompt)
            ar_bbox_prompts.append(np.array(ar_bboxes))
        else:
            ar_point_labels = np.concatenate([
                np.ones(len(ar_positive_points)), 
                np.zeros(len(tc_positive_points))
            ])
            tc_point_labels = np.concatenate([
                np.ones(len(tc_positive_points)), 
                np.zeros(len(ar_positive_points))
            ])
            ar_points =  np.concatenate((ar_positive_points, tc_positive_points), axis=0)
            tc_points = np.concatenate((tc_positive_points, ar_positive_points), axis=0)
            current_ar_point_prompt = (ar_points, ar_point_labels)
            current_tc_point_prompts = (tc_points, tc_point_labels)
            current_ar_bboxes = np.array(ar_bboxes)
            current_tc_bboxes = np.array(tc_bboxes)
            ar_point_prompts.append(current_ar_point_prompt)
            tc_point_prompts.append(current_tc_point_prompts)
            ar_bbox_prompts.append(current_ar_bboxes)
            tc_bbox_prompts.append(current_tc_bboxes)

    ar_bbox_prompts = np.array(ar_bbox_prompts)
    tc_bbox_prompts = np.array(tc_bbox_prompts)
    
    return ar_point_prompts, tc_point_prompts, ar_bbox_prompts, tc_bbox_prompts
    


def get_point_and_bbox_from_binary_mask(binary_mask, connectivity = 8, threshold = 50):

    # Convert the PyTorch tensor to a NumPy array
    masks = binary_mask.cpu().numpy() if isinstance(binary_mask, torch.Tensor) else binary_mask
    object_count, integrated_label, per_object_stats, per_label_centroids = cv2.connectedComponentsWithStats(
        image=masks.astype(np.uint8), connectivity=connectivity)
    
    if object_count-1 == 0:
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

    return np.array(points), np.array(bboxes)




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
