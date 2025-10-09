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
from matplotlib.colors import ListedColormap
import csv 

def prompt_debug(batch, text, max_x=1152, max_y=768):
    print(f"Debugging prompts for {text}")
    ar_points = batch['ar_point_prompts'] 
    tc_points = batch['tc_point_prompts']
    ar_bbox = batch['ar_bbox_prompts']  
    tc_bbox = batch['tc_bbox_prompts']
    # print(f"AR Points: {ar_points}")
    # print(f"TC Points: {tc_points}")
    # print(f"AR BBoxes: {ar_bbox}")
    # print(f"TC BBoxes: {tc_bbox}")
    
    batch_num = len(ar_points) if ar_points is not None else (len(ar_bbox))
    for i in range(batch_num):
        
        if ar_points[i] is not None:
            ar_points_i = ar_points[i][0].squeeze(1)
            x_ar, y_ar = ar_points_i[:,0], ar_points_i[:,1]
            if x_ar.max() > max_x or y_ar.max() > max_y:
                print(f"AR Points out of bounds: x max {x_ar.max()}, y max {y_ar.max()}")
                print(batch['gt_mask'][i].shape)
        if tc_points[i] is not None:
            tc_points_i = tc_points[i][0].squeeze(1)
            x_tc, y_tc = tc_points_i[:,0], tc_points_i[:,1]
            if x_tc.max() > max_x or y_tc.max() > max_y:
                print(f"TC Points out of bounds: x max {x_tc.max()}, y max {y_tc.max()}")
                print(batch['gt_mask'][i].shape)
                
        ar_bbox_i = ar_bbox[i]
        if ar_bbox_i is not None:
            ar_bbox_i = ar_bbox_i.squeeze(1)
            x_1_ar, y_1_ar, x_2_ar, y_2_ar = ar_bbox_i[:,0], ar_bbox_i[:,1], ar_bbox_i[:,2], ar_bbox_i[:,3]
            if x_1_ar.max() > max_x or y_1_ar.max() > max_y or x_2_ar.max() > max_x or y_2_ar.max() > max_y:
                print(f"AR BBox out of bounds: x1 max {x_1_ar.max()}, y1 max {y_1_ar.max()}, x2 max {x_2_ar.max()}, y2 max {y_2_ar.max()}")
                print(batch['gt_mask'][i].shape)
            
                
        tc_bbox_i = tc_bbox[i]
        if tc_bbox_i is not None:
            tc_bbox_i = tc_bbox_i.squeeze(1)
            x_1_tc, y_1_tc, x_2_tc, y_2_tc = tc_bbox_i[:,0], tc_bbox_i[:,1], tc_bbox_i[:,2], tc_bbox_i[:,3]
            if x_1_tc.max() > max_x or y_1_tc.max() > max_y or x_2_tc.max() > max_x or y_2_tc.max() > max_y:
                print(f"TC BBox out of bounds: x1 max {x_1_tc.max()}, y1 max {y_1_tc.max()}, x2 max {x_2_tc.max()}, y2 max {y_2_tc.max()}")
                print(batch['gt_mask'][i].shape)
        print(batch['gt_mask'][i].shape)
        
    print("Prompt debug check completed.")
        
    


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



def plot_mask_with_points_and_bbox(mask, ar_points=None, tc_points=None, ar_bbox=None, tc_bbox=None, 
                                   tc_pred_mask=None, ar_pred_mask=None, radius=8, save_path='exp', axis=False):
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    # Custom colormap: 0=black, 1=yellow, 2=blue
    cmap = ListedColormap(['black', 'green', 'blue'])
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(mask, cmap=cmap, vmin=0, vmax=2, alpha=0.7)

    legend_elements = [
        plt.Line2D([0], [0], marker='x', color='red', label='AR Point', markerfacecolor='red', markersize=10, markeredgecolor='red'),
        plt.Line2D([0], [0], marker='x', color='cyan', label='TC Point', markerfacecolor='cyan', markersize=10, markeredgecolor='cyan'),
        plt.Line2D([0], [0], color='red', lw=2, label='AR BBox'),
        plt.Line2D([0], [0], color='cyan', lw=2, label='TC BBox'),
        plt.Line2D([0], [0], color='green', lw=0, marker='s', label='TC Groundtruth', markerfacecolor='green', markersize=10, markeredgecolor='black'),
        plt.Line2D([0], [0], color='blue', lw=0, marker='s', label='AR Groundtruth', markerfacecolor='blue', markersize=10),
        # plt.Line2D([0], [0], color='black', lw=0, marker='s', label='Background', markerfacecolor='black', markersize=10)
    ]

    csv_rows = []  # collect rows to save: kind,label,x,y,x2,y2

    def _points_to_array(obj):
        if obj is None:
            return np.empty((0, 2))
        # If list/tuple (batched), use first element like original code
        if isinstance(obj, (list, tuple)):
            obj = obj[0]
        if obj is None:
            return np.empty((0, 2))
        if isinstance(obj, torch.Tensor):
            pts = obj
            if pts.ndim == 3:
                pts = pts.squeeze(1)
            pts = pts.cpu().numpy()
        else:
            pts = np.asarray(obj)
            if pts.ndim == 3:
                pts = pts.squeeze(1)
        if pts.size == 0:
            return np.empty((0, 2))
        pts = pts.reshape(-1, 2)
        return pts

    def _bboxes_to_list(obj):
        bboxes = []
        if obj is None:
            return bboxes
        # If it's a list/tuple of per-sample tensors, iterate through elements
        if isinstance(obj, (list, tuple)):
            for item in obj:
                if item is None:
                    continue
                if isinstance(item, torch.Tensor):
                    arr = item.squeeze(1).cpu().numpy() if item.ndim == 3 else item.cpu().numpy()
                else:
                    arr = np.asarray(item)
                    if arr.ndim == 3:
                        arr = arr.squeeze(1)
                arr = np.atleast_2d(arr.reshape(-1, 4))
                bboxes.extend(arr.tolist())
            return bboxes
        # Single tensor/array
        if isinstance(obj, torch.Tensor):
            arr = obj.squeeze(1).cpu().numpy() if obj.ndim == 3 else obj.cpu().numpy()
        else:
            arr = np.asarray(obj)
            if arr.ndim == 3:
                arr = arr.squeeze(1)
        arr = np.atleast_2d(arr.reshape(-1, 4))
        bboxes.extend(arr.tolist())
        return bboxes

    # Plot AR points
    if ar_points is not None and (not isinstance(ar_points, (list, tuple)) or ar_points[0] is not None):
        points = _points_to_array(ar_points)
        for x, y in points:
            ax.scatter(x, y, marker='x', color='red', s=50, linewidth=1)
            csv_rows.append(['point', 'AR', float(x), float(y), '', ''])

    # Plot TC points
    if tc_points is not None and (not isinstance(tc_points, (list, tuple)) or tc_points[0] is not None):
        points = _points_to_array(tc_points)
        for x, y in points:
            ax.scatter(x, y, marker='x', color='cyan', s=50, linewidth=1)
            csv_rows.append(['point', 'TC', float(x), float(y), '', ''])

    # Plot AR bounding boxes
    if ar_bbox is not None:
        bboxes = _bboxes_to_list(ar_bbox)
        for bbox in bboxes:
            x1, y1, x2, y2 = map(float, bbox)
            width = x2 - x1
            height = y2 - y1
            rect = plt.Rectangle((x1, y1), width, height, linewidth=1, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            csv_rows.append(['bbox', 'AR', x1, y1, x2, y2])

    # Plot TC bounding boxes
    if tc_bbox is not None:
        bboxes = _bboxes_to_list(tc_bbox)
        for bbox in bboxes:
            x1, y1, x2, y2 = map(float, bbox)
            width = x2 - x1
            height = y2 - y1
            rect = plt.Rectangle((x1, y1), width, height, linewidth=1, edgecolor='cyan', facecolor='none')
            ax.add_patch(rect)
            csv_rows.append(['bbox', 'TC', x1, y1, x2, y2])

    # Plot AR prediction mask as contour lines
    if ar_pred_mask is not None:
        if isinstance(ar_pred_mask, torch.Tensor):
            ar_pred_np = ar_pred_mask.cpu().numpy()
        else:
            ar_pred_np = ar_pred_mask
        
        if ar_pred_np.ndim > 2:
            ar_pred_np = ar_pred_np.squeeze()
        
        # Plot contour lines for AR predictions
        ax.contour(ar_pred_np, levels=[0.5], colors=['orange'], linewidths=2, linestyles='--')
        legend_elements.append(plt.Line2D([0], [0], color='orange', lw=2, linestyle='--', label='AR Prediction'))

    # Plot TC prediction mask as contour lines
    if tc_pred_mask is not None:
        if isinstance(tc_pred_mask, torch.Tensor):
            tc_pred_np = tc_pred_mask.cpu().numpy()
        else:
            tc_pred_np = tc_pred_mask
        
        if tc_pred_np.ndim > 2:
            tc_pred_np = tc_pred_np.squeeze()
        
        # Plot contour lines for TC predictions
        ax.contour(tc_pred_np, levels=[0.5], colors=['magenta'], linewidths=2, linestyles='-.')
        legend_elements.append(plt.Line2D([0], [0], color='magenta', lw=2, linestyle='-.', label='TC Prediction'))

    ax.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1, -0.25), frameon=False, fontsize=14, ncol=4, columnspacing=0.5)
    ax.set_title("Mask with AR/TC Points, BBoxes and Predictions", fontsize=16)

    # Show or hide axis/ruler
    if axis:
        h, w = mask.shape
        max_ticks = 10
        x_ticks = np.unique(np.round(np.linspace(0, w - 1, min(max_ticks, w))).astype(int))
        y_ticks = np.unique(np.round(np.linspace(0, h - 1, min(max_ticks, h))).astype(int))
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xticklabels([str(int(t)) for t in x_ticks], fontsize=10)
        ax.set_yticklabels([str(int(t)) for t in y_ticks], fontsize=10)
        ax.tick_params(axis='both', which='major', length=6)
        ax.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.6)
    else:
        ax.axis('off')
    
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)

    # Save prompts (points and bboxes) to a CSV with the same base name as save_path
    csv_path = os.path.splitext(save_path)[0] + '.csv'
    if csv_rows:
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['kind', 'label', 'x', 'y', 'x2', 'y2'])
            writer.writerows(csv_rows)

    plt.close(fig)
    
    return fig

# def plot_mask_with_points_and_bbox(mask, ar_points=None, tc_points=None, ar_bbox=None, tc_bbox=None, 
#                                    tc_pred_mask=None, ar_pred_mask=None, radius=8, save_path='exp'):
#     if isinstance(mask, torch.Tensor):
#         mask = mask.cpu().numpy()
    
#     # Custom colormap: 0=black, 1=yellow, 2=blue
#     cmap = ListedColormap(['black', 'yellow', 'blue'])
#     fig, ax = plt.subplots(figsize=(10, 8))
#     ax.imshow(mask, cmap=cmap, vmin=0, vmax=2, alpha=0.7)

#     legend_elements = [
#         plt.Line2D([0], [0], marker='x', color='w', label='AR Point', markerfacecolor='red', markersize=10, markeredgecolor='black'),
#         plt.Line2D([0], [0], marker='x', color='w', label='TC Point', markerfacecolor='cyan', markersize=10, markeredgecolor='black'),
#         plt.Line2D([0], [0], color='red', lw=2, label='AR BBox'),
#         plt.Line2D([0], [0], color='cyan', lw=2, label='TC BBox'),
#         plt.Line2D([0], [0], color='yellow', lw=0, marker='s', label='TC Groundtruth', markerfacecolor='yellow', markersize=10),
#         plt.Line2D([0], [0], color='blue', lw=0, marker='s', label='AR Groundtruth', markerfacecolor='blue', markersize=10),
#         plt.Line2D([0], [0], color='black', lw=0, marker='s', label='Background', markerfacecolor='black', markersize=10)
#     ]

#     # Plot AR points
#     if ar_points is not None and ar_points[0] is not None:
#         points = ar_points[0]
#         if points.ndim == 3:
#             points = points.squeeze(1)
#         points = points.cpu().numpy()
#         for x, y in points:
#             ax.scatter(x, y, marker='x', color='red', s=100, linewidth=3)

#     # Plot TC points
#     if tc_points is not None and tc_points[0] is not None:
#         points = tc_points[0]
#         if points.ndim == 3:
#             points = points.squeeze(1)
#         points = points.cpu().numpy()
#         for x, y in points:
#             ax.scatter(x, y, marker='x', color='cyan', s=100, linewidth=3)

#     # Plot AR bounding boxes
#     if ar_bbox is not None:
#         if isinstance(ar_bbox, torch.Tensor):
#             bboxes = ar_bbox.squeeze(1).cpu().numpy() if ar_bbox.ndim == 3 else ar_bbox.cpu().numpy()
#             for bbox in bboxes:
#                 x1, y1, x2, y2 = bbox
#                 width = x2 - x1
#                 height = y2 - y1
#                 rect = plt.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='red', facecolor='none')
#                 ax.add_patch(rect)

#     # Plot TC bounding boxes
#     if tc_bbox is not None:
#         if isinstance(tc_bbox, torch.Tensor):
#             bboxes = tc_bbox.squeeze(1).cpu().numpy() if tc_bbox.ndim == 3 else tc_bbox.cpu().numpy()
#             for bbox in bboxes:
#                 x1, y1, x2, y2 = bbox
#                 width = x2 - x1
#                 height = y2 - y1
#                 rect = plt.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='cyan', facecolor='none')
#                 ax.add_patch(rect)

#     # Plot AR prediction mask as contour lines
#     if ar_pred_mask is not None:
#         if isinstance(ar_pred_mask, torch.Tensor):
#             ar_pred_np = ar_pred_mask.cpu().numpy()
#         else:
#             ar_pred_np = ar_pred_mask
        
#         if ar_pred_np.ndim > 2:
#             ar_pred_np = ar_pred_np.squeeze()
        
#         # Plot contour lines for AR predictions
#         contours_ar = ax.contour(ar_pred_np, levels=[0.5], colors=['orange'], linewidths=2, linestyles='--')
#         legend_elements.append(plt.Line2D([0], [0], color='orange', lw=2, linestyle='--', label='AR Prediction'))

#     # Plot TC prediction mask as contour lines
#     if tc_pred_mask is not None:
#         if isinstance(tc_pred_mask, torch.Tensor):
#             tc_pred_np = tc_pred_mask.cpu().numpy()
#         else:
#             tc_pred_np = tc_pred_mask
        
#         if tc_pred_np.ndim > 2:
#             tc_pred_np = tc_pred_np.squeeze()
        
#         # Plot contour lines for TC predictions
#         contours_tc = ax.contour(tc_pred_np, levels=[0.5], colors=['magenta'], linewidths=2, linestyles='-.')
#         legend_elements.append(plt.Line2D([0], [0], color='magenta', lw=2, linestyle='-.', label='TC Prediction'))

#     ax.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1, -0.25), frameon=False, fontsize=14, ncol=4, columnspacing=0.5)
#     ax.set_title("Mask with AR/TC Points, BBoxes and Predictions", fontsize=16)
#     ax.axis('off')
    
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
#     plt.close(fig)
    
#     return fig



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
        
        elif key in ["gt_mask", "ar_object_masks", "tc_object_masks"]:
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
