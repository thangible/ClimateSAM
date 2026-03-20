from parser_config import parse
import os
import random
import numpy as np
import torch
import os
import random
from typing import List, Tuple, Union, Optional
import torch.distributed as dist

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
                                   tc_pred_mask=None, ar_pred_mask=None, radius=8, save_path='exp', axis=False, title=None):
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    # Custom colormap: 0=black, 1=yellow, 2=blue
    cmap = ListedColormap(['#636363', '#64b163', '#6363ff'])
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(mask, cmap=cmap, vmin=0, vmax=2, alpha=1)

    # Base legend entries for masks/bboxes
    legend_elements = [
        plt.Line2D([0], [0], color='green', lw=0, marker='s', label='TC Groundtruth', markerfacecolor='green', markersize=10, markeredgecolor='black'),
        plt.Line2D([0], [0], color='blue', lw=0, marker='s', label='AR Groundtruth', markerfacecolor='blue', markersize=10),
        plt.Line2D([0], [0], color='red', lw=2, label='AR BBox'),
        plt.Line2D([0], [0], color='cyan', lw=2, label='TC BBox'),
    ]

    csv_rows = []  # collect rows to save: kind,label,x,y,x2,y2 (label carries pos/neg)

    def _extract_points_and_labels(obj):
        """Return (coords_array, labels_array_or_None).
        Supports: - None -> empty
                  - tuple (coords, labels)
                  - single coords tensor/array
                  - list/tuple of per-sample entries -> use first element
        coords returned as shape (N,2); labels as shape (N,) or None
        This version is more robust to object-dtype and oddly-shaped inputs.
        """
        if obj is None:
            return np.empty((0, 2)), None
        # If a list/tuple of samples (not the (coords, labels) format), choose first non-None element
        if isinstance(obj, (list, tuple)) and not (len(obj) == 2 and not isinstance(obj[0], (list, tuple))):
            # try to find a sensible element
            el = None
            for item in obj:
                if item is not None:
                    el = item
                    break
            if el is None:
                return np.empty((0, 2)), None
            # recurse on the chosen element
            return _extract_points_and_labels(el)
        # If it's the expected (coords, labels) tuple
        if isinstance(obj, (list, tuple)) and len(obj) == 2:
            coords_obj, labels_obj = obj
        else:
            coords_obj, labels_obj = obj, None

        # Convert coords to numpy in a safe way
        try:
            if isinstance(coords_obj, torch.Tensor):
                coords = coords_obj.detach().cpu().numpy()
            else:
                coords = np.asarray(coords_obj)
        except Exception:
            # fallback: try to build from iterable
            try:
                coords = np.array(list(coords_obj))
            except Exception:
                return np.empty((0, 2)), None

        # Helper to coerce various shapes into (N,2)
        def _coords_to_pairs(arr):
            if arr is None:
                return np.empty((0, 2))
            arr = np.asarray(arr)
            if arr.size == 0:
                return np.empty((0, 2))
            # handle object dtype (nested lists)
            if arr.dtype == object:
                # try to extract numeric subarrays
                flat_list = []
                for x in np.ravel(arr):
                    if x is None:
                        continue
                    try:
                        xa = np.asarray(x, dtype=float)
                    except Exception:
                        continue
                    if xa.size == 0:
                        continue
                    if xa.ndim == 1 and xa.size == 2:
                        flat_list.append(xa.reshape(1, 2))
                    elif xa.ndim == 2 and xa.shape[1] == 2:
                        flat_list.append(xa)
                    elif xa.size % 2 == 0:
                        flat_list.append(xa.reshape(-1, 2))
                if len(flat_list) == 0:
                    return np.empty((0, 2))
                return np.vstack(flat_list)
            # numeric dtype
            if arr.ndim == 0:
                # scalar -> cannot form point
                return np.empty((0, 2))
            if arr.ndim == 1:
                if arr.size == 2:
                    return arr.reshape(1, 2)
                if arr.size % 2 == 0:
                    return arr.reshape(-1, 2)
                return np.empty((0, 2))
            if arr.ndim == 2:
                # common case: (N,2) or (M,2)
                if arr.shape[1] == 2:
                    return arr
                # if total elements == 2 -> single point
                if arr.size == 2:
                    return arr.reshape(1, 2)
                # try flatten and reshape
                if arr.size % 2 == 0:
                    return arr.reshape(-1, 2)
                return np.empty((0, 2))
            if arr.ndim >= 3:
                # try to collapse leading dims and keep last dim as 2
                if arr.shape[-1] == 2:
                    return arr.reshape(-1, 2)
                if arr.size % 2 == 0:
                    return arr.reshape(-1, 2)
                return np.empty((0, 2))
            return np.empty((0, 2))

        coords = _coords_to_pairs(coords)

        # Handle labels
        if labels_obj is None:
            return coords, None
        try:
            if isinstance(labels_obj, torch.Tensor):
                labs = labels_obj.detach().cpu().numpy()
            else:
                labs = np.asarray(labels_obj)
        except Exception:
            labs = None
        if labs is None or labs.size == 0:
            return coords, None
        labs = np.asarray(labs)
        # Flatten labs and try to align with coords
        labs = labs.reshape(-1)
        if labs.shape[0] != coords.shape[0]:
            # If labs looks per-object while coords contains multiple points per object, attempt repeat
            try:
                if hasattr(coords_obj, 'ndim') and coords_obj.ndim == 3 and labs.shape[0] == coords_obj.shape[0]:
                    labs = np.repeat(labs, coords_obj.shape[1])
                    labs = labs.reshape(-1)
                else:
                    # truncate or pad with ones
                    if labs.shape[0] < coords.shape[0]:
                        pad = np.ones(coords.shape[0] - labs.shape[0], dtype=labs.dtype)
                        labs = np.concatenate([labs[:coords.shape[0]], pad])
                    else:
                        labs = labs[:coords.shape[0]]
            except Exception:
                # final fallback: mark all as positive
                labs = np.ones(coords.shape[0], dtype=float)
        return coords, labs

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

    # Plot function for points with labels
    def _plot_points(coords, labels, pos_color, neg_color, pos_marker='x', neg_marker='o', label_prefix='AR'):
        # coords: (N,2), labels: (N,) or None
        if coords is None or coords.size == 0:
            return
        if labels is None:
            # treat all as positive
            for x, y in coords:
                ax.scatter(x, y, marker=pos_marker, color=pos_color, s=50, linewidth=1)
                csv_rows.append(['point', f'{label_prefix}_pos', float(x), float(y), '', ''])
            # add a legend entry for positives if not present
            legend_elements.append(plt.Line2D([0], [0], marker=pos_marker, color=pos_color, label=f'{label_prefix} Pos', markerfacecolor=pos_color, markersize=8, ls='None'))
            return
        coords = coords.reshape(-1, 2)
        labels = np.asarray(labels).reshape(-1)
        pos_idx = labels > 0.5
        neg_idx = ~pos_idx
        if pos_idx.any():
            for x, y in coords[pos_idx]:
                ax.scatter(x, y, marker=pos_marker, color=pos_color, s=50, linewidth=1)
                csv_rows.append(['point', f'{label_prefix}_pos', float(x), float(y), '', ''])
            legend_elements.append(plt.Line2D([0], [0], marker=pos_marker, color=pos_color, label=f'{label_prefix} Pos', markerfacecolor=pos_color, markersize=8, ls='None'))
        if neg_idx.any():
            for x, y in coords[neg_idx]:
                ax.scatter(x, y, marker=neg_marker, color=neg_color, s=40, linewidth=1)
                csv_rows.append(['point', f'{label_prefix}_neg', float(x), float(y), '', ''])
            legend_elements.append(plt.Line2D([0], [0], marker=neg_marker, color=neg_color, label=f'{label_prefix} Neg', markerfacecolor=neg_color, markersize=8, ls='None'))

    # Plot AR points (positive/negative distinction)
    if ar_points is not None:
        ar_coords, ar_labels = _extract_points_and_labels(ar_points)
        _plot_points(ar_coords, ar_labels, pos_color='red', neg_color='orange', pos_marker='x', neg_marker='o', label_prefix='AR')

    # Plot TC points (positive/negative distinction)
    if tc_points is not None:
        tc_coords, tc_labels = _extract_points_and_labels(tc_points)
        _plot_points(tc_coords, tc_labels, pos_color='cyan', neg_color='magenta', pos_marker='x', neg_marker='o', label_prefix='TC')

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

    # Plot AR prediction mask as filled overlay (alpha=0.6)
    if ar_pred_mask is not None:
        if isinstance(ar_pred_mask, torch.Tensor):
            ar_pred_np = ar_pred_mask.cpu().numpy()
        else:
            ar_pred_np = ar_pred_mask

        # Squeeze to 2D if necessary
        if ar_pred_np.ndim > 2:
            ar_pred_np = ar_pred_np.squeeze()

        # Create a masked array so only predicted pixels are shown (threshold 0.5)
        ar_masked = np.ma.masked_where(ar_pred_np <= 0.5, ar_pred_np)

        # Use a sequential colormap and alpha for filled overlay
        ax.imshow(ar_masked, cmap=ListedColormap(['#feb61f']), alpha=0.5, vmin=0, vmax=1)

        # Legend entry (filled color)
        legend_elements.append(plt.Line2D([0], [0], color='#feb61f', lw=6, alpha=0.5, label='AR Prediction'))

    # Plot TC prediction mask as filled overlay (alpha=0.6)
    if tc_pred_mask is not None:
        if isinstance(tc_pred_mask, torch.Tensor):
            tc_pred_np = tc_pred_mask.cpu().numpy()
        else:
            tc_pred_np = tc_pred_mask

        # Squeeze to 2D if necessary
        if tc_pred_np.ndim > 2:
            tc_pred_np = tc_pred_np.squeeze()

        # Mask out pixels below threshold so overlay only shows predicted pixels
        tc_masked = np.ma.masked_where(tc_pred_np <= 0.5, tc_pred_np)

        # Use a sequential colormap and alpha for filled overlay
        ax.imshow(tc_masked, cmap=ListedColormap(['#dd2ddd']), alpha=0.5, vmin=0, vmax=1)

        legend_elements.append(plt.Line2D([0], [0], color='#dd2ddd', lw=6, alpha=0.6, label='TC Prediction'))

    ax.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1, -0.25), frameon=False, fontsize=14, ncol=4, columnspacing=0.5)
    
    
    # Add title and labels
    if title is None:
        title = "Mask with AR/TC Points, BBoxes and Predictions"
    ax.set_title(title, fontsize=16)

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


def plot_mask_with_points_and_bbox_old(mask, ar_points=None, tc_points=None, ar_bbox=None, tc_bbox=None, 
                                   tc_pred_mask=None, ar_pred_mask=None, radius=8, save_path='exp', axis=False, title = None):
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    # Custom colormap: 0=black, 1=yellow, 2=blue
    cmap = ListedColormap(['black', 'green', 'blue'])
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(mask, cmap=cmap, vmin=0, vmax=2, alpha=0.7)

    # Base legend entries for masks/bboxes
    legend_elements = [
        plt.Line2D([0], [0], color='green', lw=0, marker='s', label='TC Groundtruth', markerfacecolor='green', markersize=10, markeredgecolor='black'),
        plt.Line2D([0], [0], color='blue', lw=0, marker='s', label='AR Groundtruth', markerfacecolor='blue', markersize=10),
        plt.Line2D([0], [0], color='red', lw=2, label='AR BBox'),
        plt.Line2D([0], [0], color='cyan', lw=2, label='TC BBox'),
    ]

    csv_rows = []  # collect rows to save: kind,label,x,y,x2,y2 (label carries pos/neg)

    def _extract_points_and_labels(obj):
        """Return (coords_array, labels_array_or_None).
        Supports: - None -> empty
                  - tuple (coords, labels)
                  - single coords tensor/array
                  - list/tuple of per-sample entries -> use first element
        coords returned as shape (N,2); labels as shape (N,) or None
        This version is more robust to object-dtype and oddly-shaped inputs.
        """
        if obj is None:
            return np.empty((0, 2)), None
        # If a list/tuple of samples (not the (coords, labels) format), choose first non-None element
        if isinstance(obj, (list, tuple)) and not (len(obj) == 2 and not isinstance(obj[0], (list, tuple))):
            # try to find a sensible element
            el = None
            for item in obj:
                if item is not None:
                    el = item
                    break
            if el is None:
                return np.empty((0, 2)), None
            # recurse on the chosen element
            return _extract_points_and_labels(el)
        # If it's the expected (coords, labels) tuple
        if isinstance(obj, (list, tuple)) and len(obj) == 2:
            coords_obj, labels_obj = obj
        else:
            coords_obj, labels_obj = obj, None

        # Convert coords to numpy in a safe way
        try:
            if isinstance(coords_obj, torch.Tensor):
                coords = coords_obj.detach().cpu().numpy()
            else:
                coords = np.asarray(coords_obj)
        except Exception:
            # fallback: try to build from iterable
            try:
                coords = np.array(list(coords_obj))
            except Exception:
                return np.empty((0, 2)), None

        # Helper to coerce various shapes into (N,2)
        def _coords_to_pairs(arr):
            if arr is None:
                return np.empty((0, 2))
            arr = np.asarray(arr)
            if arr.size == 0:
                return np.empty((0, 2))
            # handle object dtype (nested lists)
            if arr.dtype == object:
                # try to extract numeric subarrays
                flat_list = []
                for x in np.ravel(arr):
                    if x is None:
                        continue
                    try:
                        xa = np.asarray(x, dtype=float)
                    except Exception:
                        continue
                    if xa.size == 0:
                        continue
                    if xa.ndim == 1 and xa.size == 2:
                        flat_list.append(xa.reshape(1, 2))
                    elif xa.ndim == 2 and xa.shape[1] == 2:
                        flat_list.append(xa)
                    elif xa.size % 2 == 0:
                        flat_list.append(xa.reshape(-1, 2))
                if len(flat_list) == 0:
                    return np.empty((0, 2))
                return np.vstack(flat_list)
            # numeric dtype
            if arr.ndim == 0:
                # scalar -> cannot form point
                return np.empty((0, 2))
            if arr.ndim == 1:
                if arr.size == 2:
                    return arr.reshape(1, 2)
                if arr.size % 2 == 0:
                    return arr.reshape(-1, 2)
                return np.empty((0, 2))
            if arr.ndim == 2:
                # common case: (N,2) or (M,2)
                if arr.shape[1] == 2:
                    return arr
                # if total elements == 2 -> single point
                if arr.size == 2:
                    return arr.reshape(1, 2)
                # try flatten and reshape
                if arr.size % 2 == 0:
                    return arr.reshape(-1, 2)
                return np.empty((0, 2))
            if arr.ndim >= 3:
                # try to collapse leading dims and keep last dim as 2
                if arr.shape[-1] == 2:
                    return arr.reshape(-1, 2)
                if arr.size % 2 == 0:
                    return arr.reshape(-1, 2)
                return np.empty((0, 2))
            return np.empty((0, 2))

        coords = _coords_to_pairs(coords)

        # Handle labels
        if labels_obj is None:
            return coords, None
        try:
            if isinstance(labels_obj, torch.Tensor):
                labs = labels_obj.detach().cpu().numpy()
            else:
                labs = np.asarray(labels_obj)
        except Exception:
            labs = None
        if labs is None or labs.size == 0:
            return coords, None
        labs = np.asarray(labs)
        # Flatten labs and try to align with coords
        labs = labs.reshape(-1)
        if labs.shape[0] != coords.shape[0]:
            # If labs looks per-object while coords contains multiple points per object, attempt repeat
            try:
                if hasattr(coords_obj, 'ndim') and coords_obj.ndim == 3 and labs.shape[0] == coords_obj.shape[0]:
                    labs = np.repeat(labs, coords_obj.shape[1])
                    labs = labs.reshape(-1)
                else:
                    # truncate or pad with ones
                    if labs.shape[0] < coords.shape[0]:
                        pad = np.ones(coords.shape[0] - labs.shape[0], dtype=labs.dtype)
                        labs = np.concatenate([labs[:coords.shape[0]], pad])
                    else:
                        labs = labs[:coords.shape[0]]
            except Exception:
                # final fallback: mark all as positive
                labs = np.ones(coords.shape[0], dtype=float)
        return coords, labs

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

    # Plot function for points with labels
    def _plot_points(coords, labels, pos_color, neg_color, pos_marker='x', neg_marker='o', label_prefix='AR'):
        # coords: (N,2), labels: (N,) or None
        if coords is None or coords.size == 0:
            return
        if labels is None:
            # treat all as positive
            for x, y in coords:
                ax.scatter(x, y, marker=pos_marker, color=pos_color, s=50, linewidth=1)
                csv_rows.append(['point', f'{label_prefix}_pos', float(x), float(y), '', ''])
            # add a legend entry for positives if not present
            legend_elements.append(plt.Line2D([0], [0], marker=pos_marker, color=pos_color, label=f'{label_prefix} Pos', markerfacecolor=pos_color, markersize=8, ls='None'))
            return
        coords = coords.reshape(-1, 2)
        labels = np.asarray(labels).reshape(-1)
        pos_idx = labels > 0.5
        neg_idx = ~pos_idx
        if pos_idx.any():
            for x, y in coords[pos_idx]:
                ax.scatter(x, y, marker=pos_marker, color=pos_color, s=50, linewidth=1)
                csv_rows.append(['point', f'{label_prefix}_pos', float(x), float(y), '', ''])
            legend_elements.append(plt.Line2D([0], [0], marker=pos_marker, color=pos_color, label=f'{label_prefix} Pos', markerfacecolor=pos_color, markersize=8, ls='None'))
        if neg_idx.any():
            for x, y in coords[neg_idx]:
                ax.scatter(x, y, marker=neg_marker, color=neg_color, s=40, linewidth=1)
                csv_rows.append(['point', f'{label_prefix}_neg', float(x), float(y), '', ''])
            legend_elements.append(plt.Line2D([0], [0], marker=neg_marker, color=neg_color, label=f'{label_prefix} Neg', markerfacecolor=neg_color, markersize=8, ls='None'))

    # Plot AR points (positive/negative distinction)
    if ar_points is not None:
        ar_coords, ar_labels = _extract_points_and_labels(ar_points)
        _plot_points(ar_coords, ar_labels, pos_color='red', neg_color='orange', pos_marker='x', neg_marker='o', label_prefix='AR')

    # Plot TC points (positive/negative distinction)
    if tc_points is not None:
        tc_coords, tc_labels = _extract_points_and_labels(tc_points)
        _plot_points(tc_coords, tc_labels, pos_color='cyan', neg_color='magenta', pos_marker='x', neg_marker='o', label_prefix='TC')

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
        legend_elements.append(plt.Line2D([0], [0], color='orange', lw=0, marker='s', label='AR Prediction', markerfacecolor='orange', markersize=10, markeredgecolor='black'))

    # Plot TC prediction mask as contour lines
    if tc_pred_mask is not None:
        if isinstance(tc_pred_mask, torch.Tensor):
            tc_pred_np = tc_pred_mask.cpu().numpy()
        else:
            tc_pred_np = tc_pred_mask
        
        if tc_pred_np.ndim > 2:
            tc_pred_np = tc_pred_np.squeeze()
        
        # Plot contour lines for TC predictions
        ax.contour(tc_pred_np, levels=[0.5], colors=['magenta'], linewidths=2, linestyles='-')
        legend_elements.append(plt.Line2D([0], [0], color='magenta', lw=0, marker='s', label='TC Prediction', markerfacecolor='magenta', markersize=10, markeredgecolor='black'))

    ax.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1, -0.25), frameon=False, fontsize=14, ncol=4, columnspacing=0.5)
    
    
    # Add title and labels
    if title is None:
        title = "Mask with AR/TC Points, BBoxes and Predictions"
    ax.set_title(title, fontsize=16)

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
                if (item is not None and item[0] is not None)
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

def setup_optimizer_and_scheduler(model, prompter, worker_args):
    """
    Sets up optimizer and scheduler for the prompt generator.
    """
    lr = getattr(worker_args, 'lr', 1e-4)
    weight_decay = getattr(worker_args, 'weight_decay', 1e-4)

    if model is None:
        all_trainable_params = list(p for p in prompter.parameters() if p.requires_grad)
    elif prompter is None:
        all_trainable_params = list(p for p in model.parameters() if p.requires_grad)
    else:
        all_trainable_params = list(p for p in model.parameters() if p.requires_grad) + list(p for p in prompter.parameters() if p.requires_grad)
    

    optimizer = torch.optim.AdamW(
        params=all_trainable_params, lr=lr, weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=worker_args.max_epoch_num, eta_min=1e-5
    )
    return optimizer, scheduler

def setup_device():
    """
    Setup device for training (single GPU only).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def setup_device_and_distributed(worker_id, worker_args):
    gpu_num = len(worker_args.used_gpu)
    world_size = os.environ['WORLD_SIZE'] if 'WORLD_SIZE' in os.environ.keys() else gpu_num
    base_rank = os.environ['RANK'] if 'RANK' in os.environ.keys() else 0
    local_rank = (base_rank * gpu_num) + worker_id
    if gpu_num > 1:
        dist.init_process_group(backend='nccl', init_method=worker_args.dist_url,
                                world_size=world_size, rank=local_rank)
    device = torch.device(f"cuda:{worker_id}")
    torch.cuda.set_device(device)
    return device, local_rank