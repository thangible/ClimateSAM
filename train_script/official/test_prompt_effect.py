"""
Script to test the effect of different prompt types on IoU metrics.
Freezes all non-trainable components and evaluates prompt effectiveness.

Prompt configurations:
1. Point prompts: (pos_points, neg_points) pairs
2. BBox prompts: enlarge_ratio variations
3. Mask prompts: direct mask-based prompting
"""

import sys
import os 

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(TRAIN_SCRIPT_DIR)

for p in (PROJECT_ROOT, TRAIN_SCRIPT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
        

import random
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from functools import partial
from torch.utils.data import DataLoader
from utility import batch_to_cuda, get_idle_gpu, get_idle_port, set_randomness, plot_with_projection, plot_mask_with_points_and_bbox, prompt_debug, setup_device_and_distributed, setup_optimizer_and_scheduler, worker_init_fn
from loss_function import ClimateLoss, compute_climate_loss
from tqdm import tqdm
from contextlib import nullcontext
from parser_config import parse
from model.climatesam import ClimateSAM
from dataset.climatenet import ClimateDataset
from evaluator import StreamSegMetrics
import copy
import wandb
from model.prompt.cgnet import CGNetPrompter
from model.prompt.prompt_maker import PromptMaker
import pandas as pd
import json
from datetime import datetime
import torchvision.ops as ops


def profile_prompt_errors(gt_bboxes_list, pred_bboxes_list, iou_threshold=0.3):
    """
    Diagnoses bounding box error topologies.
    """
    stats = {
        'total_gt_objects': 0,
        'total_pred_prompts': 0,
        'missed_objects': 0,      
        'ghost_prompts': 0,       
        'fragmented_objects': 0,  
        'matched_bbox_iou': []    
    }

    for gt, pred in zip(gt_bboxes_list, pred_bboxes_list):
        if gt is None or len(gt) == 0:
            if pred is not None and len(pred) > 0:
                stats['ghost_prompts'] += len(pred)
                stats['total_pred_prompts'] += len(pred)
            continue

        gt = gt.view(-1, 4).float()
        stats['total_gt_objects'] += len(gt)

        if pred is None or len(pred) == 0:
            stats['missed_objects'] += len(gt)
            continue

        pred = pred.view(-1, 4).float()
        stats['total_pred_prompts'] += len(pred)

        iou_matrix = ops.box_iou(gt, pred)

        max_iou_per_gt, _ = iou_matrix.max(dim=1)
        stats['missed_objects'] += (max_iou_per_gt < iou_threshold).sum().item()

        max_iou_per_pred, _ = iou_matrix.max(dim=0)
        stats['ghost_prompts'] += (max_iou_per_pred < iou_threshold).sum().item()

        hits_per_gt = (iou_matrix > iou_threshold).sum(dim=1)
        stats['fragmented_objects'] += (hits_per_gt > 1).sum().item()

        valid_ious = max_iou_per_gt[max_iou_per_gt >= iou_threshold]
        if len(valid_ious) > 0:
            stats['matched_bbox_iou'].extend(valid_ious.tolist())

    return stats


def analyze_scale_ratios(gt_bboxes_list, pred_bboxes_list, device, iou_threshold=0.5):
    """
    Analyzes the empirical scale expansion of predicted bounding boxes 
    compared to ground truth boxes to find the optimal jitter ratio.
    """
    import torchvision.ops as ops
    import numpy as np
    
    all_ratios = []

    for gt, pred in zip(gt_bboxes_list, pred_bboxes_list):
        if gt is None or len(gt) == 0 or pred is None or len(pred) == 0:
            continue

        gt = gt.view(-1, 4).float().to(device)
        pred = pred.view(-1, 4).float().to(device)

        iou_matrix = ops.box_iou(gt, pred)
        
        max_iou_per_gt, best_pred_idx = iou_matrix.max(dim=1)
        
        for i, iou in enumerate(max_iou_per_gt):
            if iou >= iou_threshold:
                matched_gt = gt[i]
                matched_pred = pred[best_pred_idx[i]]
                
                w_gt = matched_gt[2] - matched_gt[0]
                h_gt = matched_gt[3] - matched_gt[1]
                
                w_pr = matched_pred[2] - matched_pred[0]
                h_pr = matched_pred[3] - matched_pred[1]
                
                if w_gt > 0 and h_gt > 0:
                    r_w = max(0, (w_pr / w_gt).item() - 1.0)
                    r_h = max(0, (h_pr / h_gt).item() - 1.0)
                    
                    all_ratios.extend([r_w, r_h])

    stats = {
        'mean': 0.0,
        'p90': 0.0,
        'p95': 0.0,
        'raw_ratios': all_ratios
    }

    if all_ratios:
        stats['mean'] = float(np.mean(all_ratios))
        stats['p90'] = float(np.percentile(all_ratios, 90))
        stats['p95'] = float(np.percentile(all_ratios, 95))
        
    return stats


def freeze_model_parameters(model, trainable_modules=None):
    """
    Freeze all model parameters except those in trainable_modules.
    
    Args:
        model: PyTorch model to freeze
        trainable_modules: List of module names to keep trainable (e.g., ['image_encoder', 'mask_decoder'])
                           If None or empty, all parameters are frozen.
    """
    if trainable_modules is None:
        trainable_modules = []
    
    for name, param in model.named_parameters():
        is_trainable = any(module_name in name for module_name in trainable_modules)
        param.requires_grad = is_trainable
    
    print(f"Trainable modules: {trainable_modules if trainable_modules else 'None (all frozen)'}")


@torch.no_grad()
def validate_cgnet_baseline(val_dataloader, prompter, device, worker_args):
    """
    Validate CGNet baseline by calculating IoU metrics from auxiliary mask predictions.
    
    Args:
        val_dataloader: Validation data loader
        prompter: CGNetPrompter instance for generating auxiliary masks
        device: Device to run on
        worker_args: Worker arguments containing configuration
    
    Returns:
        Dictionary with baseline metrics
    """
    print("\n" + "="*60)
    print("VALIDATING CGNET BASELINE")
    print("="*60)
    
    val_metrics = StreamSegMetrics(class_names=['Background', 'TC', 'AR'])
    
    valid_pbar = tqdm(
        total=len(val_dataloader),
        desc='CGNet Baseline Validation',
        leave=False
    )
    
    with torch.no_grad():
        for val_step, batch in enumerate(val_dataloader):
            batch = batch_to_cuda(batch, device)
            
            features = batch['cgnet_input'].to(device=device, dtype=torch.float32)
            aux_mask = prompter.get_aux_mask(features) 
            
            if aux_mask.dim() == 3:
                combined_preds = aux_mask.unsqueeze(1).unsqueeze(1)
            else:
                combined_preds = torch.argmax(aux_mask, dim=1).unsqueeze(1).unsqueeze(1)
                
            masks_gt = batch['gt_mask'] 
            
            formatted_preds = []
            for i in range(len(masks_gt)):
                pred = combined_preds[i].view_as(masks_gt[i])
                formatted_preds.append(pred)
            
            for masks in [masks_gt, formatted_preds]:
                for i in range(len(masks)):
                    if len(masks[i].shape) == 2:
                        masks[i] = masks[i][None, None, :]
                    if len(masks[i].shape) == 3:
                        masks[i] = masks[i][:, None, :]
                    if len(masks[i].shape) != 4:
                        raise RuntimeError(f"Unexpected mask shape: {masks[i].shape}")
            
            val_metrics.update(formatted_preds, masks_gt, batch['index_name'])
            
            valid_pbar.update(1)
    
    valid_pbar.close()
    
    metric_dict, _ = val_metrics.compute()
    
    baseline_result = {
        'prompt_type': 'cgnet_baseline',
        'positive_point_num': 0,
        'negative_point_num': 0,
        'enlarge_ratio': 0,
        'centroid_ratio': 0,
        'iou_tc': metric_dict['TC IoU'],
        'iou_ar': metric_dict['AR IoU'],
        'iou_bg': metric_dict['Background IoU'],
        'mean_iou': metric_dict['Mean IoU'],
        'mean_foreground_iou': metric_dict['Mean Foreground IoU'],
        'mean_acc': metric_dict['Mean Acc'],
        'overall_acc': metric_dict['Overall Acc'],
        'freqw_acc': metric_dict['FreqW Acc']
    }
    
    print(f"\n✓ CGNet Baseline Results:")
    print(f"  TC IoU: {baseline_result['iou_tc']:.4f}, AR IoU: {baseline_result['iou_ar']:.4f}, Mean FG IoU: {baseline_result['mean_foreground_iou']:.4f}")
    
    return baseline_result

@torch.no_grad()
def validate_with_combined_prompts(
    val_dataloader, 
    val_metrics, 
    model, 
    prompter, 
    device, 
    positive_point_num,
    negative_point_num,
    enlarge_ratio,
    centroid_ratio,
    worker_args,
    prompt_types=None,
    max_samples=None
):
    """
    Validate model with combined prompt types (e.g., point + bbox simultaneously).
    This function generates multiple prompt types and passes them to the model at the same time.
    """
    if prompt_types is None:
        prompt_types = ['point', 'bbox']
    
    model.eval()
    
    prompt_types_str = '+'.join(prompt_types)
    valid_pbar = tqdm(
        total=len(val_dataloader),
        desc=f'Combined Validation ({prompt_types_str}, pos={positive_point_num}, neg={negative_point_num}, enlarge={enlarge_ratio})',
        leave=False
    )
    
    total_samples = 0
    
    with torch.no_grad():
        for val_step, batch in enumerate(val_dataloader):
            if max_samples and total_samples >= max_samples:
                break
            
            batch = batch_to_cuda(batch, device)
            total_samples += batch['input'].shape[0]
            
            features = batch['cgnet_input'].to(device=device, dtype=torch.float32)
            aux_mask = prompter.get_aux_mask(features)
            
            combined_prompt_dict = {
                'ar_point_prompts': None,
                'tc_point_prompts': None,
                'ar_bbox_prompts': None,
                'tc_bbox_prompts': None,
                'ar_mask_prompts': None,
                'tc_mask_prompts': None,
            }
            
            for prompt_type in prompt_types:
                prompt_maker = PromptMaker(
                    prompt_type=prompt_type,
                    positive_point_num=positive_point_num,
                    negative_point_num=negative_point_num,
                    centroid_ratio=centroid_ratio
                )
                
                prompt_dict = prompt_maker.make_prompts(
                    multiclass_mask=aux_mask,
                    prompt_type=prompt_type,
                    positive_point_num=positive_point_num,
                    negative_point_num=negative_point_num,
                    enlarge_ratio=enlarge_ratio,
                    centroid_ratio=centroid_ratio
                )
                
                if prompt_type == 'point':
                    combined_prompt_dict['ar_point_prompts'] = prompt_dict.get('ar_point_prompts')
                    combined_prompt_dict['tc_point_prompts'] = prompt_dict.get('tc_point_prompts')
                elif prompt_type == 'bbox':
                    combined_prompt_dict['ar_bbox_prompts'] = prompt_dict.get('ar_bbox_prompts')
                    combined_prompt_dict['tc_bbox_prompts'] = prompt_dict.get('tc_bbox_prompts')
                elif prompt_type == 'mask':
                    combined_prompt_dict['ar_mask_prompts'] = prompt_dict.get('ar_mask_prompts')
                    combined_prompt_dict['tc_mask_prompts'] = prompt_dict.get('tc_mask_prompts')
            
            for key in combined_prompt_dict:
                if combined_prompt_dict[key] is None:
                    continue
                    
                if key in ["ar_bbox_prompts", "tc_bbox_prompts", "ar_mask_prompts", "tc_mask_prompts"]:
                    combined_prompt_dict[key] = [
                        item.to(device=device, dtype=torch.float32) if item is not None else None
                        for item in combined_prompt_dict[key]
                    ]
                elif key in ["ar_point_prompts", "tc_point_prompts"]:
                    combined_prompt_dict[key] = [
                        (item[0].to(device=device, dtype=torch.float32),
                         item[1].to(device=device, dtype=torch.float32))
                        if (item is not None and item[0] is not None)
                        else None
                        for item in combined_prompt_dict[key]
                    ]
            
            images = model.set_infer_img(batch['input'])
            
            tc_masks, ar_masks = model.infer(
                ar_point_prompts=combined_prompt_dict['ar_point_prompts'],
                tc_point_prompts=combined_prompt_dict['tc_point_prompts'],
                ar_bbox_prompts=combined_prompt_dict['ar_bbox_prompts'],
                tc_bbox_prompts=combined_prompt_dict['tc_bbox_prompts'],
                ar_mask_prompts=combined_prompt_dict.get('ar_mask_prompts'),
                tc_mask_prompts=combined_prompt_dict.get('tc_mask_prompts')
            )
            
            masks_gt = batch['gt_mask']
            
            combined_preds = []
            for i in range(len(masks_gt)):
                pred = torch.zeros_like(masks_gt[i])
                tc_mask_matched = tc_masks[i].view_as(pred)
                ar_mask_matched = ar_masks[i].view_as(pred)
                pred[tc_mask_matched == 1] = 1
                pred[ar_mask_matched == 1] = 2
                combined_preds.append(pred)
                
            for masks in [masks_gt, combined_preds]:
                for i in range(len(masks)):
                    if len(masks[i].shape) == 2:
                        masks[i] = masks[i][None, None, :]
                    if len(masks[i].shape) == 3:
                        masks[i] = masks[i][:, None, :]
                    if len(masks[i].shape) != 4:
                        raise RuntimeError
            
            val_metrics.update(combined_preds, masks_gt, batch['index_name'])
            
            valid_pbar.update(1)
    
    valid_pbar.close()
    
    metric_dict, _ = val_metrics.compute()
    
    results = {
        'prompt_type': prompt_types_str,
        'positive_point_num': positive_point_num,
        'negative_point_num': negative_point_num,
        'enlarge_ratio': enlarge_ratio,
        'centroid_ratio': centroid_ratio,
        'iou_tc': metric_dict['TC IoU'],
        'iou_ar': metric_dict['AR IoU'],
        'iou_bg': metric_dict['Background IoU'],
        'mean_iou': metric_dict['Mean IoU'],
        'mean_foreground_iou': metric_dict['Mean Foreground IoU'],
        'mean_acc': metric_dict['Mean Acc'],
        'overall_acc': metric_dict['Overall Acc'],
        'freqw_acc': metric_dict['FreqW Acc']
    }
    
    val_metrics.reset()
    
    return results

def validate_with_prompt_config(
    val_dataloader, val_metrics, model, prompter, device, 
    prompt_type, positive_point_num, negative_point_num, enlarge_ratio,
    centroid_ratio, worker_args, max_samples=None
):
    model.eval()
    
    total_samples = 0
    valid_pbar = tqdm(
        total=len(val_dataloader), 
        desc=f'Validation (prompt={prompt_type}, pos={positive_point_num}, neg={negative_point_num}, enlarge={enlarge_ratio})',
        leave=False
    )
    
    prompt_maker = PromptMaker(
        prompt_type=prompt_type, positive_point_num=positive_point_num, 
        negative_point_num=negative_point_num, centroid_ratio=centroid_ratio
    )

    if prompt_type == 'bbox':
        ar_diag_totals = {'gt': 0, 'pred': 0, 'missed': 0, 'ghosts': 0, 'fragmented': 0, 'ious': []}
        tc_diag_totals = {'gt': 0, 'pred': 0, 'missed': 0, 'ghosts': 0, 'fragmented': 0, 'ious': []}
        
        ar_scale_ratios = []
        tc_scale_ratios = []

    # Counter for our target visualizations
    vis_saved_count = 0
    max_vis_to_save = 4

    with torch.no_grad():
        for val_step, batch in enumerate(val_dataloader):
            if max_samples and total_samples >= max_samples:
                break
                
            batch = batch_to_cuda(batch, device)
            total_samples += batch['input'].shape[0]
            
            features = batch['cgnet_input'].to(device=device, dtype=torch.float32)
            aux_mask = prompter.get_aux_mask(features)
            
            prompt_dict = prompt_maker.make_prompts(
                multiclass_mask=aux_mask, prompt_type=prompt_type,
                positive_point_num=positive_point_num, negative_point_num=negative_point_num,
                enlarge_ratio=enlarge_ratio, centroid_ratio=centroid_ratio
            )
            
            prompt_dict = batch_to_cuda(prompt_dict, device)

            # ==========================================================
            # VISUALIZATION BLOCK: Save 4 examples for point or bbox
            # ==========================================================
            if prompt_type in ['point', 'bbox'] and vis_saved_count < max_vis_to_save:
                batch_size_b = features.shape[0]
                exp_dir = getattr(worker_args, 'exp_dir', 'exp')
                vis_save_dir = os.path.join(exp_dir, f"{prompt_type}_visualizations")
                os.makedirs(vis_save_dir, exist_ok=True)
                
                for b in range(batch_size_b):
                    if vis_saved_count >= max_vis_to_save:
                        break
                        
                    # Extract single batch ground truth mask
                    gt_mask_b = batch['gt_mask'][b]
                    if isinstance(gt_mask_b, torch.Tensor):
                        gt_mask_b = gt_mask_b.cpu().numpy()
                        
                    # Extract points (if they exist)
                    ar_pts_b = prompt_dict['ar_point_prompts'][b] if prompt_dict.get('ar_point_prompts') is not None else None
                    tc_pts_b = prompt_dict['tc_point_prompts'][b] if prompt_dict.get('tc_point_prompts') is not None else None
                    
                    # Extract bboxes (if they exist)
                    ar_box_b = prompt_dict['ar_bbox_prompts'][b] if prompt_dict.get('ar_bbox_prompts') is not None else None
                    tc_box_b = prompt_dict['tc_bbox_prompts'][b] if prompt_dict.get('tc_bbox_prompts') is not None else None
                    
                    # Create a specific configuration string for the file name
                    cfg_str = f"pos{positive_point_num}_neg{negative_point_num}" if prompt_type == 'point' else f"enlarge{enlarge_ratio}"
                    save_path = os.path.join(vis_save_dir, f"vis_{prompt_type}_{cfg_str}_step{val_step}_idx{b}.png")
                    
                    plot_mask_with_points_and_bbox(
                        mask=gt_mask_b,
                        ar_points=ar_pts_b,
                        tc_points=tc_pts_b,
                        ar_bbox=ar_box_b,
                        tc_bbox=tc_box_b,
                        save_path=save_path,
                        title=f"{prompt_type.capitalize()} Prompt | {cfg_str}"
                    )
                    vis_saved_count += 1
            # ==========================================================


            if prompt_type == 'bbox' and 'ar_bbox_prompts' in batch:
                batch_size = features.shape[0]
                exp_dir = getattr(worker_args, 'exp_dir', 'exp')
                ghost_save_dir = os.path.join(exp_dir, "ghost_visualizations")
                os.makedirs(ghost_save_dir, exist_ok=True)
                
                for b in range(batch_size):
                    gt_ar = batch['ar_bbox_prompts'][b]
                    pr_ar = prompt_dict['ar_bbox_prompts'][b]
                    
                    ghost_count = 0
                    if pr_ar is not None and len(pr_ar) > 0:
                        if gt_ar is None or len(gt_ar) == 0:
                            ghost_count = len(pr_ar)
                        else:
                            gt_flat = gt_ar.view(-1, 4).float().to(device)
                            pr_flat = pr_ar.view(-1, 4).float().to(device)
                            iou_matrix = ops.box_iou(gt_flat, pr_flat)
                            max_iou_per_pred, _ = iou_matrix.max(dim=0)
                            ghost_count = (max_iou_per_pred < 0.3).sum().item()
                    
                    if ghost_count >= 1: 
                        save_path = os.path.join(ghost_save_dir, f"ghost_ar_ratio_{enlarge_ratio}_step_{val_step}_img_{b}.png")
                        
                        plot_mask_with_points_and_bbox(
                            mask=batch['gt_mask'][b].cpu().numpy() if isinstance(batch['gt_mask'][b], torch.Tensor) else batch['gt_mask'][b],
                            ar_bbox=pr_ar.cpu().numpy() if pr_ar is not None else None,
                            tc_bbox=prompt_dict['tc_bbox_prompts'][b].cpu().numpy() if prompt_dict['tc_bbox_prompts'][b] is not None else None,
                            save_path=save_path,
                            title=f"AR Ghosts Found: {ghost_count} (Ratio: {enlarge_ratio})"
                        )
                        
                        if hasattr(worker_args, 'wandb') and worker_args.wandb and wandb.run is not None and ghost_count >= 2:
                            wandb.log({f"ghost_analysis/ratio_{enlarge_ratio}": wandb.Image(save_path)})

                ar_stats = profile_prompt_errors(batch['ar_bbox_prompts'], prompt_dict['ar_bbox_prompts'], iou_threshold=0.3)
                tc_stats = profile_prompt_errors(batch['tc_bbox_prompts'], prompt_dict['tc_bbox_prompts'], iou_threshold=0.3)
                
                ar_scale_stats = analyze_scale_ratios(batch['ar_bbox_prompts'], prompt_dict['ar_bbox_prompts'], device, iou_threshold=0.3)
                tc_scale_stats = analyze_scale_ratios(batch['tc_bbox_prompts'], prompt_dict['tc_bbox_prompts'], device, iou_threshold=0.3)

                ar_scale_ratios.extend(ar_scale_stats['raw_ratios'])
                tc_scale_ratios.extend(tc_scale_stats['raw_ratios'])

                ar_diag_totals['gt'] += ar_stats['total_gt_objects']
                ar_diag_totals['pred'] += ar_stats['total_pred_prompts']
                ar_diag_totals['missed'] += ar_stats['missed_objects']
                ar_diag_totals['ghosts'] += ar_stats['ghost_prompts']
                ar_diag_totals['fragmented'] += ar_stats['fragmented_objects']
                ar_diag_totals['ious'].extend(ar_stats['matched_bbox_iou'])

                tc_diag_totals['gt'] += tc_stats['total_gt_objects']
                tc_diag_totals['pred'] += tc_stats['total_pred_prompts']
                tc_diag_totals['missed'] += tc_stats['missed_objects']
                tc_diag_totals['ghosts'] += tc_stats['ghost_prompts']
                tc_diag_totals['fragmented'] += tc_stats['fragmented_objects']
                tc_diag_totals['ious'].extend(tc_stats['matched_bbox_iou'])
            
            images = model.set_infer_img(batch['input'])
            
            tc_masks, ar_masks = model.infer(
                ar_point_prompts=prompt_dict.get('ar_point_prompts'),
                tc_point_prompts=prompt_dict.get('tc_point_prompts'),
                ar_bbox_prompts=prompt_dict.get('ar_bbox_prompts'),
                tc_bbox_prompts=prompt_dict.get('tc_bbox_prompts')
            )
            
            masks_gt = batch['gt_mask']
            
            combined_preds = []
            for i in range(len(masks_gt)):
                pred = torch.zeros_like(masks_gt[i])
                tc_mask_matched = tc_masks[i].view_as(pred)
                ar_mask_matched = ar_masks[i].view_as(pred)
                pred[tc_mask_matched == 1] = 1
                pred[ar_mask_matched == 1] = 2
                combined_preds.append(pred)
                
            for masks in [masks_gt, combined_preds]:
                for i in range(len(masks)):
                    if len(masks[i].shape) == 2:
                        masks[i] = masks[i][None, None, :]
                    if len(masks[i].shape) == 3:
                        masks[i] = masks[i][:, None, :]
                    if len(masks[i].shape) != 4:
                        raise RuntimeError
            
            val_metrics.update(combined_preds, masks_gt, batch['index_name'])
            
            valid_pbar.update(1)
    
    valid_pbar.close()
    
    metric_dict, _ = val_metrics.compute()
    
    results = {
        'prompt_type': prompt_type,
        'positive_point_num': positive_point_num,
        'negative_point_num': negative_point_num,
        'enlarge_ratio': enlarge_ratio,
        'centroid_ratio': centroid_ratio,
        'iou_tc': metric_dict['TC IoU'],
        'iou_ar': metric_dict['AR IoU'],
        'iou_bg': metric_dict['Background IoU'],
        'mean_iou': metric_dict['Mean IoU'],
        'mean_foreground_iou': metric_dict['Mean Foreground IoU'],
        'mean_acc': metric_dict['Mean Acc'],
        'overall_acc': metric_dict['Overall Acc'],
        'freqw_acc': metric_dict['FreqW Acc']
    }

    if prompt_type == 'bbox' and 'ar_diag_totals' in locals():
        avg_ar_bbox_iou = np.mean(ar_diag_totals['ious']) if ar_diag_totals['ious'] else 0.0
        avg_tc_bbox_iou = np.mean(tc_diag_totals['ious']) if tc_diag_totals['ious'] else 0.0
        avg_ar_scale = float(np.mean(ar_scale_ratios)) if ar_scale_ratios else 0.0
        avg_tc_scale = float(np.mean(tc_scale_ratios)) if tc_scale_ratios else 0.0
        
        results.update({
            'diag_ar_avg_bbox_iou': avg_ar_bbox_iou,
            'diag_ar_avg_scale': avg_ar_scale,
            'diag_ar_missed_pct': ar_diag_totals['missed'] / max(1, ar_diag_totals['gt']),
            'diag_ar_ghost_pct': ar_diag_totals['ghosts'] / max(1, ar_diag_totals['pred']),
            'diag_tc_avg_bbox_iou': avg_tc_bbox_iou,
            'diag_tc_avg_scale': avg_tc_scale,
            'diag_tc_missed_pct': tc_diag_totals['missed'] / max(1, tc_diag_totals['gt']),
            'diag_tc_ghost_pct': tc_diag_totals['ghosts'] / max(1, tc_diag_totals['pred']),
        })

        if hasattr(worker_args, 'wandb') and worker_args.wandb and wandb.run is not None:
            wandb.log({
                f"diag/ar_bbox_iou_ratio_{enlarge_ratio}": results['diag_ar_avg_bbox_iou'],
                f"diag/ar_scale_ratio_{enlarge_ratio}": results['diag_ar_avg_scale'],
                f"diag/ar_missed_pct_ratio_{enlarge_ratio}": results['diag_ar_missed_pct'],
                f"diag/ar_ghost_pct_ratio_{enlarge_ratio}": results['diag_ar_ghost_pct'],
                f"diag/tc_bbox_iou_ratio_{enlarge_ratio}": results['diag_tc_avg_bbox_iou'],
                f"diag/tc_scale_ratio_{enlarge_ratio}": results['diag_tc_avg_scale'],
                f"diag/tc_missed_pct_ratio_{enlarge_ratio}": results['diag_tc_missed_pct'],
                f"diag/tc_ghost_pct_ratio_{enlarge_ratio}": results['diag_tc_ghost_pct'],
            })
    
    val_metrics.reset()
    
    return results


def main_worker(worker_id, worker_args):
    """
    Main worker function to test prompt effects on IoU.
    """
    set_randomness()
    
    if isinstance(worker_id, str):
        worker_id = int(worker_id)
    
    device, local_rank = setup_device_and_distributed(worker_id, worker_args)
    print(f"Worker {worker_id} initialized on device {device} with local_rank {local_rank}.")
    
    # ==================== PREPARE DATASET ====================
    dataset_dir = worker_args.data_dir
    val_dataset = ClimateDataset(
        data_dir=dataset_dir, 
        train_flag=False, 
        augmented=False, 
        prompt_type='bbox',
        generate_prompt=False
    )
    val_collate_fn = val_dataset.collate_fn
    
    if hasattr(worker_args, 'debugging') and worker_args.debugging:
        debug_val_size = getattr(worker_args, 'debug_val_size', 20)
        val_indices = list(range(min(debug_val_size, len(val_dataset))))
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
        print(f"Debug mode: Using only {len(val_dataset)} validation samples")
    
    val_bs = worker_args.val_bs if worker_args.val_bs else 2
    val_workers = getattr(worker_args, 'num_workers', 2)
    
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=val_bs,
        shuffle=False,
        num_workers=val_workers,
        drop_last=False,
        collate_fn=val_collate_fn,
        worker_init_fn=partial(worker_init_fn, base_seed=3407)
    )
    
    # ==================== SET UP MODEL ====================
    climatesam = ClimateSAM(
        model_type=worker_args.sam_type,
        mlp_ratio=worker_args.image_encoder_mlp_ratio,
        enable_wandb_logging=False
    ).to(device=device)
    
    image_encoder_path = os.path.join(
        worker_args.exp_dir, 
        f"{worker_args.encoder_weights_name}.pth"
    )
    
    if not os.path.exists(image_encoder_path):
        raise FileNotFoundError(
            f"Pretrained weights not found at {image_encoder_path}. "
            "Please check the path and try again."
        )
    
    phase_1_checkpoint = torch.load(image_encoder_path, map_location=device)
    print(f"Pretrained weights from phase 1 loaded from {image_encoder_path}")
    
    if 'image_encoder' in phase_1_checkpoint:
        climatesam.image_encoder.load_state_dict(phase_1_checkpoint['image_encoder'])
        print(f"✓ Image encoder weights loaded")
    else:
        raise ValueError("Image encoder weights not found in checkpoint.")
    
    if 'mask_decoder' in phase_1_checkpoint:
        climatesam.mask_decoder.load_state_dict(phase_1_checkpoint['mask_decoder'])
        print(f"✓ Mask decoder weights loaded")
    else:
        raise ValueError("Mask decoder weights not found in checkpoint.")
    
    if 'input_adapter' in phase_1_checkpoint:
        climatesam.input_adapter.load_state_dict(phase_1_checkpoint['input_adapter'])
        print(f"✓ Input adapter weights loaded")
    else:
        raise ValueError("Input adapter weights not found in checkpoint.")
    
    # ==================== FREEZE MODEL PARAMETERS ====================
    print("\n" + "="*60)
    print("FREEZING MODEL PARAMETERS")
    print("="*60)
    
    trainable_modules = []
    if hasattr(worker_args, 'trainable_modules'):
        trainable_modules = worker_args.trainable_modules
    
    freeze_model_parameters(climatesam, trainable_modules=trainable_modules)
    
    total_params = sum(p.numel() for p in climatesam.parameters())
    frozen_params = sum(p.numel() for p in climatesam.parameters() if not p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Frozen parameters: {frozen_params:,}")
    print(f"Trainable parameters: {total_params - frozen_params:,}")
    
    # ==================== SET UP PROMPTER ====================
    save_path = os.path.join(worker_args.exp_dir, "cgnet_weight.pth")
    prompter = CGNetPrompter(
        weights_path=save_path, 
        device=device, 
        worker_args=worker_args
    )
    
    # ==================== VALIDATE CGNET BASELINE ====================
    baseline_result = validate_cgnet_baseline(val_dataloader, prompter, device, worker_args)
    all_results = [baseline_result]
    
    # ==================== DEFINE PROMPT CONFIGURATIONS ====================
    print("\n" + "="*60)
    print("PROMPT TEST CONFIGURATIONS")
    print("="*60)
    
    point_configs = [
        {'positive_point_num': 1, 'negative_point_num': 1},
        {'positive_point_num': 1, 'negative_point_num': 2},
        {'positive_point_num': 2, 'negative_point_num': 2},
        {'positive_point_num': 1, 'negative_point_num': 3},
        {'positive_point_num': 5, 'negative_point_num': 5},
        {'positive_point_num': 10, 'negative_point_num': 10},
        {'positive_point_num': 5, 'negative_point_num': 10},
        {'positive_point_num': 15, 'negative_point_num': 5},
        {'positive_point_num': 10, 'negative_point_num': 5},
        {'positive_point_num': 20, 'negative_point_num': 10},
        {'positive_point_num': 20, 'negative_point_num': 5},
        {'positive_point_num': 20, 'negative_point_num': 20},
    ]
    
    bbox_configs = [
        {'enlarge_ratio': 0.0},
        {'enlarge_ratio': 0.1},
        {'enlarge_ratio': 0.2},
        {'enlarge_ratio': 0.3},
        {'enlarge_ratio': 0.4},
        {'enlarge_ratio': 0.5},
        {'enlarge_ratio': -0.1},
        {'enlarge_ratio': -0.2},
    ]
    
    mask_configs = [{}]
    
    # ==================== TEST POINT PROMPTS ====================
    print("\n" + "="*60)
    print("TESTING POINT PROMPTS")
    print("="*60)
    
    for config in point_configs:
        print(f"\nTesting Point Prompt: pos={config['positive_point_num']}, neg={config['negative_point_num']}")
        
        val_metrics = StreamSegMetrics(class_names=['Background', 'TC', 'AR'])
        
        results = validate_with_prompt_config(
            val_dataloader=val_dataloader,
            val_metrics=val_metrics,
            model=climatesam,
            prompter=prompter,
            device=device,
            prompt_type='point',
            positive_point_num=config['positive_point_num'],
            negative_point_num=config['negative_point_num'],
            enlarge_ratio=0,
            centroid_ratio=0,
            worker_args=worker_args,
            max_samples=None
        )
        
        all_results.append(results)
        print(f"  TC IoU: {results['iou_tc']:.4f}, AR IoU: {results['iou_ar']:.4f}, BG IoU: {results['iou_bg']:.4f}, Mean IoU: {results['mean_iou']:.4f}, Mean FG IoU: {results['mean_foreground_iou']:.4f}")
        
        if worker_args.wandb:
            wandb.log({
                'point_prompt/iou_tc': results['iou_tc'],
                'point_prompt/iou_ar': results['iou_ar'],
                'point_prompt/iou_bg': results['iou_bg'],
                'point_prompt/mean_iou': results['mean_iou'],
                'point_prompt/mean_foreground_iou': results['mean_foreground_iou'],
                'point_prompt/config': f"pos={config['positive_point_num']}_neg={config['negative_point_num']}"
            })
    
    # ==================== TEST BBOX PROMPTS ====================
    print("\n" + "="*60)
    print("TESTING BBOX PROMPTS")
    print("="*60)
    
    for config in bbox_configs:
        print(f"\nTesting BBox Prompt: enlarge_ratio={config['enlarge_ratio']}")
        
        val_metrics = StreamSegMetrics(class_names=['Background', 'TC', 'AR'])
        
        results = validate_with_prompt_config(
            val_dataloader=val_dataloader,
            val_metrics=val_metrics,
            model=climatesam,
            prompter=prompter,
            device=device,
            prompt_type='bbox',
            positive_point_num=0,
            negative_point_num=0,
            enlarge_ratio=config['enlarge_ratio'],
            centroid_ratio=0,
            worker_args=worker_args,
            max_samples=None
        )
        
        all_results.append(results)
        print(f"  TC IoU: {results['iou_tc']:.4f}, AR IoU: {results['iou_ar']:.4f}, BG IoU: {results['iou_bg']:.4f}, Mean IoU: {results['mean_iou']:.4f}, Mean FG IoU: {results['mean_foreground_iou']:.4f}")
        
        if worker_args.wandb:
            wandb.log({
                'bbox_prompt/iou_tc': results['iou_tc'],
                'bbox_prompt/iou_ar': results['iou_ar'],
                'bbox_prompt/iou_bg': results['iou_bg'],
                'bbox_prompt/mean_iou': results['mean_iou'],
                'bbox_prompt/mean_foreground_iou': results['mean_foreground_iou'],
                'bbox_prompt/enlarge_ratio': config['enlarge_ratio']
            })
    
    # ==================== TEST MASK PROMPTS ====================
    print("\n" + "="*60)
    print("TESTING MASK PROMPTS")
    print("="*60)
    
    for config in mask_configs:
        print(f"\nTesting Mask Prompt")
        
        val_metrics = StreamSegMetrics(class_names=['Background', 'TC', 'AR'])
        
        results = validate_with_prompt_config(
            val_dataloader=val_dataloader,
            val_metrics=val_metrics,
            model=climatesam,
            prompter=prompter,
            device=device,
            prompt_type='mask',
            positive_point_num=0,
            negative_point_num=0,
            enlarge_ratio=0,
            centroid_ratio=0,
            worker_args=worker_args,
            max_samples=None
        )
        
        all_results.append(results)
        print(f"  TC IoU: {results['iou_tc']:.4f}, AR IoU: {results['iou_ar']:.4f}, BG IoU: {results['iou_bg']:.4f}, Mean IoU: {results['mean_iou']:.4f}, Mean FG IoU: {results['mean_foreground_iou']:.4f}")
        
        if worker_args.wandb:
            wandb.log({
                'mask_prompt/iou_tc': results['iou_tc'],
                'mask_prompt/iou_ar': results['iou_ar'],
                'mask_prompt/iou_bg': results['iou_bg'],
                'mask_prompt/mean_iou': results['mean_iou'],
                'mask_prompt/mean_foreground_iou': results['mean_foreground_iou'],
            })
    
    # ==================== TEST COMBINED PROMPTS ====================
    print("\n" + "="*60)
    print("TESTING COMBINED PROMPTS (Point + BBox + Mask)")
    print("="*60)
    
    combined_configs = [
        {
            'prompt_types': ['point', 'bbox'],
            'positive_point_num': 10,
            'negative_point_num': 5,
            'enlarge_ratio': 0.0,
        },
        {
            'prompt_types': ['point', 'bbox'],
            'positive_point_num': 15,
            'negative_point_num': 10,
            'enlarge_ratio': 0.0,
        },
        {
            'prompt_types': ['point', 'bbox', 'mask'],
            'positive_point_num': 10,
            'negative_point_num': 5,
            'enlarge_ratio': 0.0,
        },
        {
            'prompt_types': ['point', 'bbox'],
            'positive_point_num': 20,
            'negative_point_num': 10,
            'enlarge_ratio': 0.0,
        },
    ]
    
    for config in combined_configs:
        prompt_types = config['prompt_types']
        prompt_types_str = '+'.join(prompt_types)
        print(f"\nTesting Combined: {prompt_types_str}")
        print(f"  Config: pos={config['positive_point_num']}, neg={config['negative_point_num']}, enlarge={config['enlarge_ratio']}")
        
        val_metrics = StreamSegMetrics(class_names=['Background', 'TC', 'AR'])
        
        results = validate_with_combined_prompts(
            val_dataloader=val_dataloader,
            val_metrics=val_metrics,
            model=climatesam,
            prompter=prompter,
            device=device,
            positive_point_num=config['positive_point_num'],
            negative_point_num=config['negative_point_num'],
            enlarge_ratio=config['enlarge_ratio'],
            centroid_ratio=0,
            worker_args=worker_args,
            prompt_types=prompt_types,
            max_samples=None
        )
        
        all_results.append(results)
        print(f"  ✓ TC IoU: {results['iou_tc']:.4f}, AR IoU: {results['iou_ar']:.4f}, Mean FG IoU: {results['mean_foreground_iou']:.4f}")
        print(f"  ✓ Mean Acc: {results['mean_acc']:.4f}, Overall Acc: {results['overall_acc']:.4f}")
        
        if worker_args.wandb:
            wandb.log({
                'combined_prompts/iou_tc': results['iou_tc'],
                'combined_prompts/iou_ar': results['iou_ar'],
                'combined_prompts/iou_bg': results['iou_bg'],
                'combined_prompts/mean_iou': results['mean_iou'],
                'combined_prompts/mean_foreground_iou': results['mean_foreground_iou'],
                'combined_prompts/mean_acc': results['mean_acc'],
                'combined_prompts/overall_acc': results['overall_acc'],
                'combined_prompts/config': f"{prompt_types_str}_pos={config['positive_point_num']}_neg={config['negative_point_num']}_enlarge={config['enlarge_ratio']}"
            })
    
    # ==================== SAVE RESULTS ====================
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    results_df = pd.DataFrame(all_results)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(
        worker_args.exp_dir, 
        worker_args.run_name,
        f"prompt_effect_results_{timestamp}.csv"
    )
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to: {results_path}")
    
    json_path = os.path.join(
        worker_args.exp_dir, 
        worker_args.run_name,
        f"prompt_effect_report_{timestamp}.json"
    )
    with open(json_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'model_config': {
                'sam_type': worker_args.sam_type,
                'encoder_weights': worker_args.encoder_weights_name,
            },
            'results': all_results,
            'summary_stats': {
                'point_prompts': {
                    'avg_iou_tc': results_df[results_df['prompt_type'] == 'point']['iou_tc'].mean() if len(results_df[results_df['prompt_type'] == 'point']) > 0 else None,
                    'avg_iou_ar': results_df[results_df['prompt_type'] == 'point']['iou_ar'].mean() if len(results_df[results_df['prompt_type'] == 'point']) > 0 else None,
                    'max_iou_tc': results_df[results_df['prompt_type'] == 'point']['iou_tc'].max() if len(results_df[results_df['prompt_type'] == 'point']) > 0 else None,
                    'max_iou_ar': results_df[results_df['prompt_type'] == 'point']['iou_ar'].max() if len(results_df[results_df['prompt_type'] == 'point']) > 0 else None,
                },
                'bbox_prompts': {
                    'avg_iou_tc': results_df[results_df['prompt_type'] == 'bbox']['iou_tc'].mean() if len(results_df[results_df['prompt_type'] == 'bbox']) > 0 else None,
                    'avg_iou_ar': results_df[results_df['prompt_type'] == 'bbox']['iou_ar'].mean() if len(results_df[results_df['prompt_type'] == 'bbox']) > 0 else None,
                    'max_iou_tc': results_df[results_df['prompt_type'] == 'bbox']['iou_tc'].max() if len(results_df[results_df['prompt_type'] == 'bbox']) > 0 else None,
                    'max_iou_ar': results_df[results_df['prompt_type'] == 'bbox']['iou_ar'].max() if len(results_df[results_df['prompt_type'] == 'bbox']) > 0 else None,
                },
                'mask_prompts': {
                    'iou_tc': results_df[results_df['prompt_type'] == 'mask']['iou_tc'].values[0] if len(results_df[results_df['prompt_type'] == 'mask']) > 0 else None,
                    'iou_ar': results_df[results_df['prompt_type'] == 'mask']['iou_ar'].values[0] if len(results_df[results_df['prompt_type'] == 'mask']) > 0 else None,
                }
            }
        }, f, indent=2)
    print(f"✓ Detailed report saved to: {json_path}")
    
    print("\n" + "="*80)
    print("DETAILED RESULTS TABLE")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80)


if __name__ == '__main__':
    print("Starting Prompt Effect Test...")
    args = parse()
    
    if hasattr(args, 'wandb') and args.wandb:
        project_name = args.project_name if hasattr(args, 'project_name') else "climate-sam-prompt-test"
        run_name = args.run_name if hasattr(args, 'run_name') else f"prompt_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(project=project_name, name=run_name, config=vars(args))
    
    if torch.cuda.is_available():
        if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
            used_gpu = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        else:
            used_gpu = get_idle_gpu(gpu_num=1)
            os.environ['CUDA_VISIBLE_DEVICES'] = str(used_gpu[0])
        args.used_gpu, args.gpu_num = used_gpu, len(used_gpu)
    else:
        args.used_gpu, args.gpu_num = [], 1
    
    print(f"Using GPU(s): {args.used_gpu}")
    
    if len(args.used_gpu) == 1:
        main_worker(worker_id=0, worker_args=args)
    else:
        print("Multi-GPU testing not yet implemented. Please use single GPU.")