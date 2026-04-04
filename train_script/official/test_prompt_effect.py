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
import os
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
        # Check if this parameter belongs to a trainable module
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
    print("\n" + "-"*60)
    print("VALIDATING CGNET BASELINE")
    print("-"*60)
    
    # Initialize metrics for baseline (using CGNet aux_mask directly)
    ar_metrics = StreamSegMetrics(class_names=['Background', 'Foreground'])
    tc_metrics = StreamSegMetrics(class_names=['Background', 'Foreground'])
    
    valid_pbar = tqdm(
        total=len(val_dataloader),
        desc='CGNet Baseline Validation',
        leave=False
    )
    
    with torch.no_grad():
        for val_step, batch in enumerate(val_dataloader):
            batch = batch_to_cuda(batch, device)
            
            # Generate auxiliary mask using CGNet (baseline prediction)
            features = batch['cgnet_input'].to(device=device, dtype=torch.float32)
            aux_mask = prompter.get_aux_mask(features)  # B, 3, H, W with channels [BG, TC, AR]
            
            # Extract AR and TC predictions from aux_mask
            # aux_mask channels: [Background, TC, AR]
            ar_pred_mask = aux_mask[:, 2, :, :].unsqueeze(1).unsqueeze(1)  # B, H, W -> B, 1, 1, H, W
            tc_pred_mask = aux_mask[:, 1, :, :].unsqueeze(1).unsqueeze(1)  # B, H, W -> B, 1, 1, H, W
            
            # Ensure proper shape for metric computation: B, 1, H, W
            ar_pred_mask = ar_pred_mask.squeeze(2)  # B, 1, H, W
            tc_pred_mask = tc_pred_mask.squeeze(2)  # B, 1, H, W
            
            # Extract ground truth masks
            masks_gt = batch['gt_mask']  # List of B tensors with shape [H, W]
            masks_ar_gts = [(mask == 2).to(torch.uint8) for mask in masks_gt]  # AR is class 2
            masks_tc_gts = [(mask == 1).to(torch.uint8) for mask in masks_gt]  # TC is class 1
            
            # Convert to list format expected by metrics.update()
            ar_masks = [ar_pred_mask[i:i+1] for i in range(ar_pred_mask.shape[0])]
            tc_masks = [tc_pred_mask[i:i+1] for i in range(tc_pred_mask.shape[0])]
            
            # Ensure correct shape for metrics: B, 1, 1, H, W
            for masks in [masks_ar_gts, masks_tc_gts, ar_masks, tc_masks]:
                for i in range(len(masks)):
                    if len(masks[i].shape) == 2:
                        masks[i] = masks[i][None, None, :]
                    if len(masks[i].shape) == 3:
                        masks[i] = masks[i][:, None, :]
                    if len(masks[i].shape) != 4:
                        raise RuntimeError(f"Unexpected mask shape: {masks[i].shape}")
            
            # Update metrics
            tc_metrics.update(tc_masks, masks_tc_gts, batch['index_name'])
            ar_metrics.update(ar_masks, masks_ar_gts, batch['index_name'])
            
            valid_pbar.update(1)
    
    valid_pbar.close()
    
    # Compute metrics
    ar_metric_dict, _ = ar_metrics.compute()
    tc_metric_dict, _ = tc_metrics.compute()
    
    # Extract metrics
    miou_ar = ar_metric_dict['Mean Foreground IoU']
    miou_tc = tc_metric_dict['Mean Foreground IoU']
    mean_acc_ar = ar_metric_dict['Mean Acc']
    mean_acc_tc = tc_metric_dict['Mean Acc']
    overall_acc_ar = ar_metric_dict['Overall Acc']
    overall_acc_tc = tc_metric_dict['Overall Acc']
    freqw_acc_ar = ar_metric_dict['FreqW Acc']
    freqw_acc_tc = tc_metric_dict['FreqW Acc']
    miou_including_bg_ar = ar_metric_dict['Mean IoU']
    miou_including_bg_tc = tc_metric_dict['Mean IoU']
    
    # Create baseline result dictionary
    baseline_result = {
        'prompt_type': 'cgnet_baseline',
        'positive_point_num': 0,
        'negative_point_num': 0,
        'enlarge_ratio': 0,
        'centroid_ratio': 0,
        'miou_ar': miou_ar,
        'miou_tc': miou_tc,
        'mean_acc_ar': mean_acc_ar,
        'mean_acc_tc': mean_acc_tc,
        'overall_acc_ar': overall_acc_ar,
        'overall_acc_tc': overall_acc_tc,
        'freqw_acc_ar': freqw_acc_ar,
        'freqw_acc_tc': freqw_acc_tc,
        'miou_including_bg_ar': miou_including_bg_ar,
        'miou_including_bg_tc': miou_including_bg_tc,
    }
    
    # Print baseline results
    print(f"\n✓ CGNet Baseline Results:")
    print(f"  mIoU TC: {miou_tc:.4f}, mIoU AR: {miou_ar:.4f}")
    print(f"  Mean Acc TC: {mean_acc_tc:.4f}, Mean Acc AR: {mean_acc_ar:.4f}")
    print(f"  Overall Acc TC: {overall_acc_tc:.4f}, Overall Acc AR: {overall_acc_ar:.4f}")
    
    return baseline_result


def validate_with_combined_prompts(
    val_dataloader, 
    ar_metrics, 
    tc_metrics, 
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
    
    Args:
        val_dataloader: Validation data loader
        ar_metrics: StreamSegMetrics for AR evaluation
        tc_metrics: StreamSegMetrics for TC evaluation
        model: ClimateSAM model
        prompter: CGNetPrompter instance
        device: Device to run on
        positive_point_num: Number of positive points
        negative_point_num: Number of negative points
        enlarge_ratio: BBox enlargement ratio
        centroid_ratio: Centroid ratio for prompt generation
        worker_args: Worker arguments
        prompt_types: List of prompt types to combine (e.g., ['point', 'bbox', 'mask'])
        max_samples: Limit number of validation samples for faster testing
    
    Returns:
        Dictionary with combined metrics
    """
    if prompt_types is None:
        prompt_types = ['point', 'bbox']
    
    model.eval()
    
    # Create progress bar with all prompt types in description
    prompt_types_str = '+'.join(prompt_types)
    valid_pbar = tqdm(
        total=len(val_dataloader),
        desc=f'Combined Validation ({prompt_types_str}, pos={positive_point_num}, neg={negative_point_num}, enlarge={enlarge_ratio})',
        leave=False
    )
    
    total_samples = 0
    
    with torch.no_grad():
        for val_step, batch in enumerate(val_dataloader):
            # Break early if max_samples is reached
            if max_samples and total_samples >= max_samples:
                break
            
            batch = batch_to_cuda(batch, device)
            total_samples += batch['input'].shape[0]
            
            # Generate auxiliary mask using CGNet
            features = batch['cgnet_input'].to(device=device, dtype=torch.float32)
            aux_mask = prompter.get_aux_mask(features)
            
            # Initialize prompt dictionaries for each type
            combined_prompt_dict = {
                'ar_point_prompts': None,
                'tc_point_prompts': None,
                'ar_bbox_prompts': None,
                'tc_bbox_prompts': None,
                'ar_mask_prompts': None,
                'tc_mask_prompts': None,
            }
            
            # Generate prompts for each requested type
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
                
                # Merge prompts from this type into combined dict
                if prompt_type == 'point':
                    combined_prompt_dict['ar_point_prompts'] = prompt_dict.get('ar_point_prompts')
                    combined_prompt_dict['tc_point_prompts'] = prompt_dict.get('tc_point_prompts')
                elif prompt_type == 'bbox':
                    combined_prompt_dict['ar_bbox_prompts'] = prompt_dict.get('ar_bbox_prompts')
                    combined_prompt_dict['tc_bbox_prompts'] = prompt_dict.get('tc_bbox_prompts')
                elif prompt_type == 'mask':
                    combined_prompt_dict['ar_mask_prompts'] = prompt_dict.get('ar_mask_prompts')
                    combined_prompt_dict['tc_mask_prompts'] = prompt_dict.get('tc_mask_prompts')
            
            # Convert combined prompts to device
            for key in combined_prompt_dict:
                if  key in ["ar_bbox_prompts", "tc_bbox_prompts", "ar_mask_prompts", "tc_mask_prompts"]:
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
            
            # Set inference images
            images = model.set_infer_img(batch['input'])
            
            # Perform inference with combined prompts - all at once
            tc_masks, ar_masks = model.infer(
                ar_point_prompts=combined_prompt_dict['ar_point_prompts'],
                tc_point_prompts=combined_prompt_dict['tc_point_prompts'],
                ar_bbox_prompts=combined_prompt_dict['ar_bbox_prompts'],
                tc_bbox_prompts=combined_prompt_dict['tc_bbox_prompts'],
                ar_mask_prompts=combined_prompt_dict.get('ar_mask_prompts'),
                tc_mask_prompts=combined_prompt_dict.get('tc_mask_prompts')
            )
            
            # Extract ground truth masks
            masks_gt = batch['gt_mask']
            masks_ar_gts = [(mask == 2).to(torch.uint8) for mask in masks_gt]
            masks_tc_gts = [(mask == 1).to(torch.uint8) for mask in masks_gt]
            
            # Ensure correct shape for metrics
            for masks in [masks_ar_gts, masks_tc_gts, ar_masks, tc_masks]:
                for i in range(len(masks)):
                    if len(masks[i].shape) == 2:
                        masks[i] = masks[i][None, None, :]
                    if len(masks[i].shape) == 3:
                        masks[i] = masks[i][:, None, :]
                    if len(masks[i].shape) != 4:
                        raise RuntimeError(f"Unexpected mask shape: {masks[i].shape}")
            
            # Update metrics
            tc_metrics.update(tc_masks, masks_tc_gts, batch['index_name'])
            ar_metrics.update(ar_masks, masks_ar_gts, batch['index_name'])
            
            valid_pbar.update(1)
    
    valid_pbar.close()
    
    # Compute metrics
    ar_metric_dict, _ = ar_metrics.compute()
    tc_metric_dict, _ = tc_metrics.compute()
    
    # Create result dictionary
    results = {
        'prompt_type': prompt_types_str,
        'positive_point_num': positive_point_num,
        'negative_point_num': negative_point_num,
        'enlarge_ratio': enlarge_ratio,
        'centroid_ratio': centroid_ratio,
        'miou_ar': ar_metric_dict['Mean Foreground IoU'],
        'miou_tc': tc_metric_dict['Mean Foreground IoU'],
        'mean_acc_ar': ar_metric_dict['Mean Acc'],
        'mean_acc_tc': tc_metric_dict['Mean Acc'],
        'overall_acc_ar': ar_metric_dict['Overall Acc'],
        'overall_acc_tc': tc_metric_dict['Overall Acc'],
        'freqw_acc_ar': ar_metric_dict['FreqW Acc'],
        'freqw_acc_tc': tc_metric_dict['FreqW Acc'],
        'miou_including_bg_ar': ar_metric_dict['Mean IoU'],
        'miou_including_bg_tc': tc_metric_dict['Mean IoU'],
    }
    
    # Reset metrics for next validation
    ar_metrics.reset()
    tc_metrics.reset()
    
    return results


def validate_with_prompt_config(
    val_dataloader, 
    ar_metrics, 
    tc_metrics, 
    model, 
    prompter, 
    device, 
    prompt_type,
    positive_point_num,
    negative_point_num,
    enlarge_ratio,
    centroid_ratio,
    worker_args,
    max_samples=None
):
    """
    Validate model with specific prompt configuration.
    
    Args:
        max_samples: Limit number of validation samples for faster testing
    
    Returns:
        Dictionary with metrics
    """
    model.eval()
    
    # ==================== PROMPT-BASED VALIDATION ====================
    total_samples = 0
    valid_pbar = tqdm(
        total=len(val_dataloader), 
        desc=f'Validation (prompt={prompt_type}, pos={positive_point_num}, neg={negative_point_num}, enlarge={enlarge_ratio})',
        leave=False
    )
    
    prompt_maker = PromptMaker(
        prompt_type=prompt_type, 
        positive_point_num=positive_point_num, 
        negative_point_num=negative_point_num, 
        centroid_ratio=centroid_ratio
    )
    
    with torch.no_grad():
        for val_step, batch in enumerate(val_dataloader):
            # Break early if max_samples is reached
            if max_samples and total_samples >= max_samples:
                break
                
            batch = batch_to_cuda(batch, device)
            total_samples += batch['input'].shape[0]
            
            # Generate prompts using CGNet
            features = batch['cgnet_input'].to(device=device, dtype=torch.float32)
            aux_mask = prompter.get_aux_mask(features)
            
            prompt_dict = prompt_maker.make_prompts(
                multiclass_mask=aux_mask,
                prompt_type=prompt_type,
                positive_point_num=positive_point_num,
                negative_point_num=negative_point_num,
                enlarge_ratio=enlarge_ratio,
                centroid_ratio=centroid_ratio
            )
            
            prompt_dict = batch_to_cuda(prompt_dict, device)
            
            # Set inference images
            images = model.set_infer_img(batch['input'])
            
            # Perform inference with prompts
            tc_masks, ar_masks = model.infer(
                ar_point_prompts=prompt_dict['ar_point_prompts'],
                tc_point_prompts=prompt_dict['tc_point_prompts'],
                ar_bbox_prompts=prompt_dict['ar_bbox_prompts'],
                tc_bbox_prompts=prompt_dict['tc_bbox_prompts']
            )
            
            # Extract ground truth masks
            masks_gt = batch['gt_mask']
            masks_ar_gts = [(mask == 2).to(torch.uint8) for mask in masks_gt]
            masks_tc_gts = [(mask == 1).to(torch.uint8) for mask in masks_gt]
            
            # Ensure correct shape for metrics
            for masks in [masks_ar_gts, masks_tc_gts, ar_masks, tc_masks]:
                for i in range(len(masks)):
                    if len(masks[i].shape) == 2:
                        masks[i] = masks[i][None, None, :]
                    if len(masks[i].shape) == 3:
                        masks[i] = masks[i][:, None, :]
                    if len(masks[i].shape) != 4:
                        raise RuntimeError(f"Unexpected mask shape: {masks[i].shape}")
            
            # Update metrics
            tc_metrics.update(tc_masks, masks_tc_gts, batch['index_name'])
            ar_metrics.update(ar_masks, masks_ar_gts, batch['index_name'])
            
            valid_pbar.update(1)
    
    valid_pbar.close()
    
    # Compute metrics
    ar_metric_dict, _ = ar_metrics.compute()
    tc_metric_dict, _ = tc_metrics.compute()
    
    results = {
        'prompt_type': prompt_type,
        'positive_point_num': positive_point_num,
        'negative_point_num': negative_point_num,
        'enlarge_ratio': enlarge_ratio,
        'centroid_ratio': centroid_ratio,
        'miou_ar': ar_metric_dict['Mean Foreground IoU'],
        'miou_tc': tc_metric_dict['Mean Foreground IoU'],
        'mean_acc_ar': ar_metric_dict['Mean Acc'],
        'mean_acc_tc': tc_metric_dict['Mean Acc'],
        'overall_acc_ar': ar_metric_dict['Overall Acc'],
        'overall_acc_tc': ar_metric_dict['Overall Acc'],
        'freqw_acc_ar': ar_metric_dict['FreqW Acc'],
        'freqw_acc_tc': tc_metric_dict['FreqW Acc'],
        'miou_including_bg_ar': ar_metric_dict['Mean IoU'],
        'miou_including_bg_tc': tc_metric_dict['Mean IoU'],
    }
    
    # Reset metrics for next validation
    ar_metrics.reset()
    tc_metrics.reset()
    
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
        generate_prompt=False
    )
    val_collate_fn = val_dataset.collate_fn
    
    # Debug mode: use smaller dataset
    if hasattr(worker_args, 'debugging') and worker_args.debugging:
        debug_val_size = getattr(worker_args, 'debug_val_size', 20)
        val_indices = list(range(min(debug_val_size, len(val_dataset))))
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
        print(f"Debug mode: Using only {len(val_dataset)} validation samples")
    
    # Create validation dataloader
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
    
    # Load pretrained weights
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
    
    # Load image encoder
    if 'image_encoder' in phase_1_checkpoint:
        climatesam.image_encoder.load_state_dict(phase_1_checkpoint['image_encoder'])
        print(f"✓ Image encoder weights loaded")
    else:
        raise ValueError("Image encoder weights not found in checkpoint.")
    
    # Load mask decoder
    if 'mask_decoder' in phase_1_checkpoint:
        climatesam.mask_decoder.load_state_dict(phase_1_checkpoint['mask_decoder'])
        print(f"✓ Mask decoder weights loaded")
    else:
        raise ValueError("Mask decoder weights not found in checkpoint.")
    
    # Load input adapter
    if 'input_adapter' in phase_1_checkpoint:
        climatesam.input_adapter.load_state_dict(phase_1_checkpoint['input_adapter'])
        print(f"✓ Input adapter weights loaded")
    else:
        raise ValueError("Input adapter weights not found in checkpoint.")
    
    # ==================== FREEZE MODEL PARAMETERS ====================
    print("\n" + "="*60)
    print("FREEZING MODEL PARAMETERS")
    print("="*60)
    
    # Freeze all parameters by default
    trainable_modules = []  # All modules are frozen
    if hasattr(worker_args, 'trainable_modules'):
        trainable_modules = worker_args.trainable_modules
    
    freeze_model_parameters(climatesam, trainable_modules=trainable_modules)
    
    # Verify all parameters are frozen
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
    # baseline_result = validate_cgnet_baseline(val_dataloader, prompter, device, worker_args)
    all_results = []
    
    # ==================== DEFINE PROMPT CONFIGURATIONS ====================
    print("\n" + "="*60)
    print("PROMPT TEST CONFIGURATIONS")
    print("="*60)
    
    # Configuration 1: Point prompts with different (pos, neg) pairs
    point_configs = [
        # {'positive_point_num': 1, 'negative_point_num': 1},
        # {'positive_point_num': 1, 'negative_point_num': 2},
        # {'positive_point_num': 2, 'negative_point_num': 2},
        # {'positive_point_num': 1, 'negative_point_num': 3},
        # {'positive_point_num': 5, 'negative_point_num': 5},
        # {'positive_point_num': 10, 'negative_point_num': 10},
        # {'positive_point_num': 5, 'negative_point_num': 10},
        {'positive_point_num': 15, 'negative_point_num': 5},
        {'positive_point_num': 10, 'negative_point_num': 5},
        {'positive_point_num': 20, 'negative_point_num': 10},
        {'positive_point_num': 20, 'negative_point_num': 5},
        {'positive_point_num': 20, 'negative_point_num': 20},
    ]
    
    # Configuration 2: BBox prompts with different enlarge ratios
    bbox_configs = [
        # {'enlarge_ratio': 0.0},
        # {'enlarge_ratio': 0.1},
        # {'enlarge_ratio': 0.2},
        # {'enlarge_ratio': 0.3},
        # {'enlarge_ratio': 0.5},
        # {'enlarge_ratio': -0.1},
        # {'enlarge_ratio': -0.2},
    ]
    
    # Configuration 3: Mask prompt (single config)
    mask_configs = [{}]
    
    # # ==================== TEST POINT PROMPTS ====================
    # print("\n" + "-"*60)
    # print("TESTING POINT PROMPTS")
    # print("-"*60)
    
    # for config in point_configs:
    #     print(f"\nTesting Point Prompt: pos={config['positive_point_num']}, neg={config['negative_point_num']}")
        
    #     ar_metrics = StreamSegMetrics(class_names=['Background', 'Foreground'])
    #     tc_metrics = StreamSegMetrics(class_names=['Background', 'Foreground'])
        
    #     results = validate_with_prompt_config(
    #         val_dataloader=val_dataloader,
    #         ar_metrics=ar_metrics,
    #         tc_metrics=tc_metrics,
    #         model=climatesam,
    #         prompter=prompter,
    #         device=device,
    #         prompt_type='point',
    #         positive_point_num=config['positive_point_num'],
    #         negative_point_num=config['negative_point_num'],
    #         enlarge_ratio=0,
    #         centroid_ratio=0,
    #         worker_args=worker_args,
    #         max_samples=None
    #     )
        
    #     all_results.append(results)
    #     print(f"  mIoU TC: {results['miou_tc']:.4f}, mIoU AR: {results['miou_ar']:.4f}")
        
    #     if worker_args.wandb:
    #         wandb.log({
    #             'point_prompt/miou_tc': results['miou_tc'],
    #             'point_prompt/miou_ar': results['miou_ar'],
    #             'point_prompt/config': f"pos={config['positive_point_num']}_neg={config['negative_point_num']}"
    #         })
    
    # # ==================== TEST BBOX PROMPTS ====================
    # print("\n" + "-"*60)
    # print("TESTING BBOX PROMPTS")
    # print("-"*60)
    
    # for config in bbox_configs:
    #     print(f"\nTesting BBox Prompt: enlarge_ratio={config['enlarge_ratio']}")
        
    #     ar_metrics = StreamSegMetrics(class_names=['Background', 'Foreground'])
    #     tc_metrics = StreamSegMetrics(class_names=['Background', 'Foreground'])
        
    #     results = validate_with_prompt_config(
    #         val_dataloader=val_dataloader,
    #         ar_metrics=ar_metrics,
    #         tc_metrics=tc_metrics,
    #         model=climatesam,
    #         prompter=prompter,
    #         device=device,
    #         prompt_type='bbox',
    #         positive_point_num=0,
    #         negative_point_num=0,
    #         enlarge_ratio=config['enlarge_ratio'],
    #         centroid_ratio=0,
    #         worker_args=worker_args,
    #         max_samples=None
    #     )
        
    #     all_results.append(results)
    #     print(f"  mIoU TC: {results['miou_tc']:.4f}, mIoU AR: {results['miou_ar']:.4f}")
        
    #     if worker_args.wandb:
    #         wandb.log({
    #             'bbox_prompt/miou_tc': results['miou_tc'],
    #             'bbox_prompt/miou_ar': results['miou_ar'],
    #             'bbox_prompt/enlarge_ratio': config['enlarge_ratio']
    #         })
    
    # # ==================== TEST MASK PROMPTS ====================
    # print("\n" + "-"*60)
    # print("TESTING MASK PROMPTS")
    # print("-"*60)
    
    # for config in mask_configs:
    #     print(f"\nTesting Mask Prompt")
        
    #     ar_metrics = StreamSegMetrics(class_names=['Background', 'Foreground'])
    #     tc_metrics = StreamSegMetrics(class_names=['Background', 'Foreground'])
        
    #     results = validate_with_prompt_config(
    #         val_dataloader=val_dataloader,
    #         ar_metrics=ar_metrics,
    #         tc_metrics=tc_metrics,
    #         model=climatesam,
    #         prompter=prompter,
    #         device=device,
    #         prompt_type='mask',
    #         positive_point_num=0,
    #         negative_point_num=0,
    #         enlarge_ratio=0,
    #         centroid_ratio=0,
    #         worker_args=worker_args,
    #         max_samples=None
    #     )
        
    #     all_results.append(results)
    #     print(f"  mIoU TC: {results['miou_tc']:.4f}, mIoU AR: {results['miou_ar']:.4f}")
        
    #     if worker_args.wandb:
    #         wandb.log({
    #             'mask_prompt/miou_tc': results['miou_tc'],
    #             'mask_prompt/miou_ar': results['miou_ar'],
    #         })
    
    # ==================== TEST COMBINED PROMPTS ====================
    print("\n" + "-"*60)
    print("TESTING COMBINED PROMPTS (Point + BBox + Mask)")
    print("-"*60)
    
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
            'prompt_types': ['point', 'bbox', 'mask'],
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
        
        ar_metrics = StreamSegMetrics(class_names=['Background', 'Foreground'])
        tc_metrics = StreamSegMetrics(class_names=['Background', 'Foreground'])
        
        results = validate_with_combined_prompts(
            val_dataloader=val_dataloader,
            ar_metrics=ar_metrics,
            tc_metrics=tc_metrics,
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
        print(f"  ✓ mIoU TC: {results['miou_tc']:.4f}, mIoU AR: {results['miou_ar']:.4f}")
        print(f"  ✓ Mean Acc TC: {results['mean_acc_tc']:.4f}, Mean Acc AR: {results['mean_acc_ar']:.4f}")
        
        if worker_args.wandb:
            wandb.log({
                'combined_prompts/miou_tc': results['miou_tc'],
                'combined_prompts/miou_ar': results['miou_ar'],
                'combined_prompts/mean_acc_tc': results['mean_acc_tc'],
                'combined_prompts/mean_acc_ar': results['mean_acc_ar'],
                'combined_prompts/config': f"{prompt_types_str}_pos={config['positive_point_num']}_neg={config['negative_point_num']}_enlarge={config['enlarge_ratio']}"
            })
    
    # ==================== SAVE RESULTS ====================
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    # Convert to DataFrame for better visualization
    results_df = pd.DataFrame(all_results)
    
    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(
        worker_args.exp_dir, 
        worker_args.run_name,
        f"prompt_effect_results_{timestamp}.csv"
    )
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to: {results_path}")
    
    # Save detailed JSON report
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
                    'avg_miou_tc': results_df[results_df['prompt_type'] == 'point']['miou_tc'].mean(),
                    'avg_miou_ar': results_df[results_df['prompt_type'] == 'point']['miou_ar'].mean(),
                    'max_miou_tc': results_df[results_df['prompt_type'] == 'point']['miou_tc'].max(),
                    'max_miou_ar': results_df[results_df['prompt_type'] == 'point']['miou_ar'].max(),
                },
                'bbox_prompts': {
                    'avg_miou_tc': results_df[results_df['prompt_type'] == 'bbox']['miou_tc'].mean(),
                    'avg_miou_ar': results_df[results_df['prompt_type'] == 'bbox']['miou_ar'].mean(),
                    'max_miou_tc': results_df[results_df['prompt_type'] == 'bbox']['miou_tc'].max(),
                    'max_miou_ar': results_df[results_df['prompt_type'] == 'bbox']['miou_ar'].max(),
                },
                'mask_prompts': {
                    'miou_tc': results_df[results_df['prompt_type'] == 'mask']['miou_tc'].values[0] if len(results_df[results_df['prompt_type'] == 'mask']) > 0 else None,
                    'miou_ar': results_df[results_df['prompt_type'] == 'mask']['miou_ar'].values[0] if len(results_df[results_df['prompt_type'] == 'mask']) > 0 else None,
                }
            }
        }, f, indent=2)
    print(f"✓ Detailed report saved to: {json_path}")
    
    # Print summary table
    print("\n" + "="*80)
    print("DETAILED RESULTS TABLE")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80)


if __name__ == '__main__':
    print("Starting Prompt Effect Test...")
    args = parse()
    
    # Initialize W&B if enabled
    if hasattr(args, 'wandb') and args.wandb:
        project_name = args.project_name if hasattr(args, 'project_name') else "climate-sam-prompt-test"
        run_name = args.run_name if hasattr(args, 'run_name') else f"prompt_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(project=project_name, name=run_name, config=vars(args))
    
    # Setup GPU
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
    
    # Run main worker
    if len(args.used_gpu) == 1:
        main_worker(worker_id=0, worker_args=args)
    else:
        print("Multi-GPU testing not yet implemented. Please use single GPU.")