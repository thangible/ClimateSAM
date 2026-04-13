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
from utility import batch_to_cuda, get_idle_gpu, get_idle_port, set_randomness, plot_with_projection, plot_mask_with_points_and_bbox, prompt_debug, print_param_stats
from loss_function import ClimateLoss, compute_climate_loss
from tqdm import tqdm
from contextlib import nullcontext
from parser_config import parse
from model.climatesam import ClimateSAM
from dataset.climatenet import ClimateDataset
from evaluator import StreamSegMetrics
import copy
import wandb

# Import the new BBox Prompter
from model.prompt.cgnet_bbox import CGNetBBoxPrompter 


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
    
def setup_optimizer_and_scheduler(model, worker_args):
    """
    Sets up a joint optimizer and scheduler for CAT-SAM and U-Net models.
    """
    lr = worker_args.lr if hasattr(worker_args, 'lr') else 1e-3
    weight_decay = worker_args.weight_decay if hasattr(worker_args, 'weight_decay') else 1e-4

    all_trainable_params = list(p for p in model.parameters() if p.requires_grad) 

    optimizer = torch.optim.AdamW(
        params=all_trainable_params, lr=lr, weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=worker_args.max_epoch_num, eta_min=1e-5
    )
    return optimizer, scheduler

# @torch.no_grad()
# def validate_cgnet_bboxes(
#     val_dataloader, 
#     ar_metrics, 
#     tc_metrics, 
#     model, 
#     prompter, 
#     device, 
#     conf_threshold=0.5,
#     iou_threshold=0.4,
#     max_samples=None,
#     enlarge_ratio=0
# ):
#     """
#     Validate model using direct BBox predictions from CGNetBBoxPrompter.
#     """
#     model.eval()
#     prompter.cgnet_model.eval()
    
#     total_samples = 0
#     valid_pbar = tqdm(
#         total=len(val_dataloader), 
#         desc=f'Validation (BBox conf={conf_threshold}, iou={iou_threshold})',
#         leave=False
#     )
    
#     with torch.no_grad():
#         for val_step, batch in enumerate(val_dataloader):
#             if max_samples and total_samples >= max_samples:
#                 break
                
#             batch = batch_to_cuda(batch, device)
#             total_samples += batch['input'].shape[0]
            
#             # 1. Generate BBox prompts directly using the NEW CGNetBBoxPrompter
#             features = batch['cgnet_input'].to(device=device, dtype=torch.float32)
#             prompt_dict = prompter.get_prompts(features, conf_threshold=conf_threshold, iou_threshold=iou_threshold, enlarge_ratio=enlarge_ratio)
            
#             # 2. Package for SAM
#             combined_prompt_dict = {
#                 'ar_point_prompts': None,
#                 'tc_point_prompts': None,
#                 'ar_bbox_prompts': prompt_dict['ar_bbox_prompts'],
#                 'tc_bbox_prompts': prompt_dict['tc_bbox_prompts']
#             }
            
#             # 3. Set inference images
#             images = model.set_infer_img(batch['input'])
            
#             # 4. Perform inference with combined prompts - all at once
#             tc_masks, ar_masks = model.infer(
#                 ar_point_prompts=combined_prompt_dict['ar_point_prompts'],
#                 tc_point_prompts=combined_prompt_dict['tc_point_prompts'],
#                 ar_bbox_prompts=combined_prompt_dict['ar_bbox_prompts'],
#                 tc_bbox_prompts=combined_prompt_dict['tc_bbox_prompts']
#             )
            
#             # 5. Extract ground truth masks
#             masks_gt = batch['gt_mask']
#             masks_ar_gts = [(mask == 2).to(torch.uint8) for mask in masks_gt]
#             masks_tc_gts = [(mask == 1).to(torch.uint8) for mask in masks_gt]
            
#             # Ensure correct shape for metrics [B, 1, 1, H, W]
#             for masks in [masks_ar_gts, masks_tc_gts, ar_masks, tc_masks]:
#                 for i in range(len(masks)):
#                     if len(masks[i].shape) == 2:
#                         masks[i] = masks[i][None, None, :]
#                     if len(masks[i].shape) == 3:
#                         masks[i] = masks[i][:, None, :]
#                     if len(masks[i].shape) != 4:
#                         raise RuntimeError(f"Unexpected mask shape: {masks[i].shape}")
            
#             # 6. Update metrics
#             tc_metrics.update(tc_masks, masks_tc_gts, batch['index_name'])
#             ar_metrics.update(ar_masks, masks_ar_gts, batch['index_name'])
            
#             valid_pbar.update(1)
    
#     valid_pbar.close()
    
#     # Compute metrics
#     ar_metric_dict, _ = ar_metrics.compute()
#     tc_metric_dict, _ = tc_metrics.compute()
    
#     results = {
#         'prompt_type': 'bbox_direct',
#         'conf_threshold': conf_threshold,
#         'iou_threshold': iou_threshold,
#         'miou_ar': ar_metric_dict['Mean Foreground IoU'],
#         'miou_tc': tc_metric_dict['Mean Foreground IoU'],
#         'mean_acc_ar': ar_metric_dict['Mean Acc'],
#         'mean_acc_tc': tc_metric_dict['Mean Acc'],
#         'overall_acc_ar': ar_metric_dict['Overall Acc'],
#         'overall_acc_tc': tc_metric_dict['Overall Acc'],
#         'freqw_acc_ar': ar_metric_dict['FreqW Acc'],
#         'freqw_acc_tc': tc_metric_dict['FreqW Acc'],
#         'miou_including_bg_ar': ar_metric_dict['Mean IoU'],
#         'miou_including_bg_tc': tc_metric_dict['Mean IoU'],
#     }
    
#     # Reset metrics for next validation
#     ar_metrics.reset()
#     tc_metrics.reset()
    
#     return results


@torch.no_grad()
def validate_cgnet_bboxes(
    val_dataloader, 
    ar_metrics, 
    tc_metrics, 
    model, 
    prompter, 
    device, 
    conf_threshold=0.5,
    iou_threshold=0.4,
    max_samples=None,
    enlarge_ratio=0,
    merge_threshold=10
):
    """
    Validate model using direct BBox predictions from CGNetBBoxPrompter
    with added diagnostic profiling.
    """
    model.eval()
    prompter.cgnet_model.eval()
    
    total_samples = 0
    valid_pbar = tqdm(
        total=len(val_dataloader), 
        desc=f'Validation (BBox conf={conf_threshold}, iou={iou_threshold})',
        leave=False
    )
    
    # Initialize diagnostic accumulators
    ar_diag_totals = {'gt': 0, 'pred': 0, 'missed': 0, 'ghosts': 0, 'fragmented': 0, 'ious': []}
    tc_diag_totals = {'gt': 0, 'pred': 0, 'missed': 0, 'ghosts': 0, 'fragmented': 0, 'ious': []}

    with torch.no_grad():
        for val_step, batch in enumerate(val_dataloader):
            if max_samples and total_samples >= max_samples:
                break
                
            batch = batch_to_cuda(batch, device)
            total_samples += batch['input'].shape[0]
            
            # 1. Generate BBox prompts directly
            features = batch['cgnet_input'].to(device=device, dtype=torch.float32)
            prompt_dict = prompter.get_prompts(features, conf_threshold=conf_threshold, iou_threshold=iou_threshold, enlarge_ratio=enlarge_ratio, merge_threshold= merge_threshold)
            
            # ================================================================= #
            # NEW: GHOST CATCHER & VISUALIZER
            # ================================================================= #
            batch_size = features.shape[0]
            ghost_save_dir = os.path.join('exp', "ghost_visualizations")
            os.makedirs(ghost_save_dir, exist_ok=True)
            
            for b in range(batch_size):
                gt_ar = batch['ar_bbox_prompts'][b]
                pr_ar = prompt_dict['ar_bbox_prompts'][b]
                
                ghost_count = 0
                
                # Check if we predicted boxes
                if pr_ar is not None and len(pr_ar) > 0:
                    # If there is no Ground Truth at all, EVERY prediction is a ghost
                    if gt_ar is None or len(gt_ar) == 0:
                        ghost_count = len(pr_ar)
                    # Otherwise, calculate IoU to find boxes hitting pure background
                    else:
                        gt_flat = gt_ar.view(-1, 4).float().to(device)
                        pr_flat = pr_ar.view(-1, 4).float().to(device)
                        
                        iou_matrix = ops.box_iou(gt_flat, pr_flat)
                        max_iou_per_pred, _ = iou_matrix.max(dim=0)
                        
                        # Count how many predicted boxes have < 0.3 overlap with any GT box
                        ghost_count = (max_iou_per_pred < 0.3).sum().item()
                
                # Plot and save if this image has an aggressively high ghost count
                # (You can change this threshold to >= 2 or 3 to only catch the worst offenders)
                if ghost_count >= 1: 
                    save_path = os.path.join(ghost_save_dir, f"ghost_ar_ratio_{enlarge_ratio}_step_{val_step}_img_{b}.png")
                    
                    # Plot using your existing utility
                    plot_mask_with_points_and_bbox(
                        mask=batch['gt_mask'][b].cpu().numpy() if isinstance(batch['gt_mask'][b], torch.Tensor) else batch['gt_mask'][b],
                        ar_bbox=pr_ar.cpu().numpy() if pr_ar is not None else None,
                        tc_bbox=prompt_dict['tc_bbox_prompts'][b].cpu().numpy() if prompt_dict['tc_bbox_prompts'][b] is not None else None,
                        save_path=save_path,
                        title=f"AR Ghosts Found: {ghost_count} (Ratio: {enlarge_ratio})"
                    )
                    
                    # Optional: Log the worst offenders directly to WandB
                    if wandb.run is not None and ghost_count >= 2:
                        wandb.log({f"ghost_analysis/ratio_{enlarge_ratio}": wandb.Image(save_path)})
            # ================================================================= #
            
            # --- RUN DIAGNOSTICS ---
            # Compare dataset GT boxes with predicted boxes
            ar_stats = profile_prompt_errors(batch['ar_bbox_prompts'], prompt_dict['ar_bbox_prompts'], iou_threshold=0.3)
            tc_stats = profile_prompt_errors(batch['tc_bbox_prompts'], prompt_dict['tc_bbox_prompts'], iou_threshold=0.3)

            # Accumulate AR stats
            ar_diag_totals['gt'] += ar_stats['total_gt_objects']
            ar_diag_totals['pred'] += ar_stats['total_pred_prompts']
            ar_diag_totals['missed'] += ar_stats['missed_objects']
            ar_diag_totals['ghosts'] += ar_stats['ghost_prompts']
            ar_diag_totals['fragmented'] += ar_stats['fragmented_objects']
            ar_diag_totals['ious'].extend(ar_stats['matched_bbox_iou'])

            # Accumulate TC stats
            tc_diag_totals['gt'] += tc_stats['total_gt_objects']
            tc_diag_totals['pred'] += tc_stats['total_pred_prompts']
            tc_diag_totals['missed'] += tc_stats['missed_objects']
            tc_diag_totals['ghosts'] += tc_stats['ghost_prompts']
            tc_diag_totals['fragmented'] += tc_stats['fragmented_objects']
            tc_diag_totals['ious'].extend(tc_stats['matched_bbox_iou'])
            # -----------------------

            # 2. Package for SAM
            combined_prompt_dict = {
                'ar_point_prompts': None,
                'tc_point_prompts': None,
                'ar_bbox_prompts': prompt_dict['ar_bbox_prompts'],
                'tc_bbox_prompts': prompt_dict['tc_bbox_prompts']
            }
            
            # 3. Set inference images
            images = model.set_infer_img(batch['input'])
            
            # 4. Perform inference with combined prompts
            tc_masks, ar_masks = model.infer(
                ar_point_prompts=combined_prompt_dict['ar_point_prompts'],
                tc_point_prompts=combined_prompt_dict['tc_point_prompts'],
                ar_bbox_prompts=combined_prompt_dict['ar_bbox_prompts'],
                tc_bbox_prompts=combined_prompt_dict['tc_bbox_prompts']
            )
            
            # 5. Extract ground truth masks
            masks_gt = batch['gt_mask']
            masks_ar_gts = [(mask == 2).to(torch.uint8) for mask in masks_gt]
            masks_tc_gts = [(mask == 1).to(torch.uint8) for mask in masks_gt]
            
            for masks in [masks_ar_gts, masks_tc_gts, ar_masks, tc_masks]:
                for i in range(len(masks)):
                    if len(masks[i].shape) == 2:
                        masks[i] = masks[i][None, None, :]
                    if len(masks[i].shape) == 3:
                        masks[i] = masks[i][:, None, :]
                    if len(masks[i].shape) != 4:
                        raise RuntimeError(f"Unexpected mask shape: {masks[i].shape}")
            
            # 6. Update metrics
            tc_metrics.update(tc_masks, masks_tc_gts, batch['index_name'])
            ar_metrics.update(ar_masks, masks_ar_gts, batch['index_name'])
            
            valid_pbar.update(1)
    
    valid_pbar.close()
    
    # Compute metrics
    ar_metric_dict, _ = ar_metrics.compute()
    tc_metric_dict, _ = tc_metrics.compute()

    # Compute Diagnostic Averages
    avg_ar_bbox_iou = np.mean(ar_diag_totals['ious']) if ar_diag_totals['ious'] else 0.0
    avg_tc_bbox_iou = np.mean(tc_diag_totals['ious']) if tc_diag_totals['ious'] else 0.0
    
    results = {
        'prompt_type': 'bbox_direct',
        'enlarge_ratio': enlarge_ratio,
        'miou_ar': ar_metric_dict['Mean Foreground IoU'],
        'miou_tc': tc_metric_dict['Mean Foreground IoU'],
        # Add Diagnostics to results
        'diag_ar_avg_bbox_iou': avg_ar_bbox_iou,
        'diag_ar_missed_pct': ar_diag_totals['missed'] / max(1, ar_diag_totals['gt']),
        'diag_ar_ghost_pct': ar_diag_totals['ghosts'] / max(1, ar_diag_totals['pred']),
        'diag_tc_avg_bbox_iou': avg_tc_bbox_iou,
        'diag_tc_missed_pct': tc_diag_totals['missed'] / max(1, tc_diag_totals['gt']),
        'diag_tc_ghost_pct': tc_diag_totals['ghosts'] / max(1, tc_diag_totals['pred']),
    }

    # Log to WandB
    if wandb.run is not None:
        wandb.log({
            f"eval/ar_seg_iou_ratio_{enlarge_ratio}": results['miou_ar'],
            f"eval/tc_seg_iou_ratio_{enlarge_ratio}": results['miou_tc'],
            f"diag/ar_bbox_iou_ratio_{enlarge_ratio}": results['diag_ar_avg_bbox_iou'],
            f"diag/ar_missed_pct_ratio_{enlarge_ratio}": results['diag_ar_missed_pct'],
            f"diag/ar_ghost_pct_ratio_{enlarge_ratio}": results['diag_ar_ghost_pct'],
            f"diag/tc_bbox_iou_ratio_{enlarge_ratio}": results['diag_tc_avg_bbox_iou'],
            f"diag/tc_missed_pct_ratio_{enlarge_ratio}": results['diag_tc_missed_pct'],
            f"diag/tc_ghost_pct_ratio_{enlarge_ratio}": results['diag_tc_ghost_pct'],
        })
    
    ar_metrics.reset()
    tc_metrics.reset()
    
    return results


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
        
def main_worker(worker_id, worker_args):
    set_randomness()
    max_epoch_num = worker_args.max_epoch_num 
    if isinstance(worker_id, str):
        worker_id = int(worker_id)
    device, local_rank = setup_device_and_distributed(worker_id, worker_args)
    print(f"Worker {worker_id} initialized on device {device} with local_rank {local_rank}.")
    
    # PREPARE DATASET
    # Ensure generate_prompt is True if it's required to yield bounding boxes inside your dataset
    dataset_dir = worker_args.data_dir
    train_dataset = ClimateDataset(
        data_dir=dataset_dir, train_flag=True, shot_num=worker_args.shot_num,
        augmented=worker_args.augmented, generate_prompt=True, prompt_type='bbox', enlarge_ratio=[0,0]
    )
    val_dataset = ClimateDataset(data_dir=dataset_dir, train_flag=False, augmented=False, generate_prompt=True, prompt_type='bbox', enlarge_ratio=[0,0])
    
    train_collate_fn = train_dataset.collate_fn
    val_collate_fn = val_dataset.collate_fn

    if hasattr(worker_args, 'debugging') and worker_args.debugging:
        debug_size = getattr(worker_args, 'debug_size', 10)  
        indices = list(range(min(debug_size, len(train_dataset))))
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        print(f"Debug mode: Using only {len(train_dataset)} training samples")

        debug_val_size = getattr(worker_args, 'debug_val_size', 5)  
        val_indices = list(range(min(debug_val_size, len(val_dataset))))
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
        print(f"Debug mode: Using only {len(val_dataset)} validation samples")
        
        max_epoch_num = 2
        worker_args.valid_per_epochs = 1
        print(f"Debug mode: Setting max_epoch_num to {max_epoch_num} and valid_per_epochs to {worker_args.valid_per_epochs}")
        
    
    # DataLoader
    train_bs = worker_args.train_bs if worker_args.train_bs else (1 if worker_args.shot_num == 1 else 4)
    gradient_accumulation_steps = getattr(worker_args, 'gradient_accumulation_steps', 1)
    
    actual_train_bs = train_bs // gradient_accumulation_steps
    if actual_train_bs < 1:
        actual_train_bs = 1
        print(f"Warning: gradient_accumulation_steps ({gradient_accumulation_steps}) is larger than train_bs ({train_bs}). Setting actual batch size to 1.")
    
    effective_batch_size = actual_train_bs * gradient_accumulation_steps
    if torch.distributed.is_initialized():
        effective_batch_size *= torch.distributed.get_world_size()
    
    print(f"Effective batch size: {effective_batch_size} (actual_bs: {actual_train_bs}, accumulation: {gradient_accumulation_steps})")
    
    val_bs = worker_args.val_bs if worker_args.val_bs else 2
    train_workers, val_workers = 1 if worker_args.shot_num == 1 else 4, 2
    if worker_args.num_workers is not None:
        train_workers, val_workers = worker_args.num_workers, worker_args.num_workers
        
    sampler = None
        
    g = torch.Generator()
    g.manual_seed(3407)
        
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=actual_train_bs, shuffle=sampler is None, num_workers=train_workers,
        sampler=sampler, drop_last=False, collate_fn=train_collate_fn,
        worker_init_fn=partial(worker_init_fn, base_seed=3407), generator=g
    )
    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=val_bs, shuffle=False, num_workers=val_workers,
        drop_last=False, collate_fn=val_collate_fn, worker_init_fn=partial(worker_init_fn, base_seed=3407), generator=g
    )

    pretrained_name = worker_args.pretrained_name if hasattr(worker_args, 'pretrained_name') else os.path.join(worker_args.exp_dir,'cgnet_bbox_weight.pth')
    # Initialize the new BBox Prompter
    cgnetprompter = CGNetBBoxPrompter(
        weights_path=pretrained_name, # You can update this to the new checkpoint name
        device=device, 
        worker_args=worker_args,
        num_classes=2 # 0: TC, 1: AR
    )
    
    # Execute the training loop
    print("Starting CGNet Bounding Box Training...")
    print_param_stats(cgnetprompter.cgnet_model, phase="train")
    cgnetprompter.train(train_dataloader=train_dataloader, val_dataloader=val_dataloader, epochs=max_epoch_num)

    # ---------------------------------------------------------
    # Validation / SAM Inference Phase
    # ---------------------------------------------------------
    print("Initializing SAM for final validation...")
    
    climatesam = ClimateSAM(
        model_type=worker_args.sam_type, 
        mlp_ratio=worker_args.image_encoder_mlp_ratio,
        enable_wandb_logging=getattr(worker_args, 'debugging', False)
    ).to(device=device)
    
    # Load pretrained SAM weights if available
    weights_name = getattr(worker_args, 'encoder_weights_name', 'phase_1_weights')
    image_encoder_path = os.path.join(worker_args.exp_dir, f"{weights_name}.pth")
    
    if os.path.exists(image_encoder_path):
        print(f"Loading SAM weights from {image_encoder_path}")
        checkpoint = torch.load(image_encoder_path, map_location=device)
        if 'image_encoder' in checkpoint: 
            climatesam.image_encoder.load_state_dict(checkpoint['image_encoder'])
        if 'mask_decoder' in checkpoint: 
            climatesam.mask_decoder.load_state_dict(checkpoint['mask_decoder'])
        if 'input_adapter' in checkpoint: 
            climatesam.input_adapter.load_state_dict(checkpoint['input_adapter'])
    else:
        print(f"Warning: SAM weights not found at {image_encoder_path}. Using default initialization.")

    # Initialize segmentation metrics
    ar_metrics = StreamSegMetrics(class_names=['Background', 'Foreground'])
    tc_metrics = StreamSegMetrics(class_names=['Background', 'Foreground'])
    
    print("Running final validation with integrated BBox Prompter and SAM...")
    import itertools

    # Define the search grid
    conf_thresholds = [0.3, 0.5, 0.7]
    iou_thresholds = [0.3, 0.5, 0.7]
    merge_thresholds = [10.0, 20.0, 30.0]
    fixed_enlarge_ratio = 0.1  # Fix this to avoid a massive 4D grid search

    # Generate all combinations
    hyperparameter_grid = list(itertools.product(conf_thresholds, iou_thresholds, merge_thresholds))
    
    all_results = []
    
    for conf_thresh, iou_thresh, merge_thresh in hyperparameter_grid:
        print(f"\nValidating -> Conf: {conf_thresh} | IoU: {iou_thresh} | Merge: {merge_thresh}...")
        
        results = validate_cgnet_bboxes(
            val_dataloader=val_dataloader,
            ar_metrics=ar_metrics,
            tc_metrics=tc_metrics,
            model=climatesam,
            prompter=cgnetprompter,
            device=device,
            conf_threshold=conf_thresh,
            iou_threshold=iou_thresh,
            max_samples=None,
            enlarge_ratio=fixed_enlarge_ratio,
            merge_threshold=merge_thresh
        )
        
        # Inject the thresholds into the results dict for the final pandas DataFrame
        results['conf_threshold'] = conf_thresh
        results['iou_threshold'] = iou_thresh
        results['merge_threshold'] = merge_thresh
        
        all_results.append(results)
        print(f"Complete -> TC mIoU: {results['miou_tc']:.4f} | AR mIoU: {results['miou_ar']:.4f} | AR Ghosts: {results['diag_ar_ghost_pct']:.2%}")

    # ==================== SAVE RESULTS ====================
    print("\n" + "="*80)
    print("RESULTS SUMMARY (SORTED BY AR mIoU)")
    print("="*80)
    
    # Convert to DataFrame, sort to find the best configuration easily
    import pandas as pd
    results_df = pd.DataFrame(all_results)
    
    # Reorder columns to put hyperparams first for readability
    cols = ['conf_threshold', 'iou_threshold', 'merge_threshold', 'enlarge_ratio', 'miou_ar', 'miou_tc', 'diag_ar_avg_bbox_iou', 'diag_tc_avg_bbox_iou']
    # Add any remaining columns
    cols.extend([c for c in results_df.columns if c not in cols])
    results_df = results_df[cols]
    
    # Sort by AR mIoU descending to put the best results at the top
    results_df = results_df.sort_values(by='miou_ar', ascending=False)
    
    print("="*120)
    print(results_df.to_string(index=False))
    print("="*120)

    # print("Running final validation with integrated BBox Prompter and SAM...")
    # bbox_configs = [
    #     {'enlarge_ratio': 0.0},
    #     {'enlarge_ratio': 0.1},
    #     {'enlarge_ratio': 0.2},
    #     {'enlarge_ratio': 0.3},
    #     {'enlarge_ratio': 0.4},
    #     {'enlarge_ratio': 0.5},
    #     {'enlarge_ratio': 0.6},
    #     {'enlarge_ratio': 0.7},
    #     {'enlarge_ratio': -0.1},
    #     {'enlarge_ratio': -0.2},
    # ]
    # all_results = []
    # for bbox_config in bbox_configs:
    #     enlarge_ratio = bbox_config['enlarge_ratio']
    #     print(f"\nValidating with enlarge_ratio={enlarge_ratio}...")
    #     # Validate quickly and plot 10 distinct samples
    #     ar_iou, tc_iou = cgnetprompter.quick_evaluate(val_dataloader, n_samples=10, save_dir="my_test_plots", name = f"enlarge_{enlarge_ratio}")
    #     results = validate_cgnet_bboxes(
    #         val_dataloader=val_dataloader,
    #         ar_metrics=ar_metrics,
    #         tc_metrics=tc_metrics,
    #         model=climatesam,
    #         prompter=cgnetprompter,
    #         device=device,
    #         conf_threshold=0.5,
    #         iou_threshold=0.4,
    #         max_samples=None,
    #         enlarge_ratio=enlarge_ratio,
    #         merge_threshold=10
    # )
    #     all_results.append(results)
    #     print(f"Validation completed for enlarge_ratio={enlarge_ratio}. mIoU TC: {results['miou_tc']:.4f}, mIoU AR: {results['miou_ar']:.4f}, mean AR bbox IoU: {results['diag_ar_avg_bbox_iou']:.4f}, mean TC bbox IoU: {results['diag_tc_avg_bbox_iou']:.4f}")

    # # ==================== SAVE RESULTS ====================
    # print("\n" + "="*60)
    # print("RESULTS SUMMARY")
    # print("="*60)
    
    # # Convert to DataFrame for better visualization
    # import pandas as pd
    # results_df = pd.DataFrame(all_results)
    # print("="*80)
    # print(results_df.to_string(index=False))
    # print("="*80)
    
    
if __name__ == '__main__':
    print("Starting training process...")
    args = parse()
    
    if hasattr(args, 'wandb') and args.wandb:
        project_name = args.project_name if hasattr(args, 'project_name') else "climate-sam-cgnet"
        run_name = args.run_name if hasattr(args, 'run_name') else None
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

    if len(args.used_gpu) == 1:
        main_worker(worker_id=0, worker_args=args)
        
