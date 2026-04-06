"""
Script to evaluate the Propose-and-Score (Automatic Mask Generation) Pipeline.
SAM generates dense proposals, and a CGNetPatchClassifier scores and filters them.
"""

import sys
import os
import json
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
import torchvision.ops as ops
import torchvision.transforms.functional as TF
from tqdm import tqdm
from functools import partial
from torch.utils.data import DataLoader
import wandb

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(TRAIN_SCRIPT_DIR)
for p in (PROJECT_ROOT, TRAIN_SCRIPT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from utility import batch_to_cuda, get_idle_gpu, set_randomness, setup_device_and_distributed, worker_init_fn
from parser_config import parse
from model.climatesam import ClimateSAM
from dataset.climatenet import ClimateDataset
from evaluator import StreamSegMetrics
from model.prompt.cgnet_module import CGNetModule

import matplotlib.pyplot as plt
import wandb
import numpy as np

def log_pipeline_visualizations(gt_mask, tc_pred, ar_pred, tc_points, ar_points, step, image_idx):
    """
    Creates a 1x3 plot showing:
    1. GT Mask + TC Grid Prompts
    2. GT Mask + AR Grid Prompts
    3. Final Accepted Predicted Masks
    and logs it to Weights & Biases.
    """
    # Convert tensors to numpy arrays for plotting
    gt = gt_mask.cpu().numpy()
    tc_m = tc_pred.squeeze().cpu().numpy()
    ar_m = ar_pred.squeeze().cpu().numpy()
    
    tc_pts = tc_points.cpu().numpy()
    ar_pts = ar_points.cpu().numpy()
    
    # Create a combined prediction mask for visualization (1=TC, 2=AR)
    pred_combined = np.zeros_like(gt)
    pred_combined[tc_m > 0] = 1
    pred_combined[ar_m > 0] = 2

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # --- Plot 1: GT + TC Grid ---
    axes[0].imshow(gt, cmap='viridis', interpolation='nearest')
    axes[0].scatter(tc_pts[:, 0], tc_pts[:, 1], c='red', s=2, alpha=0.5, label='TC Grid')
    axes[0].set_title('GT Mask + TC Grid Prompts')
    axes[0].axis('off')
    
    # --- Plot 2: GT + AR Grid ---
    axes[1].imshow(gt, cmap='viridis', interpolation='nearest')
    axes[1].scatter(ar_pts[:, 0], ar_pts[:, 1], c='cyan', s=2, alpha=0.5, label='AR Grid')
    axes[1].set_title('GT Mask + AR Grid Prompts')
    axes[1].axis('off')
    
    # --- Plot 3: Final Accepted Segment ---
    # We plot the GT dimly in the background, and the Prediction brightly on top
    axes[2].imshow(gt, cmap='gray', alpha=0.3) 
    axes[2].imshow(pred_combined, cmap='viridis', alpha=0.8, interpolation='nearest')
    axes[2].set_title('Final Accepted Segments (1=TC, 2=AR)')
    axes[2].axis('off')

    plt.tight_layout()
    
    # Log to WandB
    if wandb.run is not None:
        wandb.log({f"visualizations/step_{step}_img_{image_idx}": wandb.Image(fig)}, commit=False)
    
    plt.close(fig)

# ========================================== #
# 1. CGNet Patch Classifier                  #
# ========================================== #
class CGNetPatchClassifier(nn.Module):
    def __init__(self, pretrained_weights_path, device, num_classes=3, in_channels=4):
        super().__init__()
        self.backbone = CGNetModule(classes=num_classes, channels=in_channels)
        
        if pretrained_weights_path and os.path.exists(pretrained_weights_path):
            pretrained_dict = torch.load(pretrained_weights_path, map_location=device)
            model_dict = self.backbone.state_dict()
            filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
            model_dict.update(filtered_dict)
            self.backbone.load_state_dict(model_dict)
            print(f"✓ Loaded {len(filtered_dict)}/{len(model_dict)} matching layers into CGNet Classifier Backbone.")
        else:
            print(f"Warning: Pretrained weights not found at {pretrained_weights_path}! Using random init.")

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        spatial_logits = self.backbone(x)
        pooled_logits = self.global_pool(spatial_logits)
        logits = torch.flatten(pooled_logits, 1)
        return logits


# ========================================== #
# 2. Pipeline Helpers                        #
# ========================================== #
def generate_banded_point_grid(image_w, y_bands, grid_x_steps=32, y_steps_per_band=12, device='cuda'):
    all_points = []
    x_steps = torch.linspace(0, image_w - 1, grid_x_steps, device=device)
    for (y_min, y_max) in y_bands:
        y_steps = torch.linspace(y_min, y_max, y_steps_per_band, device=device)
        grid_y, grid_x = torch.meshgrid(y_steps, x_steps, indexing='ij')
        points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        all_points.append(points)
    return torch.cat(all_points, dim=0)

def extract_and_score_masks(cgnet_image, masks, classifier_model, crop_size=(128, 128)):
    model_device = next(classifier_model.parameters()).device
    if cgnet_image.device != model_device:
        cgnet_image = cgnet_image.to(model_device)
    masks = masks.to(device=model_device)
    N, H, W = masks.shape
    C = cgnet_image.shape[0]
    
    valid_masks = []
    mask_crops = []
    
    for i in range(N):
        mask = masks[i]
        if mask.sum() < 10: 
            continue
            
        y_indices, x_indices = torch.where(mask > 0)
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()
        
        # Multiply by mask and crop
        masked_image = cgnet_image * mask.unsqueeze(0)
        crop = masked_image[:, y_min:y_max+1, x_min:x_max+1]
        crop_resized = TF.resize(crop, crop_size, antialias=True)
        
        mask_crops.append(crop_resized)
        valid_masks.append(mask)

    if len(mask_crops) == 0:
        return torch.empty(0, device=model_device), torch.empty(0, device=model_device), torch.empty(0, H, W, device=model_device)

    batch_crops = torch.stack(mask_crops).to(model_device)
    
    with torch.no_grad():
        logits = classifier_model(batch_crops)
        probs = torch.softmax(logits, dim=1)
        max_probs, preds = torch.max(probs, dim=1)
        
    # Ignore background class (0)
    fg_mask = preds > 0
    final_masks = torch.stack(valid_masks)[fg_mask]
    final_scores = max_probs[fg_mask]
    final_classes = preds[fg_mask]
    
    return final_scores, final_classes, final_masks

# ========================================== #
# 3. The Propose & Score Pipeline            #
# ========================================== #
class ProposeAndScorePipeline:
    def __init__(self, sam_model, classifier_model, device='cuda'):
        self.sam = sam_model
        self.classifier = classifier_model
        self.device = device
        
    @torch.no_grad()
    def generate_objects(self, sam_image, cgnet_image, points_per_batch=64, iou_thresh=0.5, conf_thresh=0.6):
        C, H, W = sam_image.shape
        
        # Define strict geographic bounds
        tc_y_bands = [(105, 341), (427, 678)]
        ar_y_bands = [(54, 341), (427, 739)]
        
        # Generate target grids
        tc_points = generate_banded_point_grid(W, tc_y_bands, grid_x_steps=32, y_steps_per_band=10, device=self.device)
        ar_points = generate_banded_point_grid(W, ar_y_bands, grid_x_steps=32, y_steps_per_band=12, device=self.device)
        
        # Cache image embedding in SAM
        self.sam.set_infer_img(sam_image.unsqueeze(0))
        
        all_masks, all_scores, all_classes, all_boxes = [], [], [], []
        
        # --- PHASE 1: Process TC Grid ---
        all_masks, all_scores, all_classes, all_boxes = self._process_grid(
            points=tc_points, cgnet_image=cgnet_image, points_per_batch=points_per_batch, 
            prompt_type='TC', conf_thresh=conf_thresh, target_class_id=1, 
            accumulators=(all_masks, all_scores, all_classes, all_boxes)
        )
        
        # --- PHASE 2: Process AR Grid ---
        all_masks, all_scores, all_classes, all_boxes = self._process_grid(
            points=ar_points, cgnet_image=cgnet_image, points_per_batch=points_per_batch, 
            prompt_type='AR', conf_thresh=conf_thresh, target_class_id=2, 
            accumulators=(all_masks, all_scores, all_classes, all_boxes)
        )
        
        if not all_masks:
            return None, None, None
            
        # Global NMS
        global_masks = torch.cat(all_masks, dim=0)
        global_scores = torch.cat(all_scores, dim=0)
        global_classes = torch.cat(all_classes, dim=0)
        global_boxes = torch.cat(all_boxes, dim=0)
        
        keep_indices = ops.batched_nms(global_boxes, global_scores, global_classes, iou_thresh)
        
        return global_masks[keep_indices], global_classes[keep_indices], global_scores[keep_indices]

    def _process_grid(self, points, cgnet_image, points_per_batch, prompt_type, conf_thresh, target_class_id, accumulators):
        all_masks, all_scores, all_classes, all_boxes = accumulators
        num_points = points.shape[0]
        
        for i in range(0, num_points, points_per_batch):
            batch_points = points[i:i+points_per_batch]
            formatted_points = [(batch_points.unsqueeze(0), torch.ones(len(batch_points), device=self.device).unsqueeze(0))]
            
            # Request masks ONLY from the targeted SAM head
            if prompt_type == 'TC':
                tc_masks, _ = self.sam.infer(tc_point_prompts=formatted_points, ar_point_prompts=None)
                raw_masks = tc_masks[0].squeeze(1)
            else:
                _, ar_masks = self.sam.infer(tc_point_prompts=None, ar_point_prompts=formatted_points)
                raw_masks = ar_masks[0].squeeze(1)
                
            # Score crops using the CGNet Patch Classifier
            scores, classes, valid_masks = extract_and_score_masks(cgnet_image, raw_masks, self.classifier)
            
            # Enforce Confidence AND Class Consistency
            valid_idx = (scores > conf_thresh) & (classes == target_class_id)
            scores, classes, valid_masks = scores[valid_idx], classes[valid_idx], valid_masks[valid_idx]
            
            if len(scores) == 0:
                continue
                
            boxes = ops.masks_to_boxes(valid_masks)
            
            all_masks.append(valid_masks)
            all_scores.append(scores)
            all_classes.append(classes)
            all_boxes.append(boxes)
            
        return all_masks, all_scores, all_classes, all_boxes


# ========================================== #
# 4. Evaluation Loop                         #
# ========================================== #
@torch.no_grad()
def validate_propose_and_score(val_dataloader, ar_metrics, tc_metrics, pipeline, device, worker_args, max_samples=None):
    pipeline.sam.eval()
    pipeline.classifier.eval()
    
    total_samples = 0
    valid_pbar = tqdm(total=len(val_dataloader), desc='Propose & Score Eval', leave=False)
    
    # ---> ADDED: Generate the visualization grids here so they are defined <---
    W = val_dataloader.dataset[0]['input'].shape[2] 
    tc_y_bands = [(105, 341), (427, 678)]
    ar_y_bands = [(54, 341), (427, 739)]
    tc_points_vis = generate_banded_point_grid(W, tc_y_bands, grid_x_steps=32, y_steps_per_band=10, device=device)
    ar_points_vis = generate_banded_point_grid(W, ar_y_bands, grid_x_steps=32, y_steps_per_band=12, device=device)

    with torch.no_grad():
        for val_step, batch in enumerate(val_dataloader):
            if max_samples and total_samples >= max_samples: break
            
            batch = batch_to_cuda(batch, device)
            B = batch['input'].shape[0]
            total_samples += B
            
            batch_tc_preds, batch_ar_preds = [], []
            
            # Process each image in the batch independently
            for b in range(B):
                sam_img = batch['input'][b]
                cgnet_img = batch['cgnet_input'][b]
                H, W = sam_img.shape[1], sam_img.shape[2]
                
                # Run the pipeline
                final_masks, final_classes, final_scores = pipeline.generate_objects(
                    sam_image=sam_img, 
                    cgnet_image=cgnet_img,
                    points_per_batch=64, 
                    iou_thresh=0.4, 
                    conf_thresh=0.6
                )
                
                # Combine overlapping masks into a single final semantic mask
                if final_masks is None:
                    tc_pred = torch.zeros((1, 1, H, W), device=device)
                    ar_pred = torch.zeros((1, 1, H, W), device=device)
                else:
                    tc_m = final_masks[final_classes == 1]
                    ar_m = final_masks[final_classes == 2]
                    
                    tc_pred = tc_m.sum(dim=0).clamp(max=1).unsqueeze(0).unsqueeze(0) if len(tc_m) > 0 else torch.zeros((1, 1, H, W), device=device)
                    ar_pred = ar_m.sum(dim=0).clamp(max=1).unsqueeze(0).unsqueeze(0) if len(ar_m) > 0 else torch.zeros((1, 1, H, W), device=device)
                
                batch_tc_preds.append(tc_pred)
                batch_ar_preds.append(ar_pred)

                # ---> ADDED: Visualization Logging <---
                if val_step == 0 and b < 2 and getattr(worker_args, 'wandb', False):
                    gt_mask = batch['gt_mask'][b]
                    log_pipeline_visualizations(
                        gt_mask=gt_mask, 
                        tc_pred=tc_pred, 
                        ar_pred=ar_pred, 
                        tc_points=tc_points_vis, 
                        ar_points=ar_points_vis, 
                        step=val_step, 
                        image_idx=b
                    )
            
            # GT Formatting
            masks_gt = batch['gt_mask']
            masks_ar_gts = [(mask == 2).to(torch.uint8)[None, None, :] for mask in masks_gt]
            masks_tc_gts = [(mask == 1).to(torch.uint8)[None, None, :] for mask in masks_gt]
            
            # Update Metrics
            tc_metrics.update(batch_tc_preds, masks_tc_gts, batch['index_name'])
            ar_metrics.update(batch_ar_preds, masks_ar_gts, batch['index_name'])
            
            valid_pbar.update(1)
            
    valid_pbar.close()
    
    # Force wandb to commit the step if we logged images
    if getattr(worker_args, 'wandb', False):
        import wandb
        wandb.log({"eval_complete": True})

    ar_metric_dict, _ = ar_metrics.compute()
    tc_metric_dict, _ = tc_metrics.compute()
    
    return {
        'pipeline': 'Propose_and_Score',
        'miou_ar': ar_metric_dict['Mean Foreground IoU'],
        'miou_tc': tc_metric_dict['Mean Foreground IoU'],
        'mean_acc_ar': ar_metric_dict['Mean Acc'],
        'mean_acc_tc': tc_metric_dict['Mean Acc'],
    }


# ========================================== #
# 5. Main Execution                          #
# ========================================== #
def main_worker(worker_id, worker_args):
    set_randomness()
    device, local_rank = setup_device_and_distributed(worker_id, worker_args)
    print(f"Worker {worker_id} initialized on device {device}.")
    
    # Dataset
    val_dataset = ClimateDataset(data_dir=worker_args.data_dir, train_flag=False, augmented=False, generate_prompt=False)
    
    if hasattr(worker_args, 'debugging') and worker_args.debugging:
        val_indices = list(range(min(getattr(worker_args, 'debug_val_size', 20), len(val_dataset))))
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
        
    val_dataloader = DataLoader(val_dataset, batch_size=getattr(worker_args, 'val_bs', 2), shuffle=False, 
                                num_workers=getattr(worker_args, 'num_workers', 2), collate_fn=val_dataset.collate_fn, 
                                worker_init_fn=partial(worker_init_fn, base_seed=3407))
    
    # Init SAM
    climatesam = ClimateSAM(model_type=worker_args.sam_type, mlp_ratio=worker_args.image_encoder_mlp_ratio, enable_wandb_logging=False).to(device)
    sam_weights = os.path.join(worker_args.exp_dir, f"{worker_args.encoder_weights_name}.pth")
    checkpoint = torch.load(sam_weights, map_location=device)
    climatesam.image_encoder.load_state_dict(checkpoint['image_encoder'])
    climatesam.mask_decoder.load_state_dict(checkpoint['mask_decoder'])
    print(f"✓ SAM weights loaded from {sam_weights}")
    
    # Init CGNet Classifier
    cgnet_weights = os.path.join(worker_args.exp_dir, "cgnet_weight.pth")
    classifier = CGNetPatchClassifier(pretrained_weights_path=cgnet_weights, device=device, in_channels=4).to(device)
    
    # Build Pipeline
    pipeline = ProposeAndScorePipeline(sam_model=climatesam, classifier_model=classifier, device=device)
    
    # Run Evaluation
    print("\n" + "="*60)
    print("RUNNING PROPOSE & SCORE VALIDATION")
    print("="*60)
    
    ar_metrics = StreamSegMetrics(class_names=['Background', 'Foreground'])
    tc_metrics = StreamSegMetrics(class_names=['Background', 'Foreground'])
    
    # Pass worker_args so the function knows whether to log to W&B
    results = validate_propose_and_score(val_dataloader, ar_metrics, tc_metrics, pipeline, device, worker_args)
    
    print("\n" + "="*60)
    print(f"✓ Final mIoU TC: {results['miou_tc']:.4f}")
    print(f"✓ Final mIoU AR: {results['miou_ar']:.4f}")
    print("="*60)
    
    if getattr(worker_args, 'wandb', False):
        wandb.log(results)

if __name__ == '__main__':
    args = parse()
    if getattr(args, 'wandb', False):
        wandb.init(project=getattr(args, 'project_name', "climate-sam-amg"), name=getattr(args, 'run_name', "AMG_test"), config=vars(args))
        
    used_gpu = os.environ['CUDA_VISIBLE_DEVICES'].split(',') if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else get_idle_gpu(gpu_num=1)
    args.used_gpu, args.gpu_num = used_gpu, len(used_gpu)
    main_worker(0, args)