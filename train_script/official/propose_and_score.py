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
import numpy as np

def log_pipeline_visualizations(gt_mask, tc_pred, ar_pred, tc_points, ar_points, bg_points, step, image_idx):
    """
    Creates a 1x3 plot showing:
    1. GT Mask + TC Grid Prompts + BG Negative Prompts
    2. GT Mask + AR Grid Prompts + BG Negative Prompts
    3. Final Accepted Predicted Masks
    """
    gt = gt_mask.cpu().numpy()
    tc_m = tc_pred.squeeze().cpu().numpy()
    ar_m = ar_pred.squeeze().cpu().numpy()
    
    # Flatten from [N, 1, 2] to [N, 2] for plotting
    tc_pts = tc_points.view(-1, 2).cpu().numpy()
    ar_pts = ar_points.view(-1, 2).cpu().numpy()
    bg_pts = bg_points.view(-1, 2).cpu().numpy()
    
    pred_combined = np.zeros_like(gt)
    pred_combined[tc_m > 0] = 1
    pred_combined[ar_m > 0] = 2

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # --- Plot 1: GT + TC + BG ---
    axes[0].imshow(gt, cmap='viridis', interpolation='nearest')
    if len(bg_pts) > 0:
        axes[0].scatter(bg_pts[:, 0], bg_pts[:, 1], c='white', marker='x', s=2, alpha=0.3, label='BG (Neg)')
    if len(tc_pts) > 0:
        axes[0].scatter(tc_pts[:, 0], tc_pts[:, 1], c='red', s=6, alpha=0.8, label='TC (Pos)')
    axes[0].set_title('GT Mask + TC & BG Prompts')
    axes[0].axis('off')
    
    # --- Plot 2: GT + AR + BG ---
    axes[1].imshow(gt, cmap='viridis', interpolation='nearest')
    if len(bg_pts) > 0:
        axes[1].scatter(bg_pts[:, 0], bg_pts[:, 1], c='white', marker='x', s=2, alpha=0.3, label='BG (Neg)')
    if len(ar_pts) > 0:
        axes[1].scatter(ar_pts[:, 0], ar_pts[:, 1], c='cyan', s=6, alpha=0.8, label='AR (Pos)')
    axes[1].set_title('GT Mask + AR & BG Prompts')
    axes[1].axis('off')
    
    # --- Plot 3: Final Accepted Segment ---
    axes[2].imshow(gt, cmap='gray', alpha=0.3) 
    axes[2].imshow(pred_combined, cmap='viridis', alpha=0.8, interpolation='nearest')
    axes[2].set_title('Final Accepted Segments (1=TC, 2=AR)')
    axes[2].axis('off')

    plt.tight_layout()
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
def extract_and_score_masks(cgnet_image, masks, classifier_model, crop_size=(128, 128)):
    model_device = next(classifier_model.parameters()).device
    if cgnet_image.device != model_device:
        cgnet_image = cgnet_image.to(model_device)
    masks = masks.to(device=model_device)
    N, H, W = masks.shape
    
    valid_masks = []
    mask_crops = []
    
    for i in range(N):
        mask = masks[i]
        if mask.sum() < 10: 
            continue
            
        y_indices, x_indices = torch.where(mask > 0)
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()
        
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
    def generate_objects(self, sam_image, cgnet_image, tc_points, ar_points, bg_points, points_per_batch=64, iou_thresh=0.5, conf_thresh=0.6):
        C, H, W = sam_image.shape
        self.sam.set_infer_img(sam_image.unsqueeze(0))
        
        all_masks, all_scores, all_classes, all_boxes = [], [], [], []
        raw_tc = torch.zeros((H, W), device=self.device)
        raw_ar = torch.zeros((H, W), device=self.device)
        
        if tc_points is not None and len(tc_points) > 0:
            all_masks, all_scores, all_classes, all_boxes, raw_tc = self._process_grid(
                points=tc_points, bg_points=bg_points, cgnet_image=cgnet_image, points_per_batch=points_per_batch, 
                prompt_type='TC', conf_thresh=conf_thresh, target_class_id=1, 
                accumulators=(all_masks, all_scores, all_classes, all_boxes)
            )
        
        if ar_points is not None and len(ar_points) > 0:
            all_masks, all_scores, all_classes, all_boxes, raw_ar = self._process_grid(
                points=ar_points, bg_points=bg_points, cgnet_image=cgnet_image, points_per_batch=points_per_batch, 
                prompt_type='AR', conf_thresh=conf_thresh, target_class_id=2, 
                accumulators=(all_masks, all_scores, all_classes, all_boxes)
            )
        
        if not all_masks:
            return None, None, None, raw_tc, raw_ar
            
        global_masks = torch.cat(all_masks, dim=0)
        global_scores = torch.cat(all_scores, dim=0)
        global_classes = torch.cat(all_classes, dim=0)
        global_boxes = torch.cat(all_boxes, dim=0)
        
        keep_indices = ops.batched_nms(global_boxes, global_scores, global_classes, iou_thresh)
        
        return global_masks[keep_indices], global_classes[keep_indices], global_scores[keep_indices], raw_tc, raw_ar

    def _process_grid(self, points, bg_points, cgnet_image, points_per_batch, prompt_type, conf_thresh, target_class_id, accumulators):
        all_masks, all_scores, all_classes, all_boxes = accumulators
        num_points = points.shape[0]
        raw_mask_accumulator = []
        
        # Safely limit Background Points to prevent Out-Of-Memory errors
        K = bg_points.shape[0] if bg_points is not None else 0
        if K > 64:
            indices = torch.randperm(K, device=self.device)[:64]
            bg_sampled = bg_points[indices].view(-1, 2)
            K = 64
        elif K > 0:
            bg_sampled = bg_points.view(-1, 2)
        else:
            bg_sampled = None

        for i in range(0, num_points, points_per_batch):
            batch_pos = points[i:i+points_per_batch] # Shape: [B, 1, 2]
            B = batch_pos.shape[0]
            
            # Concat 1 Positive Point with K Negative Points
            if K > 0:
                bg_exp = bg_sampled.unsqueeze(0).expand(B, K, 2)
                batch_coords = torch.cat([batch_pos, bg_exp], dim=1) # Shape: [B, 1+K, 2]
                
                pos_lbl = torch.ones((B, 1), dtype=torch.float32, device=self.device)
                neg_lbl = torch.zeros((B, K), dtype=torch.float32, device=self.device)
                batch_labels = torch.cat([pos_lbl, neg_lbl], dim=1)  # Shape: [B, 1+K]
            else:
                batch_coords = batch_pos
                batch_labels = torch.ones((B, 1), dtype=torch.float32, device=self.device)
            
            # Format exactly for SAM's expected list structure: [(1, B, 1+K, 2), (1, B, 1+K)]
            formatted_points = [(batch_coords.unsqueeze(0), batch_labels.unsqueeze(0))]
            
            if prompt_type == 'TC':
                tc_masks, _ = self.sam.infer(tc_point_prompts=formatted_points, ar_point_prompts=None)
                raw_masks = tc_masks[0].squeeze(1)
            else:
                _, ar_masks = self.sam.infer(tc_point_prompts=None, ar_point_prompts=formatted_points)
                raw_masks = ar_masks[0].squeeze(1)
            
            raw_mask_accumulator.append(raw_masks.cpu())
                
            scores, classes, valid_masks = extract_and_score_masks(cgnet_image, raw_masks, self.classifier)
            
            valid_idx = (scores > conf_thresh) & (classes == target_class_id)
            scores, classes, valid_masks = scores[valid_idx], classes[valid_idx], valid_masks[valid_idx]
            
            if len(scores) == 0:
                continue
                
            boxes = ops.masks_to_boxes(valid_masks)
            all_masks.append(valid_masks)
            all_scores.append(scores)
            all_classes.append(classes)
            all_boxes.append(boxes)
            
        if len(raw_mask_accumulator) > 0:
            assembled_raw_mask = torch.cat(raw_mask_accumulator, dim=0).to(self.device).sum(dim=0).clamp(max=1)
        else:
            _, H, W = cgnet_image.shape
            assembled_raw_mask = torch.zeros((H, W), device=self.device)
            
        return all_masks, all_scores, all_classes, all_boxes, assembled_raw_mask


# ========================================== #
# 4. Evaluation Loop                         #
# ========================================== #
@torch.no_grad()
def validate_propose_and_score(train_dataloader, val_dataloader, ar_metrics, tc_metrics, pipeline, device, worker_args, max_samples=None):
    from utility import plot_mask_with_points_and_bbox
    import os
    import wandb
    
    pipeline.sam.eval()
    pipeline.classifier.eval()
    
    total_samples = 0
    valid_pbar = tqdm(total=len(val_dataloader), desc='Propose & Score Eval', leave=False)
    
    # Generate points maintaining the [N, 1, 2] structure
    tc_pts_raw, _ = train_dataloader.dataset.generate_smart_grid_prompts('tc', grid_size=(32, 32), jitter_amount=0.5, threshold=0.20)
    ar_pts_raw, _ = train_dataloader.dataset.generate_smart_grid_prompts('ar', grid_size=(16, 16), jitter_amount=0.5, threshold=0.50)
    bg_pts_raw, _ = train_dataloader.dataset.generate_smart_grid_prompts('bg', grid_size=(64, 64), jitter_amount=0.5, threshold=0.99)

    tc_points_vis = tc_pts_raw.to(device) if tc_pts_raw is not None else torch.empty((0, 1, 2), device=device)
    ar_points_vis = ar_pts_raw.to(device) if ar_pts_raw is not None else torch.empty((0, 1, 2), device=device)
    bg_points_vis = bg_pts_raw.to(device) if bg_pts_raw is not None else torch.empty((0, 1, 2), device=device)
    
    # Formatted explicitly for utility plotting
    tc_points_vis_copy = tc_points_vis.view(-1, 2).clone()
    ar_points_vis_copy = ar_points_vis.view(-1, 2).clone()

    with torch.no_grad():
        for val_step, batch in enumerate(val_dataloader):
            if max_samples and total_samples >= max_samples: break
            
            batch = batch_to_cuda(batch, device)
            B = batch['input'].shape[0]
            total_samples += B
            
            batch_tc_preds, batch_ar_preds = [], []
            
            for b in range(B):
                sam_img = batch['input'][b]
                cgnet_img = batch['cgnet_input'][b]
                H, W = sam_img.shape[1], sam_img.shape[2]
                
                final_masks, final_classes, final_scores, raw_tc, raw_ar = pipeline.generate_objects(
                    sam_image=sam_img, 
                    cgnet_image=cgnet_img,
                    tc_points=tc_points_vis,
                    ar_points=ar_points_vis,
                    bg_points=bg_points_vis,
                    points_per_batch=64,
                    iou_thresh=0.4,
                    conf_thresh=0.6
                )
                
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

                if val_step == 0 and b < 4:
                    gt_mask = batch['gt_mask'][b]
                    save_path = os.path.join(worker_args.exp_dir, f"raw_sam_masks_step{val_step}_img{b}.png")
                    
                    fig = plot_mask_with_points_and_bbox(
                        mask=gt_mask, 
                        ar_points=ar_points_vis_copy, 
                        tc_points=tc_points_vis_copy, 
                        tc_pred_mask=raw_tc, 
                        ar_pred_mask=raw_ar, 
                        save_path=save_path, 
                        axis=False, 
                        title="Raw SAM Assembled Masks (No Filter)"
                    )
                    
                    log_pipeline_visualizations(
                        gt_mask=gt_mask, 
                        tc_pred=tc_pred, 
                        ar_pred=ar_pred, 
                        tc_points=tc_points_vis, 
                        ar_points=ar_points_vis, 
                        bg_points=bg_points_vis,
                        step=val_step, 
                        image_idx=b
                    )
                    
                    if getattr(worker_args, 'wandb', False):
                        wandb.log({f"visualizations/raw_sam_batch_{val_step}_img_{b}": wandb.Image(fig)})
            
            masks_gt = batch['gt_mask']
            masks_ar_gts = [(mask == 2).to(torch.uint8)[None, None, :] for mask in masks_gt]
            masks_tc_gts = [(mask == 1).to(torch.uint8)[None, None, :] for mask in masks_gt]
            
            tc_metrics.update(batch_tc_preds, masks_tc_gts, batch['index_name'])
            ar_metrics.update(batch_ar_preds, masks_ar_gts, batch['index_name'])
            
            valid_pbar.update(1)
            
    valid_pbar.close()
    
    if getattr(worker_args, 'wandb', False):
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
    
    train_dataset = ClimateDataset(
        data_dir=worker_args.data_dir, train_flag=True, shot_num=worker_args.shot_num,
        augmented=worker_args.augmented, generate_prompt=True
    )
    val_dataset = ClimateDataset(data_dir=worker_args.data_dir, train_flag=False, augmented=False, generate_prompt=True)
    train_collate_fn = train_dataset.collate_fn
    
    if hasattr(worker_args, 'debugging') and worker_args.debugging:
        val_indices = list(range(min(getattr(worker_args, 'debug_val_size', 20), len(val_dataset))))
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
    
    g = torch.Generator()
    g.manual_seed(3407)
    
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=getattr(worker_args, 'train_bs', 2), shuffle=False is None, num_workers=getattr(worker_args, 'num_workers', 2),
         drop_last=False, collate_fn=train_collate_fn,
        worker_init_fn=partial(worker_init_fn, base_seed=3407), generator=g
    )
    
    val_dataloader = DataLoader(val_dataset, batch_size=getattr(worker_args, 'val_bs', 2), shuffle=False, 
                                num_workers=getattr(worker_args, 'num_workers', 2), collate_fn=val_dataset.collate_fn, 
                                worker_init_fn=partial(worker_init_fn, base_seed=3407))
    
    climatesam = ClimateSAM(model_type=worker_args.sam_type, mlp_ratio=worker_args.image_encoder_mlp_ratio, enable_wandb_logging=False).to(device)
    sam_weights = os.path.join(worker_args.exp_dir, f"{worker_args.encoder_weights_name}.pth")
    checkpoint = torch.load(sam_weights, map_location=device)
    climatesam.image_encoder.load_state_dict(checkpoint['image_encoder'])
    climatesam.mask_decoder.load_state_dict(checkpoint['mask_decoder'])
    print(f"✓ SAM weights loaded from {sam_weights}")
    
    cgnet_weights = os.path.join(worker_args.exp_dir, "cgnet_weight.pth")
    classifier = CGNetPatchClassifier(pretrained_weights_path=cgnet_weights, device=device, in_channels=4).to(device)
    
    pipeline = ProposeAndScorePipeline(sam_model=climatesam, classifier_model=classifier, device=device)
    
    print("\n" + "="*60)
    print("RUNNING PROPOSE & SCORE VALIDATION")
    print("="*60)
    
    ar_metrics = StreamSegMetrics(class_names=['Background', 'Foreground'])
    tc_metrics = StreamSegMetrics(class_names=['Background', 'Foreground'])
    
    results = validate_propose_and_score(
        train_dataloader=train_dataloader, val_dataloader=val_dataloader, 
        ar_metrics=ar_metrics, tc_metrics=tc_metrics, 
        pipeline=pipeline, device=device, worker_args=worker_args
    )
    
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