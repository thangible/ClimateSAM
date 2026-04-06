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
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
import copy
import wandb

import torch.distributed as dist
import torch.multiprocessing as mp
from functools import partial
from torch.utils.data import DataLoader
from tqdm import tqdm
from contextlib import nullcontext
from evaluator import StreamSegMetrics

from utility import batch_to_cuda, get_idle_gpu, get_idle_port, set_randomness, plot_mask_with_points_and_bbox, setup_device_and_distributed, worker_init_fn
from parser_config import parse
from model.climatesam import ClimateSAM
from dataset.climatenet import ClimateDataset

# ============================================================
# 1. YOLO BBOX HELPERS
# ============================================================

def parse_climatenet_bboxes(batch_ar, batch_tc, img_h, img_w):
    """Transforms absolute [xmin, ymin, xmax, ymax] into normalized [class_id, cx, cy, w, h]"""
    B = len(batch_ar)
    formatted_bboxes = []
    for b in range(B):
        b_boxes = []
        if batch_tc[b] is not None and len(batch_tc[b]) > 0:
            tc_boxes = batch_tc[b].view(-1, 4)
            for box in tc_boxes:
                xmin, ymin, xmax, ymax = box.tolist()
                b_boxes.append([0.0, (xmin+xmax)/2/img_w, (ymin+ymax)/2/img_h, (xmax-xmin)/img_w, (ymax-ymin)/img_h])
        if batch_ar[b] is not None and len(batch_ar[b]) > 0:
            ar_boxes = batch_ar[b].view(-1, 4)
            for box in ar_boxes:
                xmin, ymin, xmax, ymax = box.tolist()
                b_boxes.append([1.0, (xmin+xmax)/2/img_w, (ymin+ymax)/2/img_h, (xmax-xmin)/img_w, (ymax-ymin)/img_h])
        formatted_bboxes.append(b_boxes)
    return formatted_bboxes

def build_grid_targets_single_class(boxes_list, grid_shape, device, class_id):
    """Transforms list of bboxes into grid-based YOLO targets for a specific class"""
    B = len(boxes_list)
    GH, GW = grid_shape
    # 5 channels: obj, x, y, w, h
    targets = torch.zeros((B, 5, GH, GW), device=device)
    for b in range(B):
        for box in boxes_list[b]:
            cls_id, cx, cy, w, h = box
            if cls_id != class_id:
                continue
            gi, gj = min(max(int(cy * GH), 0), GH - 1), min(max(int(cx * GW), 0), GW - 1)
            targets[b, 0, gi, gj] = 1.0  
            targets[b, 1, gi, gj] = cx * GW - gj
            targets[b, 2, gi, gj] = cy * GH - gi
            targets[b, 3, gi, gj] = w
            targets[b, 4, gi, gj] = h
    return targets

def yolo_detection_loss_single_class(preds, targets):
    """Computes BCE for objectness and MSE for boxes for a single class map."""
    obj_mask = targets[:, 0] == 1.0
    noobj_mask = targets[:, 0] == 0.0

    loss_obj = F.binary_cross_entropy_with_logits(preds[:, 0][obj_mask], targets[:, 0][obj_mask]) if obj_mask.sum() > 0 else 0.0
    loss_noobj = F.binary_cross_entropy_with_logits(preds[:, 0][noobj_mask], targets[:, 0][noobj_mask]) if noobj_mask.sum() > 0 else 0.0
    loss_conf = loss_obj + 0.5 * loss_noobj

    if obj_mask.sum() == 0:
        return loss_conf, (loss_conf.item(), 0.0)

    pos_preds = preds.permute(0, 2, 3, 1)[obj_mask]           
    pos_targets = targets.permute(0, 2, 3, 1)[obj_mask]       

    loss_box = F.mse_loss(torch.sigmoid(pos_preds[:, 1:3]), pos_targets[:, 1:3]) + \
               F.mse_loss(torch.sigmoid(pos_preds[:, 3:5]), pos_targets[:, 3:5])

    total_loss = loss_conf + 5.0 * loss_box
    return total_loss, (loss_conf.item(), loss_box.item())

# ============================================================
# 2. TOKEN-GATED BBOX PROMPTER (DUAL HEAD ARCHITECTURE)
# ============================================================

class TokenGatedDetectionHead(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        # Separate gating mechanisms
        self.ar_gate = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.Sigmoid())
        self.tc_gate = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.Sigmoid())
        
        # Separate heads for AR and TC (5 channels: obj, x, y, w, h)
        self.ar_conv_block = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embedding_dim // 2, 5, kernel_size=1) 
        )
        
        self.tc_conv_block = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embedding_dim // 2, 5, kernel_size=1) 
        )

    def forward(self, x, ar_token, tc_token):
        # x shape: [B, 256, 64, 64]
        B = x.shape[0]
        # Expand tokens if they are [1, 256]
        if ar_token.shape[0] == 1: ar_token = ar_token.expand(B, -1)
        if tc_token.shape[0] == 1: tc_token = tc_token.expand(B, -1)
        
        ar_w = self.ar_gate(ar_token).unsqueeze(-1).unsqueeze(-1)
        tc_w = self.tc_gate(tc_token).unsqueeze(-1).unsqueeze(-1)
        
        ar_gated_x = x * ar_w
        tc_gated_x = x * tc_w
        
        ar_logits = self.ar_conv_block(ar_gated_x)
        tc_logits = self.tc_conv_block(tc_gated_x)
        
        return ar_logits, tc_logits

class SAMBBoxPrompter(nn.Module):
    def __init__(self):
        super().__init__()
        self.det_head = TokenGatedDetectionHead()

    def forward(self, image_embeddings, ar_token, tc_token):
        return self.det_head(image_embeddings, ar_token, tc_token)

    @torch.no_grad()
    def _process_single_class_logits(self, logits, original_h, original_w, conf_threshold, iou_threshold, enlarge_ratio):
        B, C, GH, GW = logits.shape
        logits_permuted = logits.permute(0, 2, 3, 1)
        pred_conf = torch.sigmoid(logits_permuted[..., 0])

        bbox_prompts = []

        for b in range(B):
            mask = pred_conf[b] > conf_threshold
            if mask.sum() == 0:
                bbox_prompts.append(None)
                continue

            pos_preds = logits_permuted[b][mask]
            grid_y, grid_x = torch.where(mask)

            dxdy = torch.sigmoid(pos_preds[:, 1:3])
            wh = torch.sigmoid(pos_preds[:, 3:5])

            # Map back to 0-1
            cx = (grid_x.float() + dxdy[:, 0]) / GW
            cy = (grid_y.float() + dxdy[:, 1]) / GH
            
            w_abs, h_abs = wh[:, 0] * original_w, wh[:, 1] * original_h
            cx_abs, cy_abs = cx * original_w, cy * original_h
            
            pad_w, pad_h = w_abs * enlarge_ratio, h_abs * enlarge_ratio
            x_min = torch.clamp(cx_abs - (w_abs / 2) - pad_w, min=0)
            y_min = torch.clamp(cy_abs - (h_abs / 2) - pad_h, min=0)
            x_max = torch.clamp(cx_abs + (w_abs / 2) + pad_w, max=original_w - 1)
            y_max = torch.clamp(cy_abs + (h_abs / 2) + pad_h, max=original_h - 1)

            boxes = torch.stack((x_min, y_min, x_max, y_max, pred_conf[b][mask]), dim=1)
            keep_idx = ops.nms(boxes[:, :4], boxes[:, 4], iou_threshold=iou_threshold)
            boxes = boxes[keep_idx][:, :4]
            
            bbox_prompts.append(boxes.unsqueeze(1) if len(boxes) > 0 else None)

        return bbox_prompts

    @torch.no_grad()
    def get_prompts(self, ar_logits, tc_logits, original_h, original_w, conf_threshold=0.5, iou_threshold=0.4, enlarge_ratio=0.0):
        """Converts separate AR and TC YOLO grid logits into BBox prompts for SAM."""
        ar_bbox_prompts = self._process_single_class_logits(
            ar_logits, original_h, original_w, conf_threshold, iou_threshold, enlarge_ratio
        )
        tc_bbox_prompts = self._process_single_class_logits(
            tc_logits, original_h, original_w, conf_threshold, iou_threshold, enlarge_ratio
        )

        return {'ar_bbox_prompts': ar_bbox_prompts, 'tc_bbox_prompts': tc_bbox_prompts}

# ============================================================
# 3. TRAINING LOOP
# ============================================================

def train_one_epoch(epoch, train_dataloader, climatesam, prompter, optimizer, scheduler, device, local_rank, worker_args, max_epoch_num, scaler, gradient_accumulation_steps):
    climatesam.eval()
    prompter.train()
    
    batch_pbar = tqdm(total=len(train_dataloader), desc=f'Epoch {epoch}/{max_epoch_num}', leave=True) if local_rank == 0 else None
    epoch_loss_dict = {"total": 0, "ar_conf": 0, "ar_box": 0, "tc_conf": 0, "tc_box": 0}
    
    for train_step, batch in enumerate(train_dataloader):
        batch = batch_to_cuda(batch, device)

        # 1. Extract SAM Embeddings (Frozen)
        with torch.no_grad():
            image_embeddings, _, _, _ = climatesam.encode_images(batch['input'])
            image_embeddings = image_embeddings.detach()
            
            # Extract task tokens
            ar_token = climatesam.mask_decoder.hf_mlp_ar(climatesam.mask_decoder.hf_token_ar.weight.to(device)).detach()
            tc_token = climatesam.mask_decoder.hf_mlp_tc(climatesam.mask_decoder.hf_token_tc.weight.to(device)).detach()

        # 2. Forward through BBox Prompter
        with torch.amp.autocast('cuda'):
            ar_logits, tc_logits = prompter(image_embeddings, ar_token, tc_token)
            
            # 3. Build Separate YOLO Targets
            img_h, img_w = batch['gt_mask'][0].shape
            bboxes = parse_climatenet_bboxes(batch['ar_bbox_prompts'], batch['tc_bbox_prompts'], img_h, img_w)
            
            ar_targets = build_grid_targets_single_class(bboxes, grid_shape=ar_logits.shape[2:], device=device, class_id=1.0)
            tc_targets = build_grid_targets_single_class(bboxes, grid_shape=tc_logits.shape[2:], device=device, class_id=0.0)
            
            # 4. Compute Loss
            ar_loss, (ar_l_conf, ar_l_box) = yolo_detection_loss_single_class(ar_logits, ar_targets)
            tc_loss, (tc_l_conf, tc_l_box) = yolo_detection_loss_single_class(tc_logits, tc_targets)
            
            total_loss = ar_loss + tc_loss
            loss_scaled = total_loss / gradient_accumulation_steps

        # 5. Backward & Step
        scaler.scale(loss_scaled).backward()
        
        epoch_loss_dict["total"] += total_loss.item()
        epoch_loss_dict["ar_conf"] += ar_l_conf
        epoch_loss_dict["ar_box"] += ar_l_box
        epoch_loss_dict["tc_conf"] += tc_l_conf
        epoch_loss_dict["tc_box"] += tc_l_box

        if (train_step + 1) % gradient_accumulation_steps == 0 or (train_step + 1) == len(train_dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if batch_pbar:
            batch_pbar.update(1)
            batch_pbar.set_postfix({'loss': f"{total_loss.item():.4f}"})

    if batch_pbar: batch_pbar.close()
    scheduler.step()
    
    if local_rank == 0 and worker_args.wandb:
        steps = len(train_dataloader)
        wandb.log({
            "train/loss_total": epoch_loss_dict["total"] / steps,
            "train/ar_loss_conf": epoch_loss_dict["ar_conf"] / steps,
            "train/ar_loss_box": epoch_loss_dict["ar_box"] / steps,
            "train/tc_loss_conf": epoch_loss_dict["tc_conf"] / steps,
            "train/tc_loss_box": epoch_loss_dict["tc_box"] / steps,
            "epoch": epoch, "lr": scheduler.get_last_lr()[0]
        }, step=epoch)

# ============================================================
# 4. EVALUATION LOOP
# ============================================================

@torch.no_grad()
def validate_one_epoch(epoch, val_dataloader, ar_metrics, tc_metrics, climatesam, prompter, device, worker_args):
    climatesam.eval()
    prompter.eval()
    valid_pbar = tqdm(total=len(val_dataloader), desc='Validating', leave=False)
    
    for val_step, batch in enumerate(val_dataloader):
        batch = batch_to_cuda(batch, device)
        img_h, img_w = batch['gt_mask'][0].shape

        # 1. Encode & Generate Prompts
        image_embeddings, interm_features, image_input, ori_img_size = climatesam.encode_images(batch['input'])
        ar_token = climatesam.mask_decoder.hf_mlp_ar(climatesam.mask_decoder.hf_token_ar.weight.to(device))
        tc_token = climatesam.mask_decoder.hf_mlp_tc(climatesam.mask_decoder.hf_token_tc.weight.to(device))

        ar_logits, tc_logits = prompter(image_embeddings, ar_token, tc_token)
        
        prompt_dict = prompter.get_prompts(
            ar_logits, tc_logits, img_h, img_w, 
            conf_threshold=0.5, iou_threshold=0.4, enlarge_ratio=getattr(worker_args, 'prompt_enlarge_ratio', 0.0)
        )

        # 2. Forward SAM using predicted bounding boxes
        tc_pred_masks, ar_pred_masks, _ = climatesam.forward(
            image_input=image_input,
            image_embeddings=image_embeddings,
            interm_embeddings=interm_features,
            ori_img_size=ori_img_size,
            ar_bbox_prompts=prompt_dict['ar_bbox_prompts'],
            tc_bbox_prompts=prompt_dict['tc_bbox_prompts']
        )
        
        # 3. Update Metrics
        masks_gt = batch['gt_mask']
        masks_ar_gts = [(m == 2).to(torch.uint8)[None, None, :] for m in masks_gt]
        masks_tc_gts = [(m == 1).to(torch.uint8)[None, None, :] for m in masks_gt]
        
        tc_metrics.update([[m] for m in tc_pred_masks], masks_tc_gts, batch['index_name'])
        ar_metrics.update([[m] for m in ar_pred_masks], masks_ar_gts, batch['index_name'])

        # 4. Visualization
        if val_step == 0 and worker_args.wandb:
            for i in range(min(2, len(masks_gt))):
                save_path = os.path.join(worker_args.exp_dir, worker_args.run_name, 'images', f"epoch_{epoch}_img_{i}.png")
                fig = plot_mask_with_points_and_bbox(
                    mask=masks_gt[i], 
                    ar_bbox=prompt_dict['ar_bbox_prompts'][i], 
                    tc_bbox=prompt_dict['tc_bbox_prompts'][i],
                    tc_pred_mask=tc_pred_masks[i], 
                    ar_pred_mask=ar_pred_masks[i], 
                    save_path=save_path, axis=True, title=f"Epoch {epoch} - Pred {i}"
                )
                wandb.log({f"val/visuals_img_{i}": wandb.Image(fig)}, commit=False)

        valid_pbar.update(1)
        
    valid_pbar.close()
    
    # 5. Compute Final Metrics
    ar_dict, _ = ar_metrics.compute()
    tc_dict, _ = tc_metrics.compute()
    
    if worker_args.wandb:
        wandb.log({
            "val/miou_ar": ar_dict['Mean Foreground IoU'],
            "val/miou_tc": tc_dict['Mean Foreground IoU'],
            "val/acc_ar": ar_dict['Mean Acc'],
            "val/acc_tc": tc_dict['Mean Acc'],
            "epoch": epoch
        }, step=epoch)
        
    ar_metrics.reset()
    tc_metrics.reset()
    return tc_dict['Mean Foreground IoU'], ar_dict['Mean Foreground IoU']

# ============================================================
# 5. DATA & MODEL SETUP
# ============================================================

def set_up_dataset(worker_args):
    train_dataset = ClimateDataset(data_dir=worker_args.data_dir, train_flag=True, generate_prompt=True)
    val_dataset = ClimateDataset(data_dir=worker_args.data_dir, train_flag=False, generate_prompt=True)
    
    if getattr(worker_args, 'debugging', False):
        train_dataset = torch.utils.data.Subset(train_dataset, range(min(10, len(train_dataset))))
        val_dataset = torch.utils.data.Subset(val_dataset, range(min(5, len(val_dataset))))
        worker_args.max_epoch_num = 2
        worker_args.valid_per_epochs = 1

    actual_train_bs = max(1, worker_args.train_bs // getattr(worker_args, 'gradient_accumulation_steps', 1))
    
    g = torch.Generator().manual_seed(3407)
    train_loader = DataLoader(train_dataset, batch_size=actual_train_bs, shuffle=True, num_workers=4, collate_fn=train_dataset.collate_fn, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=worker_args.val_bs, shuffle=False, num_workers=2, collate_fn=val_dataset.collate_fn, generator=g)
    return train_loader, val_loader

def set_up_model(worker_args, device):
    climatesam = ClimateSAM(
        model_type=worker_args.sam_type, 
        mlp_ratio=worker_args.image_encoder_mlp_ratio,
        enable_wandb_logging=False
    ).to(device)
    
    weights_path = os.path.join(worker_args.exp_dir, f"{worker_args.encoder_weights_name}.pth")
    ckpt = torch.load(weights_path, map_location=device)
    climatesam.image_encoder.load_state_dict(ckpt['image_encoder'])
    climatesam.mask_decoder.load_state_dict(ckpt['mask_decoder'])
    climatesam.input_adapter.load_state_dict(ckpt['input_adapter'])
    
    for param in climatesam.parameters():
        param.requires_grad = False
        
    prompter = SAMBBoxPrompter().to(device)
    return climatesam, prompter

# ============================================================
# 6. MAIN WORKER
# ============================================================

def main_worker(worker_id, worker_args):
    set_randomness()
    device, local_rank = setup_device_and_distributed(worker_id, worker_args)
    
    train_loader, val_loader = set_up_dataset(worker_args)
    climatesam, prompter = set_up_model(worker_args, device)
    
    # Optimizer only trains the prompter
    optimizer = torch.optim.AdamW(prompter.parameters(), lr=getattr(worker_args, 'lr', 1e-4), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=worker_args.max_epoch_num, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda') 

    ar_metrics = StreamSegMetrics(class_names=['Background', 'Foreground'])
    tc_metrics = StreamSegMetrics(class_names=['Background', 'Foreground'])

    best_miou = 0.0
    for epoch in range(1, worker_args.max_epoch_num + 1):
        train_one_epoch(epoch, train_loader, climatesam, prompter, optimizer, scheduler, device, local_rank, worker_args, worker_args.max_epoch_num, scaler, getattr(worker_args, 'gradient_accumulation_steps', 1))
        
        if epoch % getattr(worker_args, 'valid_per_epochs', 1) == 0 or epoch == worker_args.max_epoch_num:
            miou_tc, miou_ar = validate_one_epoch(epoch, val_loader, ar_metrics, tc_metrics, climatesam, prompter, device, worker_args)
            print(f"Epoch {epoch} - TC mIoU: {miou_tc:.4f}, AR mIoU: {miou_ar:.4f}")
            
            avg_miou = (miou_tc + miou_ar) / 2
            if avg_miou > best_miou:
                best_miou = avg_miou
                save_path = os.path.join(worker_args.exp_dir, f"best_bbox_prompter_{worker_args.run_name}.pth")
                torch.save(prompter.state_dict(), save_path)
                print(f"*** Best model saved! (Mean IoU: {best_miou:.4f}) ***")

if __name__ == '__main__':
    args = parse()
    if getattr(args, 'wandb', False):
        wandb.init(project=getattr(args, 'project_name', "climate-sam"), name=getattr(args, 'run_name', "bbox_prompter"), config=vars(args))
    
    used_gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')
    args.used_gpu = used_gpu
    main_worker(0, args)