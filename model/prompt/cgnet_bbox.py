import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
from tqdm import tqdm
import wandb

from .cgnet_module import CGNetModule

# ========================================== #
# 1. Dataset Adapter for ClimateNet          #
# ========================================== #

def parse_climatenet_bboxes(batch_ar, batch_tc, img_h, img_w):
    """
    Adapts the ClimateNet dataloader bounding box outputs to the YOLO format.
    Transforms absolute [xmin, ymin, xmax, ymax] into normalized [class_id, cx, cy, w, h].
    TC = Class 0, AR = Class 1.
    """
    B = len(batch_ar)
    formatted_bboxes = []
    
    for b in range(B):
        b_boxes = []
        
        # Parse Tropical Cyclones (Class 0)
        if batch_tc[b] is not None and len(batch_tc[b]) > 0:
            tc_boxes = batch_tc[b].view(-1, 4) # Flatten from [N, 1, 4] to [N, 4]
            for box in tc_boxes:
                xmin, ymin, xmax, ymax = box.tolist()
                w = (xmax - xmin) / img_w
                h = (ymax - ymin) / img_h
                cx = (xmin + xmax) / 2.0 / img_w
                cy = (ymin + ymax) / 2.0 / img_h
                b_boxes.append([0.0, cx, cy, w, h])
                
        # Parse Atmospheric Rivers (Class 1)
        if batch_ar[b] is not None and len(batch_ar[b]) > 0:
            ar_boxes = batch_ar[b].view(-1, 4) # Flatten from [N, 1, 4] to [N, 4]
            for box in ar_boxes:
                xmin, ymin, xmax, ymax = box.tolist()
                w = (xmax - xmin) / img_w
                h = (ymax - ymin) / img_h
                cx = (xmin + xmax) / 2.0 / img_w
                cy = (ymin + ymax) / 2.0 / img_h
                b_boxes.append([1.0, cx, cy, w, h])
                
        formatted_bboxes.append(b_boxes)
        
    return formatted_bboxes

# ========================================== #
# 2. Target Builder & Loss Functions         #
# ========================================== #

def build_grid_targets(boxes_list, grid_shape, device):
    """
    Transforms a list of bounding boxes into grid-based targets matching the network output.
    """
    B = len(boxes_list)
    GH, GW = grid_shape
    targets = torch.zeros((B, 6, GH, GW), device=device)

    for b in range(B):
        boxes = boxes_list[b]
        if len(boxes) == 0:
            continue

        for box in boxes:
            cls_id, cx, cy, w, h = box

            gi = int(cy * GH)
            gj = int(cx * GW)

            gi = min(max(gi, 0), GH - 1)
            gj = min(max(gj, 0), GW - 1)

            dx = cx * GW - gj
            dy = cy * GH - gi

            targets[b, 0, gi, gj] = 1.0  
            targets[b, 1, gi, gj] = dx
            targets[b, 2, gi, gj] = dy
            targets[b, 3, gi, gj] = w
            targets[b, 4, gi, gj] = h
            targets[b, 5, gi, gj] = cls_id

    return targets

def yolo_detection_loss(preds, targets):
    """
    Computes BCE for objectness, MSE for boxes, and CE for classes.
    """
    obj_mask = targets[:, 0] == 1.0
    noobj_mask = targets[:, 0] == 0.0

    pred_conf = preds[:, 0]
    target_conf = targets[:, 0]

    loss_obj = F.binary_cross_entropy_with_logits(pred_conf[obj_mask], target_conf[obj_mask]) if obj_mask.sum() > 0 else 0.0
    loss_noobj = F.binary_cross_entropy_with_logits(pred_conf[noobj_mask], target_conf[noobj_mask]) if noobj_mask.sum() > 0 else 0.0
    loss_conf = loss_obj + 0.5 * loss_noobj

    if obj_mask.sum() == 0:
        return loss_conf, (loss_conf.item(), 0.0, 0.0)

    preds_permuted = preds.permute(0, 2, 3, 1)     
    targets_permuted = targets.permute(0, 2, 3, 1) 

    pos_preds = preds_permuted[obj_mask]           
    pos_targets = targets_permuted[obj_mask]       

    pred_dxdy = torch.sigmoid(pos_preds[:, 1:3])
    pred_wh = torch.sigmoid(pos_preds[:, 3:5])

    target_dxdy = pos_targets[:, 1:3]
    target_wh = pos_targets[:, 3:5]

    loss_box = F.mse_loss(pred_dxdy, target_dxdy) + F.mse_loss(pred_wh, target_wh)

    pred_cls_logits = pos_preds[:, 5:]
    target_cls = pos_targets[:, 5].long()

    loss_cls = F.cross_entropy(pred_cls_logits, target_cls)

    total_loss = loss_conf + 5.0 * loss_box + loss_cls
    return total_loss, (loss_conf.item(), loss_box.item(), loss_cls.item())

# ========================================== #
# 3. Modified CGNet Architecture             #
# ========================================== #

class CGNetDetectionModule(CGNetModule):
    def __init__(self, num_classes=2, channels=4, M=3, N=21, dropout_flag=False):
        super().__init__(classes=num_classes, channels=channels, M=M, N=N, dropout_flag=dropout_flag)
        
        self.num_classes = num_classes
        self.out_channels = 1 + 4 + num_classes
        
        from .cgnet_module import Conv
        if dropout_flag:
            self.classifier = nn.Sequential(nn.Dropout2d(0.1, False), Conv(256, self.out_channels, 1, 1))
        else:
            self.classifier = nn.Sequential(Conv(256, self.out_channels, 1, 1))

    def forward(self, input):
        output0 = self.level1_0(input)
        output0 = self.level1_1(output0)
        output0 = self.level1_2(output0)
        inp1 = self.sample1(input)
        inp2 = self.sample2(input)
        
        output0_cat = self.b1(torch.cat([output0, inp1], 1))
        output1_0 = self.level2_0(output0_cat)
        for i, layer in enumerate(self.level2):
            output1 = layer(output1_0) if i == 0 else layer(output1)
        output1_cat = self.bn_prelu_2(torch.cat([output1, output1_0, inp2], 1))

        output2_0 = self.level3_0(output1_cat)
        for i, layer in enumerate(self.level3):
            output2 = layer(output2_0) if i == 0 else layer(output2)
        output2_cat = self.bn_prelu_3(torch.cat([output2_0, output2], 1))
       
        grid_out = self.classifier(output2_cat)
        return grid_out

# ========================================== #
# 4. The Prompter Wrapper                    #
# ========================================== #

class CGNetBBoxPrompter:
    def __init__(self, weights_path, device, worker_args, num_classes=2):
        self.device = device
        self.exp_dir = worker_args.exp_dir
        
        self.cgnet_model = CGNetDetectionModule(num_classes=num_classes, channels=4)
        
        if weights_path and os.path.exists(weights_path):
            self.cgnet_model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
            
        self.cgnet_model.to(device)
        self.optimizer = torch.optim.Adam(self.cgnet_model.parameters(), lr=1e-4)
        
        self.wandb = getattr(worker_args, 'wandb', False)
        self.run_name = getattr(worker_args, 'run_name', None)

    def train(self, train_dataloader, val_dataloader, epochs):
        best_val_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            self.cgnet_model.train()
            print(f'\nEpoch {epoch}/{epochs}:')
            epoch_loader = tqdm(train_dataloader, desc="Training")
            
            epoch_loss_sum = 0.0
            epoch_loss_count = 0
            
            epoch_conf_losses = []
            epoch_box_losses = []
            epoch_cls_losses = []

            for batch in epoch_loader:
                features = batch['cgnet_input'].to(device=self.device, dtype=torch.float32)
                
                img_h, img_w = features.shape[2], features.shape[3]
                raw_ar_boxes = batch['ar_bbox_prompts']
                raw_tc_boxes = batch['tc_bbox_prompts']
                
                bboxes = parse_climatenet_bboxes(raw_ar_boxes, raw_tc_boxes, img_h, img_w)

                outputs = self.cgnet_model(features)
                grid_shape = outputs.shape[2:] 

                targets = build_grid_targets(bboxes, grid_shape, self.device)
                
                loss, (l_conf, l_box, l_cls) = yolo_detection_loss(outputs, targets)

                epoch_loader.set_description(f'Train Loss: {loss.item():.4f}')
                
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad() 

                epoch_loss_sum += float(loss.item())
                epoch_loss_count += 1
                epoch_conf_losses.append(l_conf)
                epoch_box_losses.append(l_box)
                epoch_cls_losses.append(l_cls)

            avg_epoch_loss = epoch_loss_sum / epoch_loss_count if epoch_loss_count > 0 else 0.0

            print(f"Train Stats: Avg Loss {avg_epoch_loss:.4f} (Conf: {np.mean(epoch_conf_losses):.4f}, Box: {np.mean(epoch_box_losses):.4f}, Cls: {np.mean(epoch_cls_losses):.4f})")

            if self.wandb:
                log_dict = {
                    "train/total_loss": avg_epoch_loss,
                    "train/loss_conf": float(np.mean(epoch_conf_losses)),
                    "train/loss_box": float(np.mean(epoch_box_losses)),
                    "train/loss_cls": float(np.mean(epoch_cls_losses)),
                    "epoch": epoch,
                }
                try:
                    wandb.log(log_dict, step=epoch)
                except Exception as e:
                    print(f"WandB scalar log failed: {e}")

            # --------------------------------------------------------- #
            # Validation every 5 epochs or on the final epoch         #
            # --------------------------------------------------------- #
            if epoch % 5 == 0 or epoch == epochs:
                val_loss, val_ar_iou, val_tc_iou = self.validate(val_dataloader, epoch)
                print(f"Validation Stats: Loss {val_loss:.4f} | AR BBox IoU: {val_ar_iou:.4f} | TC BBox IoU: {val_tc_iou:.4f}")
                
                if self.wandb:
                    try:
                        wandb.log({
                            "val/loss": val_loss,
                            "val/ar_bbox_iou": val_ar_iou,
                            "val/tc_bbox_iou": val_tc_iou,
                            "epoch": epoch
                        }, step=epoch)
                    except Exception as e:
                        print(f"WandB validation log failed: {e}")

                # Save best model based on validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model()
                    print(f"*** New best model saved with Val Loss: {best_val_loss:.4f} ***")
                    if self.wandb:
                        try:
                            save_path = os.path.join(self.exp_dir, f"cgnet_bbox_weight.pth")
                            wandb.save(save_path)
                        except Exception as e:
                            print(f"WandB save failed: {e}")

    @torch.no_grad()
    def validate(self, val_dataloader, epoch):
        self.cgnet_model.eval()
        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        
        ar_ious = []
        tc_ious = []
        
        plot_saved = False
        
        from utility import plot_mask_with_points_and_bbox  # Assuming it's accessible
        
        for batch in tqdm(val_dataloader, desc="Validating"):
            features = batch['cgnet_input'].to(device=self.device, dtype=torch.float32)
            img_h, img_w = features.shape[2], features.shape[3]
            
            raw_ar_boxes = batch['ar_bbox_prompts']
            raw_tc_boxes = batch['tc_bbox_prompts']
            bboxes = parse_climatenet_bboxes(raw_ar_boxes, raw_tc_boxes, img_h, img_w)
            
            outputs = self.cgnet_model(features)
            grid_shape = outputs.shape[2:] 
            targets = build_grid_targets(bboxes, grid_shape, self.device)
            
            loss, _ = yolo_detection_loss(outputs, targets)
            epoch_loss_sum += float(loss.item())
            epoch_loss_count += 1
            
            # Get BBox Predictions
            prompt_dict = self.get_prompts(features, conf_threshold=0.5, iou_threshold=0.4)
            pred_ar_bboxes = prompt_dict['ar_bbox_prompts']
            pred_tc_bboxes = prompt_dict['tc_bbox_prompts']
            
            # Compute Average Maximum IoU Metric
            for b in range(features.shape[0]):
                # AR IoU Calculation
                gt_ar = raw_ar_boxes[b]  # [N, 1, 4]
                pr_ar = pred_ar_bboxes[b]  # [M, 1, 4]
                if gt_ar is not None and pr_ar is not None:
                    gt_ar_flat = gt_ar.view(-1, 4).to(self.device)
                    pr_ar_flat = pr_ar.view(-1, 4).to(self.device)
                    ious = ops.box_iou(pr_ar_flat, gt_ar_flat)
                    ar_ious.append(ious.max(dim=1)[0].mean().item()) # Avg IoU of matched boxes
                elif gt_ar is None and pr_ar is None:
                    ar_ious.append(1.0) # Correctly predicted empty
                else:
                    ar_ious.append(0.0) # False positive or false negative
                    
                # TC IoU Calculation
                gt_tc = raw_tc_boxes[b]
                pr_tc = pred_tc_bboxes[b]
                if gt_tc is not None and pr_tc is not None:
                    gt_tc_flat = gt_tc.view(-1, 4).to(self.device)
                    pr_tc_flat = pr_tc.view(-1, 4).to(self.device)
                    ious = ops.box_iou(pr_tc_flat, gt_tc_flat)
                    tc_ious.append(ious.max(dim=1)[0].mean().item())
                elif gt_tc is None and pr_tc is None:
                    tc_ious.append(1.0)
                else:
                    tc_ious.append(0.0)
            
            # Save visual plot for the very first batch
            if not plot_saved and self.wandb:
                gt_mask = batch['gt_mask'][0] 
                p_ar = pred_ar_bboxes[0]
                p_tc = pred_tc_bboxes[0]
                
                plot_path = os.path.join(self.exp_dir, f"val_epoch_{epoch}.png")
                
                # Pass GT Mask to display background context, and overlay predicted BBoxes
                plot_mask_with_points_and_bbox(
                    mask=gt_mask,
                    ar_bbox=p_ar,
                    tc_bbox=p_tc,
                    save_path=plot_path,
                    title=f"BBox Predictions - Epoch {epoch}"
                )
                
                try:
                    wandb.log({f"val/predictions": wandb.Image(plot_path)}, step=epoch)
                except Exception as e:
                    print(f"WandB image log failed: {e}")

                plot_saved = True
                
        avg_loss = epoch_loss_sum / epoch_loss_count if epoch_loss_count > 0 else 0.0
        avg_ar_iou = np.mean(ar_ious) if len(ar_ious) > 0 else 0.0
        avg_tc_iou = np.mean(tc_ious) if len(tc_ious) > 0 else 0.0
        
        return avg_loss, avg_ar_iou, avg_tc_iou

    def save_model(self):
        save_path = os.path.join(self.exp_dir, f"cgnet_bbox_weight.pth")
        torch.save(self.cgnet_model.state_dict(), save_path)
        
    @torch.no_grad()
    def get_prompts(self, batch_input, conf_threshold=0.5, iou_threshold=0.4):
        """
        Infers bounding boxes from the input and formats them as SAM-compatible prompts.
        Applies Non-Maximum Suppression (NMS) to filter redundant overlapping boxes.
        
        Returns:
            dict: A dictionary containing lists of tensors for 'ar_bbox_prompts' 
                  and 'tc_bbox_prompts'. Tensor shapes are [N, 1, 4] containing 
                  absolute pixel coordinates [xmin, ymin, xmax, ymax].
        """
        self.cgnet_model.eval()
        outputs = self.cgnet_model(batch_input)
        
        B, C, GH, GW = outputs.shape
        H, W = batch_input.shape[2], batch_input.shape[3]

        # Permute for easier indexing: [B, GH, GW, 7]
        outputs_permuted = outputs.permute(0, 2, 3, 1)
        pred_conf = torch.sigmoid(outputs_permuted[..., 0])

        ar_bbox_prompts = []
        tc_bbox_prompts = []

        for b in range(B):
            # 1. Filter out low-confidence cells
            mask = pred_conf[b] > conf_threshold
            if mask.sum() == 0:
                ar_bbox_prompts.append(None)
                tc_bbox_prompts.append(None)
                continue

            pos_preds = outputs_permuted[b][mask]
            grid_y, grid_x = torch.where(mask)

            # 2. Extract bounding box properties
            dxdy = torch.sigmoid(pos_preds[:, 1:3])
            wh = torch.sigmoid(pos_preds[:, 3:5])
            
            # 3. Extract class predictions
            cls_probs = torch.softmax(pos_preds[:, 5:], dim=1)
            cls_conf, cls_pred = torch.max(cls_probs, dim=1)

            # 4. Map from grid scale back to normalized image coordinates [0, 1]
            cx = (grid_x.float() + dxdy[:, 0]) / GW
            cy = (grid_y.float() + dxdy[:, 1]) / GH
            norm_w = wh[:, 0]
            norm_h = wh[:, 1]

            # 5. Convert to absolute pixel coordinates [x_min, y_min, x_max, y_max] 
            # This is crucial for SAM which expects native image scaling.
            x_min = (cx - norm_w / 2) * W
            y_min = (cy - norm_h / 2) * H
            x_max = (cx + norm_w / 2) * W
            y_max = (cy + norm_h / 2) * H

            # Stack for NMS processing
            boxes = torch.stack((x_min, y_min, x_max, y_max, pred_conf[b][mask], cls_pred.float()), dim=1)
            
            # 6. Apply Non-Maximum Suppression (NMS)
            keep_idx = ops.nms(boxes[:, :4], boxes[:, 4], iou_threshold=iou_threshold)
            boxes = boxes[keep_idx]
            
            # 7. Separate boxes by Class ID (0: TC, 1: AR)
            tc_mask = boxes[:, 5] == 0
            ar_mask = boxes[:, 5] == 1
            
            tc_boxes = boxes[tc_mask][:, :4]
            ar_boxes = boxes[ar_mask][:, :4]
            
            # 8. Format to [N, 1, 4] to mimic original climatenet_util.py output
            if len(tc_boxes) > 0:
                tc_bbox_prompts.append(tc_boxes.unsqueeze(1))
            else:
                tc_bbox_prompts.append(None)
                
            if len(ar_boxes) > 0:
                ar_bbox_prompts.append(ar_boxes.unsqueeze(1))
            else:
                ar_bbox_prompts.append(None)

        return {
            'ar_bbox_prompts': ar_bbox_prompts,
            'tc_bbox_prompts': tc_bbox_prompts
        }