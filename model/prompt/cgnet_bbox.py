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

    def train(self, dataloader, epochs):
        best_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            self.cgnet_model.train()
            print(f'Epoch {epoch}:')
            epoch_loader = tqdm(dataloader)
            
            epoch_loss_sum = 0.0
            epoch_loss_count = 0
            
            epoch_conf_losses = []
            epoch_box_losses = []
            epoch_cls_losses = []

            for batch in epoch_loader:
                features = batch['cgnet_input'].to(device=self.device, dtype=torch.float32)
                
                # Get dynamic image dimensions from the feature tensor
                img_h, img_w = features.shape[2], features.shape[3]

                # Parse the raw absolute bounding boxes into normalized target lists
                raw_ar_boxes = batch['ar_bbox_prompts']
                raw_tc_boxes = batch['tc_bbox_prompts']
                
                bboxes = parse_climatenet_bboxes(raw_ar_boxes, raw_tc_boxes, img_h, img_w)

                outputs = self.cgnet_model(features)
                grid_shape = outputs.shape[2:] 

                targets = build_grid_targets(bboxes, grid_shape, self.device)
                
                loss, (l_conf, l_box, l_cls) = yolo_detection_loss(outputs, targets)

                epoch_loader.set_description(f'Total Loss: {loss.item():.4f}')
                
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad() 

                epoch_loss_sum += float(loss.item())
                epoch_loss_count += 1
                epoch_conf_losses.append(l_conf)
                epoch_box_losses.append(l_box)
                epoch_cls_losses.append(l_cls)

            avg_epoch_loss = epoch_loss_sum / epoch_loss_count if epoch_loss_count > 0 else 0.0

            print(f"Epoch stats: Avg Loss {avg_epoch_loss:.4f} (Conf: {np.mean(epoch_conf_losses):.4f}, Box: {np.mean(epoch_box_losses):.4f}, Cls: {np.mean(epoch_cls_losses):.4f})")

            if avg_epoch_loss < best_loss and epoch > 10:
                best_loss = avg_epoch_loss
                self.save_model()
                print(f"New best model saved with Loss: {best_loss:.4f}")
                if self.wandb:
                    try:
                        save_path = os.path.join(self.exp_dir, f"cgnet_bbox_weight.pth")
                        wandb.save(save_path)
                    except Exception as e:
                        print(f"WandB save failed: {e}")

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
            
    def save_model(self):
        save_path = os.path.join(self.exp_dir, f"cgnet_bbox_weight.pth")
        torch.save(self.cgnet_model.state_dict(), save_path)