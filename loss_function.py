import torch
import torch.nn.functional as F
from typing import List, Dict, Optional


class ClimateLoss:
    """Loss computation class for Climate SAM model"""
    
    def __init__(self, device: torch.device, theta_tc: float = 5.0, theta_total: float = 1.0):
        self.device = device
        self.theta_tc = theta_tc
        self.theta_total = theta_total
    
    def compute_loss(
        self,
        ar_masks: List[torch.Tensor],
        tc_masks: List[torch.Tensor],
        ar_masks_gt: List[torch.Tensor],
        tc_masks_gt: List[torch.Tensor],
        worker_args
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss for AR and TC masks
        
        Args:
            ar_masks: Predicted AR masks
            tc_masks: Predicted TC masks  
            ar_masks_gt: Ground truth AR masks
            tc_masks_gt: Ground truth TC masks
            worker_args: Training arguments containing loss weights
            
        Returns:
            Dictionary containing all computed losses
        """
        
        # Compute individual losses
        tversky_loss_list_ar, focal_loss_list_ar = self._compute_mask_losses(
            ar_masks, ar_masks_gt, 
            worker_args.gamma_ar, worker_args.alpha_ar
        )
        
        tversky_loss_list_tc, focal_loss_list_tc = self._compute_mask_losses(
            tc_masks, tc_masks_gt,
            worker_args.gamma_tc, worker_args.alpha_tc
        )
        
        # Aggregate losses
        return self._aggregate_losses(
            tversky_loss_list_ar, focal_loss_list_ar,
            tversky_loss_list_tc, focal_loss_list_tc
        )
    
    def _compute_mask_losses(
        self, 
        pred_masks: List[torch.Tensor], 
        gt_masks: List[torch.Tensor],
        gamma: float, 
        alpha: float
    ) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Compute Tversky and focal losses for a set of masks"""
        
        tversky_losses = []
        focal_losses = []
        
        for i in range(len(gt_masks)):
            if gt_masks[i] is not None:
                pred, label = pred_masks[i], gt_masks[i]
                
                # Binarize ground truth
                label = torch.where(torch.gt(label, 0.), 1., 0.)
                
                # Tversky loss (replaces Dice loss)
                tversky_loss = calculate_tversky_loss(pred, label, alpha=0.7, beta=0.3)
                
                # Focal loss
                focal_loss = calculate_focal_loss(pred, label, gamma=gamma, alpha=alpha)
                
                tversky_losses.append(tversky_loss)
                focal_losses.append(focal_loss)
        
        return tversky_losses, focal_losses
    
    def _aggregate_losses(
        self,
        tversky_loss_list_ar: List[torch.Tensor],
        focal_loss_list_ar: List[torch.Tensor], 
        tversky_loss_list_tc: List[torch.Tensor],
        focal_loss_list_tc: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Aggregate individual losses into final loss components"""
        
        # Average Tversky losses
        tversky_loss_ar = self._safe_mean(tversky_loss_list_ar)
        tversky_loss_tc = self._safe_mean(tversky_loss_list_tc) * self.theta_tc
        tversky_loss = tversky_loss_ar + tversky_loss_tc
        
        # Average focal losses  
        focal_loss_ar = self._safe_mean(focal_loss_list_ar) * self.theta_total
        focal_loss_tc = self._safe_mean(focal_loss_list_tc) * self.theta_tc * self.theta_total
        focal_loss = focal_loss_ar + focal_loss_tc
        
        # Total losses
        total_loss_ar = tversky_loss_ar + focal_loss_ar
        total_loss_tc = tversky_loss_tc + focal_loss_tc  
        total_loss = tversky_loss + focal_loss
        
        return {
            'total_loss': total_loss.clone().detach(),
            'total_loss_ar': total_loss_ar.clone().detach(),
            'total_loss_tc': total_loss_tc.clone().detach(),
            'tversky_loss_ar': tversky_loss_ar.clone().detach(),
            'tversky_loss_tc': tversky_loss_tc.clone().detach(),
            'focal_loss_ar': focal_loss_ar.clone().detach(),
            'focal_loss_tc': focal_loss_tc.clone().detach(),
            'tversky_loss': tversky_loss.clone().detach(),
            'focal_loss': focal_loss.clone().detach(),
            'total_loss_for_backward': total_loss  # Keep one without detach for backprop
        }
    
    def _safe_mean(self, loss_list: List[torch.Tensor]) -> torch.Tensor:
        """Safely compute mean of loss list, return zero tensor if empty"""
        if len(loss_list) > 0:
            return sum(loss_list) / len(loss_list)
        else:
            return torch.tensor(0.0).to(self.device)


def compute_climate_loss(
    ar_masks: List[torch.Tensor],
    tc_masks: List[torch.Tensor], 
    ar_masks_gt: List[torch.Tensor],
    tc_masks_gt: List[torch.Tensor],
    device: torch.device,
    worker_args,
    theta_tc: float = 5.0,
    theta_total: float = 10.0
) -> Dict[str, torch.Tensor]:
    """
    Convenience function for computing climate loss
    
    Usage:
        loss_dict = compute_climate_loss(ar_masks, tc_masks, ar_masks_gt, tc_masks_gt, device, worker_args)
        total_loss = loss_dict['total_loss_for_backward'] 
    """
    
    loss_computer = ClimateLoss(device, theta_tc, theta_total)
    return loss_computer.compute_loss(ar_masks, tc_masks, ar_masks_gt, tc_masks_gt, worker_args)


def calculate_focal_loss(inputs: torch.Tensor, targets: torch.Tensor, gamma: float = 5, alpha: float = 0.75):
    """
    Compute the Focal Loss for binary classification.
    
    Args:
        inputs: A float tensor of arbitrary shape. These are the raw logits.
        targets: A float tensor with the same shape as inputs.
                 Contains binary labels (0 for negative, 1 for positive).
        gamma: Focusing parameter that reduces the loss contribution from easy examples. Default is 2.0.
        alpha: Balancing parameter to balance the importance of positive/negative examples. Default is 0.25.
    
    Returns:
        A scalar focal loss value.
    """
    # Apply sigmoid to get probabilities
    p = inputs.sigmoid()
    # Compute p_t, which is p if target is 1 and (1 - p) otherwise
    p_t = p * targets + (1 - p) * (1 - targets)
    
    # Compute the alpha factor according to targets
    alpha_factor = alpha * targets + (1 - alpha) * (1 - targets)
    # Compute focal weight
    focal_weight = alpha_factor * (1 - p_t).pow(gamma)
    
    # Compute the focal loss
    loss = -focal_weight * torch.log(p_t.clamp(min=1e-8))
    return loss.mean()


def calculate_tversky_loss(inputs: torch.Tensor, targets: torch.Tensor, alpha: float = 0.9, beta: float = 0.1):
    """
    Compute the Tversky loss for binary classification.
    
    The Tversky loss is a generalization of Dice loss that allows controlling the 
    trade-off between false positives and false negatives.
    
    Args:
        inputs: A float tensor of arbitrary shape. The predictions for each example (logits).
        targets: A float tensor with the same shape as inputs. Binary classification labels.
        alpha: Weight for false positives. Higher alpha penalizes false positives more.
        beta: Weight for false negatives. Higher beta penalizes false negatives more.
        
    Note: alpha + beta should typically equal 1.0, but this is not enforced.
    Common configurations:
    - alpha=0.5, beta=0.5: Equivalent to Dice loss
    - alpha=0.7, beta=0.3: More penalty on false positives (good for your use case)
    - alpha=0.3, beta=0.7: More penalty on false negatives
    
    Returns:
        A scalar Tversky loss value.
    """
    assert inputs.size(0) == targets.size(0)
    
    # Apply sigmoid to get probabilities
    inputs = inputs.sigmoid()
    
    # Flatten tensors
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    
    # True Positives, False Positives, False Negatives
    TP = (inputs * targets).sum(dim=1)
    FP = (inputs * (1 - targets)).sum(dim=1)
    FN = ((1 - inputs) * targets).sum(dim=1)
    
    # Tversky index
    tversky_index = (TP + 1) / (TP + alpha * FP + beta * FN + 1)
    
    # Tversky loss
    tversky_loss = 1 - tversky_index
    
    return tversky_loss.mean()