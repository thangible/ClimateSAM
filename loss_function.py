import torch
import torch.nn.functional as F
from typing import List, Dict, Optional
from train_util import calculate_focal_loss, calculate_dice_loss


class ClimateLoss:
    """Loss computation class for Climate SAM model"""
    
    def __init__(self, device: torch.device, theta_tc: float = 5.0, theta_total: float = 10.0):
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
        dice_loss_list_ar, focal_loss_list_ar = self._compute_mask_losses(
            ar_masks, ar_masks_gt, 
            worker_args.gamma_ar, worker_args.alpha_ar
        )
        
        dice_loss_list_tc, focal_loss_list_tc = self._compute_mask_losses(
            tc_masks, tc_masks_gt,
            worker_args.gamma_tc, worker_args.alpha_tc
        )
        
        # Aggregate losses
        return self._aggregate_losses(
            dice_loss_list_ar, focal_loss_list_ar,
            dice_loss_list_tc, focal_loss_list_tc
        )
    
    def _compute_mask_losses(
        self, 
        pred_masks: List[torch.Tensor], 
        gt_masks: List[torch.Tensor],
        gamma: float, 
        alpha: float
    ) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Compute BCE and focal losses for a set of masks"""
        
        dice_losses = []
        focal_losses = []
        
        for i in range(len(gt_masks)):
            if gt_masks[i] is not None:
                pred, label = pred_masks[i], gt_masks[i]
                
                # Binarize ground truth
                label = torch.where(torch.gt(label, 0.), 1., 0.)
                
                # Dice loss
                dice_loss = calculate_dice_loss(pred, label)
                
                # Focal loss
                focal_loss = calculate_focal_loss(pred, label, gamma=gamma, alpha=alpha)
                
                dice_losses.append(dice_loss)
                focal_losses.append(focal_loss)
        
        return dice_losses, focal_losses
    
    def _aggregate_losses(
        self,
        dice_loss_list_ar: List[torch.Tensor],
        focal_loss_list_ar: List[torch.Tensor], 
        dice_loss_list_tc: List[torch.Tensor],
        focal_loss_list_tc: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Aggregate individual losses into final loss components"""
        
        # Average Dice losses
        dice_loss_ar = self._safe_mean(dice_loss_list_ar)
        dice_loss_tc = self._safe_mean(dice_loss_list_tc) * self.theta_tc
        dice_loss = dice_loss_ar + dice_loss_tc
        
        # Average focal losses  
        focal_loss_ar = self._safe_mean(focal_loss_list_ar) * self.theta_total
        focal_loss_tc = self._safe_mean(focal_loss_list_tc) * self.theta_tc * self.theta_total
        focal_loss = focal_loss_ar + focal_loss_tc
        
        # Total losses
        total_loss_ar = dice_loss_ar + focal_loss_ar
        total_loss_tc = dice_loss_tc + focal_loss_tc  
        total_loss = dice_loss + focal_loss
        
        return {
            'total_loss': total_loss.clone().detach(),
            'total_loss_ar': total_loss_ar.clone().detach(),
            'total_loss_tc': total_loss_tc.clone().detach(),
            'dice_loss_ar': dice_loss_ar.clone().detach(),
            'dice_loss_tc': dice_loss_tc.clone().detach(),
            'focal_loss_ar': focal_loss_ar.clone().detach(),
            'focal_loss_tc': focal_loss_tc.clone().detach(),
            'dice_loss': dice_loss.clone().detach(),
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