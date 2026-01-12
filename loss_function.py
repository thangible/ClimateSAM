import torch
import torch.nn.functional as F
from typing import List, Dict, Optional
import torch.nn as nn

class ClimateLoss:
    """Loss computation class for Climate SAM model"""
    
    def __init__(self, device: torch.device, 
                 theta_tc: float = 5.0, 
                 focal_weight: float = 1.0,
                 tversky_weight: float = 3.0,
                 bce_weight: float = 1.0):
        self.device = device
        self.theta_tc = theta_tc
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        self.bce_weight = bce_weight

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
        tversky_loss_list_ar, focal_loss_list_ar, bce_loss_list_ar = self._compute_mask_losses(
            ar_masks, ar_masks_gt, 
            gamma_focal=worker_args.gamma_ar, 
            alpha_focal=worker_args.alpha_ar,
            alpha_tversky=worker_args.alpha_ar_tversky,
            beta_tversky=worker_args.beta_ar_tversky,
            bce_weight=worker_args.bce_weight_ar
        )
        
        tversky_loss_list_tc, focal_loss_list_tc, bce_loss_list_tc = self._compute_mask_losses(
            tc_masks, tc_masks_gt,
            gamma_focal=worker_args.gamma_tc,
            alpha_focal=worker_args.alpha_tc,
            alpha_tversky=worker_args.alpha_tc_tversky,
            beta_tversky=worker_args.beta_tc_tversky,
            bce_weight=worker_args.bce_weight_tc
        )
        
        # Aggregate losses
        return self._aggregate_losses(
            tversky_loss_list_ar, focal_loss_list_ar, bce_loss_list_ar,
            tversky_loss_list_tc, focal_loss_list_tc, bce_loss_list_tc
        )
    
    def _compute_mask_losses(
        self, 
        pred_masks: List[torch.Tensor], 
        gt_masks: List[torch.Tensor],
        gamma_focal: float, 
        alpha_focal: float,
        alpha_tversky: float = 0.7,
        beta_tversky: float = 0.3,
        bce_weight: int = 10
    ) -> tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Compute Tversky, focal, and BCE losses for a set of masks"""
        
        tversky_losses = []
        focal_losses = []
        bce_losses = []
        
        for i in range(len(gt_masks)):
            if gt_masks[i] is not None:
                pred, label = pred_masks[i], gt_masks[i]
                
                # Binarize ground truth
                label = torch.where(torch.gt(label, 0.), 1., 0.)
                
                # Tversky loss
                tversky_loss = calculate_tversky_loss(pred, label, alpha=alpha_tversky, beta=beta_tversky)

                # Focal loss
                focal_loss = calculate_focal_loss(pred, label, gamma=gamma_focal, alpha=alpha_focal)
                
                # BCE loss
                bce_loss = calculate_bce_loss(pred, label, weight=bce_weight)
                
                tversky_losses.append(tversky_loss)
                focal_losses.append(focal_loss)
                bce_losses.append(bce_loss)
        
        return tversky_losses, focal_losses, bce_losses
    
    def _aggregate_losses(
        self,
        tversky_loss_list_ar: List[torch.Tensor],
        focal_loss_list_ar: List[torch.Tensor],
        bce_loss_list_ar: List[torch.Tensor],
        tversky_loss_list_tc: List[torch.Tensor],
        focal_loss_list_tc: List[torch.Tensor],
        bce_loss_list_tc: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Aggregate individual losses into final loss components"""
        
        # Average Tversky losses
        tversky_loss_ar = self._safe_mean(tversky_loss_list_ar) * self.tversky_weight
        tversky_loss_tc = self._safe_mean(tversky_loss_list_tc) * self.tversky_weight * self.theta_tc 
        tversky_loss = tversky_loss_ar + tversky_loss_tc
        
        # Average focal losses  
        focal_loss_ar = self._safe_mean(focal_loss_list_ar) * self.focal_weight
        focal_loss_tc = self._safe_mean(focal_loss_list_tc) * self.theta_tc * self.focal_weight
        focal_loss = focal_loss_ar + focal_loss_tc
        
        # Average BCE losses
        bce_loss_ar = self._safe_mean(bce_loss_list_ar) * self.bce_weight
        bce_loss_tc = self._safe_mean(bce_loss_list_tc) * self.theta_tc * self.bce_weight
        bce_loss = bce_loss_ar + bce_loss_tc
        
        # Total losses
        total_loss_ar = tversky_loss_ar + focal_loss_ar + bce_loss_ar
        total_loss_tc = tversky_loss_tc + focal_loss_tc + bce_loss_tc
        total_loss = tversky_loss + focal_loss + bce_loss
        
        return {
            'total_loss': total_loss.clone().detach(),
            'total_loss_ar': total_loss_ar.clone().detach(),
            'total_loss_tc': total_loss_tc.clone().detach(),
            'tversky_loss_ar': tversky_loss_ar.clone().detach(),
            'tversky_loss_tc': tversky_loss_tc.clone().detach(),
            'focal_loss_ar': focal_loss_ar.clone().detach(),
            'focal_loss_tc': focal_loss_tc.clone().detach(),
            'bce_loss_ar': bce_loss_ar.clone().detach(),
            'bce_loss_tc': bce_loss_tc.clone().detach(),
            'tversky_loss': tversky_loss.clone().detach(),
            'focal_loss': focal_loss.clone().detach(),
            'bce_loss': bce_loss.clone().detach(),
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
    theta_tc: float = 5.0
) -> Dict[str, torch.Tensor]:
    """
    Convenience function for computing climate loss
    
    Usage:
        loss_dict = compute_climate_loss(ar_masks, tc_masks, ar_masks_gt, tc_masks_gt, device, worker_args)
        total_loss = loss_dict['total_loss_for_backward'] 
    """
    
    loss_computer = ClimateLoss(
        device, 
        theta_tc=theta_tc, 
        focal_weight=worker_args.focal_weight, 
        tversky_weight=worker_args.tversky_weight,
        bce_weight= worker_args.bce_weight
    )
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

def calculate_bce_loss(inputs: torch.Tensor, targets: torch.Tensor, weight: int):
    pos_weight = torch.tensor([weight]).to(inputs.device)
    bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=pos_weight)
    return bce_loss

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



# class GeneratorLoss:
    
#     """Loss computation class for the generator in GAN setup"""
    
#     def __init__(self, device: torch.device):
#         self.device = device

#     def compute_loss(
#         self,
#         ar_mask_pred: List[torch.Tensor],
#         tc_mask_pred: List[torch.Tensor],
#         gt_masks: List[torch.Tensor]
#     ):
#         ar_mask = [(gt_masks == 2).to(torch.uint8) for gt_masks in gt_masks]
#         tc_mask = [(gt_masks == 1).to(torch.uint8) for gt_masks in gt_masks]
#         ar_mask = torch.stack(ar_mask, dim=0).float().to(self.device)
#         tc_mask = torch.stack(tc_mask, dim=0).float().to(self.device)
        
        
        
#         # print(ar_mask[0].shape, len(ar_mask))
#         # print(f"ar_mask_pred shape: {ar_mask_pred.shape}, ar_mask shape: {ar_mask.shape}")
#         # print(f"tc_mask_pred shape: {tc_mask_pred.shape}, tc_mask shape: {tc_mask.shape}")

#         # Compute losses
#         losses = {
#             'ar_loss': F.binary_cross_entropy_with_logits(ar_mask_pred, ar_mask),
#             'tc_loss': F.binary_cross_entropy_with_logits(tc_mask_pred, tc_mask)
#         }

#         # Total loss
#         losses['total_loss'] = losses['ar_loss'] + losses['tc_loss']

#         return losses

class GeneratorLoss(nn.Module):
    """
    Computes the total loss including the final segmentation loss and weighted
    intermediate segmentation losses for multi-level supervision.
    """
    def __init__(self, device, lambda_factors: list = [0.4, 0.3, 0.2, 0.1], num_blocks: int = 4):
        super().__init__()
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        # Default weights, e.g., [0.4, 0.3, 0.2, 0.1] for 4 blocks, decaying for coarser levels
        if lambda_factors is None:
            self.lambda_factors = [0.4 / (i + 1) for i in range(num_blocks)]
        else:
            self.lambda_factors = lambda_factors
        
    def compute_loss(self, final_logit: torch.Tensor, intermediate_logits: list, gt_mask: torch.Tensor):
        """
        Calculates the total weighted loss.

        Args:
            final_logit: Logits from the highest resolution output (B, C, H, W).
            intermediate_logits: List of logits from intermediate blocks (B, C, H', W').
            gt_mask: Ground truth mask (B, H_gt, W_gt) of type long.
        """
        gt_mask = gt_mask.to(self.device).long()
        
        # 1. Final Loss (Full Resolution)
        # Assuming final_logit is already interpolated to match gt_mask's H, W
        loss_final = self.criterion(final_logit, gt_mask)
        total_loss = loss_final
        
        # 2. Intermediate Losses (Multi-Level Supervision)
        num_levels = len(intermediate_logits)
        loss_intermediate_sum = 0.0

        # We assume intermediate_logits are ordered from the lowest resolution (first block)
        for i, logit in enumerate(intermediate_logits):
            # i=0 is the coarsest level, i=num_levels-1 is the finest intermediate level
            
            # Determine target size for downsampling GT
            target_size = logit.shape[-2:]
            
            # Downsample the ground truth mask to match logit's spatial size
            # IMPORTANT: Use 'nearest' for ground truth to maintain discrete class labels
            downsampled_gt = F.interpolate(
                gt_mask.unsqueeze(1).float(), # B, 1, H_gt, W_gt
                size=target_size, 
                mode='nearest'
            ).squeeze(1).long() # B, H', W'

            # Calculate loss for this level
            loss_level = self.criterion(logit, downsampled_gt)
            
            # Apply weighting factor
            lambda_i = self.lambda_factors[i] if i < len(self.lambda_factors) else 0.1
            weighted_loss_level = lambda_i * loss_level
            
            loss_intermediate_sum += weighted_loss_level
            total_loss += weighted_loss_level

        return {
            'total_loss': total_loss,
            'final_loss': loss_final.item(),
            'intermediate_loss_sum': loss_intermediate_sum.item()}
            # Optionally log individual intermediate losses:
            
            
def compute_generator_loss(multiclass_mask, interm_masks, gt_masks, device, worker_args, 
                          lambda_factors: list = None, num_blocks: int = 4):
    """
    Compute loss for generator (auxiliary predictions) with multi-level supervision.
    
    Args:
        multiclass_mask: Final multiclass logits (B, C, H, W)
        interm_masks: List of intermediate multiclass logits from different resolution levels
        gt_masks: Ground truth masks (B, H_gt, W_gt) of type long
        device: Device to compute on
        worker_args: Training arguments (currently unused but kept for compatibility)
        lambda_factors: Weights for intermediate losses, if None will use default decaying weights
        num_blocks: Number of blocks for default lambda calculation
    
    Returns:
        Dictionary containing loss components
    """
    # Handle lambda factors like in GeneratorLoss class
    if lambda_factors is None:
        lambda_factors = [0.4 / (i + 1) for i in range(num_blocks)]
    
    # Move ground truth to device and ensure correct type
    gt_masks = gt_masks.to(device).long()
    
    # Initialize CrossEntropy criterion
    criterion = nn.CrossEntropyLoss()
    
    # 1. Final Loss (Full Resolution)
    # Assuming multiclass_mask is already interpolated to match gt_masks' H, W
    loss_final = criterion(multiclass_mask, gt_masks)
    total_loss = loss_final
    
    # 2. Intermediate Losses (Multi-Level Supervision)
    loss_intermediate_sum = 0.0
    
    if interm_masks is not None and len(interm_masks) > 0:
        # We assume interm_masks are ordered from the lowest resolution (first block)
        for i, logit in enumerate(interm_masks):
            # i=0 is the coarsest level, i=len(interm_masks)-1 is the finest intermediate level
            
            # Determine target size for downsampling GT
            target_size = logit.shape[-2:]
            
            # Downsample the ground truth mask to match logit's spatial size
            # IMPORTANT: Use 'nearest' for ground truth to maintain discrete class labels
            downsampled_gt = F.interpolate(
                gt_masks.unsqueeze(1).float(),  # B, 1, H_gt, W_gt
                size=target_size, 
                mode='nearest'
            ).squeeze(1).long()  # B, H', W'

            # Calculate loss for this level
            loss_level = criterion(logit, downsampled_gt)
            
            # Apply weighting factor
            lambda_i = lambda_factors[i] if i < len(lambda_factors) else 0.1
            weighted_loss_level = lambda_i * loss_level
            
            loss_intermediate_sum += weighted_loss_level
            total_loss += weighted_loss_level
    
    # # Convert individual class predictions for additional metrics (optional)
    # with torch.no_grad():
    #     # Convert multiclass mask to individual class masks for logging
    #     tc_gen_mask = (multiclass_mask.argmax(dim=1) == 1).float()
    #     ar_gen_mask = (multiclass_mask.argmax(dim=1) == 2).float()
        
    #     # Convert ground truth
    #     tc_gt_mask = (gt_masks == 1).float()
    #     ar_gt_mask = (gt_masks == 2).float()
        
    #     # Compute individual class BCE losses for monitoring (detached)
    #     tc_focal_loss = F.binary_cross_entropy(tc_gen_mask, tc_gt_mask, reduction='mean')
    #     ar_focal_loss = F.binary_cross_entropy(ar_gen_mask, ar_gt_mask, reduction='mean')

    return {
        'total_loss_for_backward': total_loss,
        'total_loss': total_loss.item(),
        'final_loss': loss_final.item(),
        'intermediate_loss_sum': loss_intermediate_sum.item() if isinstance(loss_intermediate_sum, torch.Tensor) else loss_intermediate_sum,
        # 'tc_focal_loss': tc_focal_loss.item(),
        # 'ar_focal_loss': ar_focal_loss.item()
    }