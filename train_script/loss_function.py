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
                 bce_weight: float = 1.0,
                 smooth_label: bool = False,
                 ar_kernel_size: int = 9,
                 tc_kernel_size: int = 5,
                 ar_sigma: float = 2.0,
                 tc_sigma: float = 1.0):
        self.device = device
        self.theta_tc = theta_tc
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        self.bce_weight = bce_weight
        self.smooth_label = smooth_label
        self.ar_kernel_size = ar_kernel_size
        self.tc_kernel_size = tc_kernel_size
        self.ar_sigma = ar_sigma
        self.tc_sigma = tc_sigma

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
        
        # Compute individual losses; pass smoothing kernel and sigma for each type
        tversky_loss_list_ar, focal_loss_list_ar, bce_loss_list_ar = self._compute_mask_losses(
            ar_masks, ar_masks_gt, 
            gamma_focal=worker_args.gamma_ar, 
            alpha_focal=worker_args.alpha_ar,
            alpha_tversky=worker_args.alpha_ar_tversky,
            beta_tversky=worker_args.beta_ar_tversky,
            bce_weight=worker_args.bce_weight_ar,
            kernel_size=self.ar_kernel_size,
            sigma=self.ar_sigma
        )
        
        tversky_loss_list_tc, focal_loss_list_tc, bce_loss_list_tc = self._compute_mask_losses(
            tc_masks, tc_masks_gt,
            gamma_focal=worker_args.gamma_tc,
            alpha_focal=worker_args.alpha_tc,
            alpha_tversky=worker_args.alpha_tc_tversky,
            beta_tversky=worker_args.beta_tc_tversky,
            bce_weight=worker_args.bce_weight_tc,
            kernel_size=self.tc_kernel_size,
            sigma=self.tc_sigma
        )
        
        # Aggregate losses
        return self._aggregate_losses(
            tversky_loss_list_ar, focal_loss_list_ar, bce_loss_list_ar,
            tversky_loss_list_tc, focal_loss_list_tc, bce_loss_list_tc
        )
    
    def _smooth_label_tensor(self, label: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
        """Apply 2D Gaussian smoothing to a label tensor.

        label: expected shape (B,1,H,W) and float dtype
        Returns tensor same shape and device
        """
        if (kernel_size is None) or (kernel_size <= 1) or (sigma is None) or (sigma <= 0):
            return label

        # Ensure odd kernel size
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Build Gaussian kernel
        half = kernel_size // 2
        coords = torch.arange(-half, half + 1, device=label.device, dtype=torch.float32)
        x_grid, y_grid = torch.meshgrid(coords, coords, indexing='xy')
        kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * (sigma ** 2)))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, kernel_size, kernel_size)

        # Convolve using groups to preserve channels
        padding = kernel_size // 2
        # Use F.conv2d; label has shape (B, C=1, H, W)
        smoothed = F.conv2d(label, kernel.to(label.device), padding=padding)
        return smoothed

    def _compute_mask_losses(
        self, 
        pred_masks: List[torch.Tensor], 
        gt_masks: List[torch.Tensor],
        gamma_focal: float, 
        alpha_focal: float,
        alpha_tversky: float = 0.7,
        beta_tversky: float = 0.3,
        bce_weight: int = 10,
        kernel_size: Optional[int] = None,
        sigma: Optional[float] = None
    ) -> tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Compute Tversky, focal, and BCE losses for a set of masks with optional spatial smoothing of GT labels"""
        
        tversky_losses = []
        focal_losses = []
        bce_losses = []
        
        for i in range(len(gt_masks)):
            if gt_masks[i] is not None:
                pred, label = pred_masks[i], gt_masks[i]

                # Ensure tensors are on same device
                pred = pred.to(self.device)
                label = label.to(self.device)
                
                # Binarize ground truth
                label = torch.where(torch.gt(label, 0.), 1., 0.).float()

                # If smoothing enabled, apply gaussian smoothing to label
                if self.smooth_label and (kernel_size is not None) and (sigma is not None):
                    # Make sure label has shape (B,1,H,W) for conv
                    added_batch = False
                    added_channel = False
                    if label.dim() == 2:
                        label = label.unsqueeze(0).unsqueeze(0)  # 1,1,H,W
                        added_batch = True
                        added_channel = True
                    elif label.dim() == 3:
                        # Could be (1,H,W) or (B,H,W)
                        if label.size(0) == 1:
                            label = label.unsqueeze(1)  # 1,1,H,W
                            added_channel = True
                        else:
                            label = label.unsqueeze(1)  # B,1,H,W
                    elif label.dim() == 4:
                        # assume already B,1,H,W
                        pass

                    smoothed = self._smooth_label_tensor(label, kernel_size, sigma)

                    # Remove added dims to match pred
                    if added_batch and added_channel:
                        smoothed = smoothed.squeeze(0).squeeze(0)
                    elif added_channel and (not added_batch):
                        smoothed = smoothed.squeeze(1)
                    else:
                        smoothed = smoothed

                    label = smoothed

                # Ensure label and pred shapes match
                if pred.shape != label.shape:
                    # try to expand/squeeze where appropriate
                    try:
                        # if pred: (1,H,W) and label: (H,W)
                        if pred.dim() == 3 and label.dim() == 2:
                            label = label.unsqueeze(0)
                        elif pred.dim() == 3 and label.dim() == 4 and label.size(0) == 1:
                            label = label.squeeze(0)
                        elif pred.dim() == 2 and label.dim() == 3 and label.size(0) == 1:
                            pred = pred.unsqueeze(0)
                    except Exception:
                        # fallback: reshape label to pred shape
                        label = label.reshape(pred.shape)

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
        bce_weight= worker_args.bce_weight,
        smooth_label=worker_args.smooth_label,
        ar_kernel_size=worker_args.ar_kernel_size,
        tc_kernel_size=worker_args.tc_kernel_size,
        ar_sigma=worker_args.ar_sigma,
        tc_sigma=worker_args.tc_sigma
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
                           lambda_factors: list = None, num_blocks: int = 4) -> Dict[str, torch.Tensor]:
    """
    Compute loss for generator (auxiliary predictions) with multi-level supervision using Tversky and BCE losses.
    
    Args:
        multiclass_mask: Final multiclass logits (B, 3, H, W) - [background, TC, AR]
        interm_masks: List of intermediate multiclass logits from different resolution levels
        gt_masks: Ground truth masks (B, H_gt, W_gt) of type long - [0=background, 1=TC, 2=AR]
        device: Device to compute on
        worker_args: Training arguments containing loss weights and parameters
        lambda_factors: Weights for intermediate losses, if None will use default decaying weights
        num_blocks: Number of blocks for default lambda calculation
    
    Returns:
        Dictionary containing loss components
    """
    # Handle lambda factors
    if lambda_factors is None:
        lambda_factors = [0.4 / (i + 1) for i in range(num_blocks)]
    
    # Move ground truth to device
    gt_masks = gt_masks.to(device).long()
    
    def compute_multiclass_losses(logits, gt_mask):
        """Compute Tversky and BCE losses for multiclass logits"""
        # Extract TC and AR logits (channels 1 and 2)
        tc_logits = logits[:, 1, :, :]  # Channel 1: TC
        ar_logits = logits[:, 2, :, :]  # Channel 2: AR
        
        # Create binary ground truth masks
        tc_gt = (gt_mask == 1).float()  # TC class
        ar_gt = (gt_mask == 2).float()  # AR class
        
        # Compute TC losses
        tc_tversky = calculate_tversky_loss(
            tc_logits, tc_gt, 
            alpha=worker_args.alpha_tc_tversky, 
            beta=worker_args.beta_tc_tversky
        )
        tc_bce = calculate_bce_loss(tc_logits, tc_gt, weight=worker_args.bce_weight_tc)
        
        # Compute AR losses  
        ar_tversky = calculate_tversky_loss(
            ar_logits, ar_gt,
            alpha=worker_args.alpha_ar_tversky,
            beta=worker_args.beta_ar_tversky
        )
        ar_bce = calculate_bce_loss(ar_logits, ar_gt, weight=worker_args.bce_weight_ar)
        
        # Apply weights and theta_tc factor
        theta_tc = getattr(worker_args, 'theta_tc', 5.0)
        tversky_weight = getattr(worker_args, 'tversky_weight', 3.0)
        bce_weight = getattr(worker_args, 'bce_weight', 1.0)
        
        weighted_tc_tversky = tc_tversky * tversky_weight * theta_tc
        weighted_ar_tversky = ar_tversky * tversky_weight
        
        weighted_tc_bce = tc_bce * bce_weight * theta_tc  
        weighted_ar_bce = ar_bce * bce_weight
        
        total_tversky = weighted_tc_tversky + weighted_ar_tversky
        total_bce = weighted_tc_bce + weighted_ar_bce
        total_loss = total_tversky + total_bce
        
        return {
            'total_loss': total_loss,
            'tversky_loss': total_tversky,
            'bce_loss': total_bce,
            'tc_tversky': weighted_tc_tversky,
            'ar_tversky': weighted_ar_tversky,
            'tc_bce': weighted_tc_bce,
            'ar_bce': weighted_ar_bce
        }
    
    # 1. Final Loss (Full Resolution)
    final_losses = compute_multiclass_losses(multiclass_mask, gt_masks)
    total_loss = final_losses['total_loss'].clone()
    
    # 2. Intermediate Losses (Multi-Level Supervision)
    loss_intermediate_sum = torch.tensor(0.0, device=device)
    intermediate_tversky_sum = torch.tensor(0.0, device=device)
    intermediate_bce_sum = torch.tensor(0.0, device=device)
    
    if interm_masks is not None and len(interm_masks) > 0:
        for i, logit in enumerate(interm_masks):
            # Determine target size for downsampling GT
            target_size = logit.shape[-2:]
            
            # Downsample the ground truth mask to match logit's spatial size
            downsampled_gt = F.interpolate(
                gt_masks.unsqueeze(1).float(),  # B, 1, H_gt, W_gt
                size=target_size, 
                mode='nearest'
            ).squeeze(1).long()  # B, H', W'

            # Calculate losses for this level
            level_losses = compute_multiclass_losses(logit, downsampled_gt)
            
            # Apply weighting factor
            lambda_i = lambda_factors[i] if i < len(lambda_factors) else 0.1
            weighted_level_loss = lambda_i * level_losses['total_loss']
            weighted_tversky = lambda_i * level_losses['tversky_loss']
            weighted_bce = lambda_i * level_losses['bce_loss']
            
            loss_intermediate_sum = loss_intermediate_sum + weighted_level_loss
            intermediate_tversky_sum = intermediate_tversky_sum + weighted_tversky
            intermediate_bce_sum = intermediate_bce_sum + weighted_bce
            total_loss = total_loss + weighted_level_loss
    
    # Return comprehensive loss dictionary
    return {
        'total_loss_for_backward': total_loss,                                    # tensor with grad for backprop
        'total_loss': total_loss.detach(),                                       # detached for logging
        'final_loss': final_losses['total_loss'].detach(),                       # detached for logging
        'final_tversky_loss': final_losses['tversky_loss'].detach(),            # detached for logging
        'final_bce_loss': final_losses['bce_loss'].detach(),                    # detached for logging
        'final_tc_tversky': final_losses['tc_tversky'].detach(),                # detached for logging
        'final_ar_tversky': final_losses['ar_tversky'].detach(),                # detached for logging
        'final_tc_bce': final_losses['tc_bce'].detach(),                        # detached for logging
        'final_ar_bce': final_losses['ar_bce'].detach(),                        # detached for logging
        'intermediate_loss_sum': loss_intermediate_sum.detach(),                 # detached for logging
        'intermediate_tversky_sum': intermediate_tversky_sum.detach(),          # detached for logging
        'intermediate_bce_sum': intermediate_bce_sum.detach()                   # detached for logging
    }


def calculate_generator_token_loss(
    multiclass_mask,
    interm_masks,
    gt_masks,
    device: torch.device,
    worker_args,
    ar_masks_pred=None,
    tc_masks_pred=None,
    ar_masks_gt=None,
    tc_masks_gt=None,
    lambda_factors: list = None,
    num_blocks: int = 4
):
    """
    Compute combined loss for the prompt generator tokens.

    This function wraps `compute_generator_loss` (multiclass + intermediate supervision)
    and optionally `compute_climate_loss` (AR/TC binary head supervision).

    Args:
        multiclass_mask: Final multiclass logits (B, C, H, W)
        interm_masks: list of intermediate multiclass logits
        gt_masks: ground-truth multiclass masks (B, H, W)
        device: device
        worker_args: training hyperparameters
        ar_masks_pred: optional list or tensor of predicted AR binary masks (per-image tensors)
        tc_masks_pred: optional list or tensor of predicted TC binary masks (per-image tensors)
        ar_masks_gt: optional list of ground-truth AR object masks (from PromptMaker)
        tc_masks_gt: optional list of ground-truth TC object masks (from PromptMaker)
        lambda_factors, num_blocks: forwarded to compute_generator_loss

    Returns:
        merged loss dictionary containing generator and (optional) model losses, plus
        a summed 'total_loss_for_backward' suitable for backprop.
    """
    # 1) Generator loss (multiclass + intermediate supervision)
    gen_loss = compute_generator_loss(
        multiclass_mask=multiclass_mask,
        interm_masks=interm_masks,
        gt_masks=gt_masks,
        device=device,
        worker_args=worker_args,
        lambda_factors=lambda_factors,
        num_blocks=num_blocks
    )

    merged = {f'gen_{k}': v for k, v in gen_loss.items()}

    # 2) Optional AR/TC binary head loss using compute_climate_loss
    model_loss = None
    if ar_masks_pred is not None and tc_masks_pred is not None and ar_masks_gt is not None and tc_masks_gt is not None:
        # Ensure predictions are lists of tensors per image
        if isinstance(ar_masks_pred, torch.Tensor):
            ar_masks_pred_list = [ar_masks_pred[i:i+1] for i in range(ar_masks_pred.shape[0])]
        else:
            ar_masks_pred_list = list(ar_masks_pred)

        if isinstance(tc_masks_pred, torch.Tensor):
            tc_masks_pred_list = [tc_masks_pred[i:i+1] for i in range(tc_masks_pred.shape[0])]
        else:
            tc_masks_pred_list = list(tc_masks_pred)

        model_loss = compute_climate_loss(
            ar_masks=ar_masks_pred_list,
            tc_masks=tc_masks_pred_list,
            ar_masks_gt=ar_masks_gt,
            tc_masks_gt=tc_masks_gt,
            device=device,
            worker_args=worker_args
        )

        merged.update({f'model_{k}': v for k, v in model_loss.items()})

    # 3) total loss for backward
    total = gen_loss['total_loss_for_backward']
    if model_loss is not None:
        total = total + model_loss['total_loss_for_backward']

    # Package combined values for logging. Keep detached total for easy logging and the backward total
    merged['total_loss_for_backward'] = total
    merged['total_loss'] = total.detach()

    return merged


def compute_centroid_heatmaps(
    gt_centroids_list,
    H,
    W,
    sigma=10,
    device=None,
    radius=None,
    amplitude=1.0,
    normalize_coords=False,
    swap_xy=False,
    clamp=True,
):
    """
    Build target gaussian heatmaps (B,1,H,W) from a list of centroid tensors (or None).

    New parameters:
    - sigma: gaussian std (pixels)
    - radius: optional pixel cutoff radius (None = no cutoff). If set, values outside radius are zeroed.
    - amplitude: peak amplitude of each gaussian
    - normalize_coords: if True and centroids are in [0,1], scale to pixel coords
    - swap_xy: if True, interpret each centroid as (y,x) instead of (x,y)
    - clamp: clamp centroid coords to [0, W-1]/[0, H-1]
    """
    if device is None:
        device = torch.device('cpu')
    y_range = torch.arange(H, device=device).float()
    x_range = torch.arange(W, device=device).float()
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')

    heatmaps = []
    for b, centroids in enumerate(gt_centroids_list):
        heatmap = torch.zeros((H, W), device=device)
        if centroids is not None and isinstance(centroids, torch.Tensor) and centroids.numel() > 0:
            centroids = centroids.to(device).float().clone()

            # auto-scale normalized coords if requested or detected
            if normalize_coords or (centroids.max() <= 1.0 and centroids.min() >= 0.0):
                centroids[:, 0] = centroids[:, 0] * (W - 1)  # x
                centroids[:, 1] = centroids[:, 1] * (H - 1)  # y

            if swap_xy:
                centroids = centroids[:, [1, 0]]

            if clamp:
                centroids[:, 0].clamp_(0, W - 1)
                centroids[:, 1].clamp_(0, H - 1)

            for i in range(centroids.shape[0]):
                cx, cy = centroids[i, 0], centroids[i, 1]  # (x, y)
                dist_sq = (grid_y - cy) ** 2 + (grid_x - cx) ** 2

                if radius is not None:
                    # mask out outside radius to limit influence / speed up
                    mask = (dist_sq <= (float(radius) ** 2)).float()
                else:
                    mask = 1.0

                peak = amplitude * torch.exp(-dist_sq / (2 * (float(sigma) ** 2)))
                peak = peak * mask
                heatmap = torch.max(heatmap, peak)
        heatmaps.append(heatmap.unsqueeze(0))
    heatmaps = torch.stack(heatmaps, dim=0).unsqueeze(1)  # B,1,H,W
    return heatmaps

def calculate_generator_token_centroid_loss(
    multiclass_mask, interm_masks, gt_masks, device, worker_args,
    ar_masks_pred, tc_masks_pred, ar_centroids, tc_centroids
):
    # 1. Keep your existing Multiclass + Multi-level Loss
    merged = compute_generator_loss(multiclass_mask, interm_masks, gt_masks, device, worker_args)
    
    # 2. Generate Heatmaps using your new function
    # Size from multiclass_mask: (B, 3, H, W) -> H, W = 768, 1152 [cite: 47]
    H, W = multiclass_mask.shape[-2:]
    ar_target_heatmap = compute_centroid_heatmaps(ar_centroids, H, W, sigma=5.0, device=device)
    tc_target_heatmap = compute_centroid_heatmaps(tc_centroids, H, W, sigma=5.0, device=device)

    # 3. Compute MSE Loss for Centroid Localization
    # ar_masks_pred and tc_masks_pred are [B, 1, H, W] logits from the PromptGenerator
    l_cent_ar = F.mse_loss(torch.sigmoid(ar_masks_pred), ar_target_heatmap)
    l_cent_tc = F.mse_loss(torch.sigmoid(tc_masks_pred), tc_target_heatmap)

    # 4. Merge (Applying weights for rare TCs) [cite: 56, 307]
    theta_tc = getattr(worker_args, 'theta_tc', 5.0)
    total_cent_loss = l_cent_ar + (theta_tc * l_cent_tc)
    
    merged['total_loss_for_backward'] += total_cent_loss
    merged['gen_centroid_ar_loss'] = l_cent_ar.detach()
    merged['gen_centroid_tc_loss'] = l_cent_tc.detach()
    merged['gen_centroid_loss'] = total_cent_loss.detach()
    
    return merged

def compute_climate_loss_unified(
    ar_masks: List[torch.Tensor],
    tc_masks: List[torch.Tensor], 
    ar_masks_gt: List[torch.Tensor],
    tc_masks_gt: List[torch.Tensor],
    device: torch.device,
    worker_args,
    theta_tc: float = 5.0
) -> Dict[str, torch.Tensor]:
    """
    Computes climate loss by unifying all predicted object masks and all ground truth
    object masks into a single semantic mask per image. This allows SAM to train on
    noisy prompts where the number of predicted masks doesn't match the number of GT objects.
    """
    
    def merge_masks(mask_tensor):
        # mask_tensor is expected to be [N, 1, H, W] or [N, H, W] for a single image
        if mask_tensor is None or mask_tensor.numel() == 0:
            return None
        mask_tensor = mask_tensor.to(device)
        # Take the union (max logit/value) over the object dimension (dim=0)
        merged, _ = torch.max(mask_tensor, dim=0, keepdim=True)
        return merged

    # Find a reference shape to handle edge cases where a prompter predicts 0 boxes
    ref_shape_pred = None
    for m in ar_masks + tc_masks:
        if m is not None and m.numel() > 0:
            ref_shape_pred = (1, 1, m.shape[-2], m.shape[-1])
            break
            
    ref_shape_gt = None
    for m in ar_masks_gt + tc_masks_gt:
        if m is not None and m.numel() > 0:
            ref_shape_gt = (1, 1, m.shape[-2], m.shape[-1]) if m.dim() >= 3 else (1, m.shape[-2], m.shape[-1])
            break

    unified_ar_preds, unified_tc_preds = [], []
    unified_ar_gts, unified_tc_gts = [], []

    batch_size = max(len(ar_masks_gt), len(tc_masks_gt))
    for i in range(batch_size):
        # If prediction is empty (0 boxes), fill with -20.0 logits (sigmoid(-20) ≈ 0)
        empty_pred = torch.full(ref_shape_pred, -20.0, device=device) if ref_shape_pred else None
        # If GT is empty, fill with 0s
        empty_gt = torch.zeros(ref_shape_gt, device=device) if ref_shape_gt else None

        # Safely extract and merge
        ar_p = merge_masks(ar_masks[i] if i < len(ar_masks) else None)
        tc_p = merge_masks(tc_masks[i] if i < len(tc_masks) else None)
        ar_g = merge_masks(ar_masks_gt[i] if i < len(ar_masks_gt) else None)
        tc_g = merge_masks(tc_masks_gt[i] if i < len(tc_masks_gt) else None)

        unified_ar_preds.append(ar_p if ar_p is not None else empty_pred)
        unified_tc_preds.append(tc_p if tc_p is not None else empty_pred)
        unified_ar_gts.append(ar_g if ar_g is not None else empty_gt)
        unified_tc_gts.append(tc_g if tc_g is not None else empty_gt)

    # Initialize the standard ClimateLoss class
    loss_computer = ClimateLoss(
        device, 
        theta_tc=theta_tc, 
        focal_weight=worker_args.focal_weight, 
        tversky_weight=worker_args.tversky_weight,
        bce_weight= worker_args.bce_weight,
        smooth_label=worker_args.smooth_label,
        ar_kernel_size=worker_args.ar_kernel_size,
        tc_kernel_size=worker_args.tc_kernel_size,
        ar_sigma=worker_args.ar_sigma,
        tc_sigma=worker_args.tc_sigma
    )
    
    # Compute loss against the unified lists
    return loss_computer.compute_loss(
        unified_ar_preds, 
        unified_tc_preds, 
        unified_ar_gts, 
        unified_tc_gts, 
        worker_args
    )