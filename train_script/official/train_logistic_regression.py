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
import os
import copy
import wandb
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from functools import partial
from torch.utils.data import DataLoader
from tqdm import tqdm
from contextlib import nullcontext
from evaluator import StreamSegMetrics


from utility import batch_to_cuda, get_idle_gpu, get_idle_port, set_randomness, plot_with_projection, plot_mask_with_points_and_bbox, prompt_debug, setup_device_and_distributed, setup_optimizer_and_scheduler_for_generator, worker_init_fn
from loss_function import ClimateLoss, compute_climate_loss, compute_generator_loss
from parser_config import parse
from model.climatesam import ClimateSAM
from dataset.climatenet import ClimateDataset
from model.prompt.prompt_maker import PromptMaker


# ============================================================
# LOGISTIC REGRESSION CLASSIFIER
# ============================================================

class PixelWiseLogisticRegression:
    """
    Pixel-wise logistic regression classifier for climate data.
    Trains on image embeddings to predict TC and AR masks separately.
    """
    
    def __init__(self, embedding_dim: int = 256, device: str = 'cuda'):
        """
        Args:
            embedding_dim: Dimensionality of image embeddings (typically 256 for SAM ViT-B)
            device: Device for computations ('cuda' or 'cpu')
        """
        self.embedding_dim = embedding_dim
        self.device = device
        self.clf_tc = None
        self.clf_ar = None
        self.scaler_tc = None
        self.scaler_ar = None
        self.embedding_spatial_size = None  # Will be set during training
        
    def _resize_mask(self, mask: np.ndarray, target_size: tuple) -> np.ndarray:
        """
        Resize mask to target size using nearest neighbor interpolation.
        
        Args:
            mask: Binary mask of shape (H, W)
            target_size: Target size (H', W')
            
        Returns:
            Resized mask of shape target_size
        """
        import cv2
        if mask.shape != target_size:
            resized = cv2.resize(mask.astype(np.uint8), (target_size[1], target_size[0]), 
                                interpolation=cv2.INTER_NEAREST)
            return resized.astype(np.float32)
        return mask.astype(np.float32)
    
    def train_classifier(self, image_embeddings: np.ndarray, gt_masks: list, 
                        mask_type: str = 'tc', max_iter: int = 1000):
        """
        Train logistic regression classifier for a specific mask type (TC or AR).
        
        Args:
            image_embeddings: Shape (B, C, H, W) where C=256, H=W=64 (SAM embedding size)
            gt_masks: List of binary masks, each shape (H, W) in original resolution
            mask_type: 'tc' or 'ar' to specify which classifier to train
            max_iter: Maximum iterations for LogisticRegression
        """
        # Store embedding spatial size for inference
        if self.embedding_spatial_size is None:
            self.embedding_spatial_size = (image_embeddings.shape[2], image_embeddings.shape[3])
        
        # Reshape embeddings: (B, C, H, W) -> (B*H*W, C)
        B, C, H, W = image_embeddings.shape
        X = image_embeddings.transpose(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        
        # Prepare labels: resize masks to embedding spatial size and flatten
        y_list = []
        for mask in gt_masks:
            # Resize to embedding spatial size
            resized_mask = self._resize_mask(mask, self.embedding_spatial_size)
            y_list.append(resized_mask.flatten())
        
        y = np.concatenate(y_list)  # (B*H*W,)
        
        # Standardize features for better convergence
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train logistic regression
        clf = LogisticRegression(max_iter=max_iter, random_state=42, n_jobs=-1, solver='lbfgs')
        clf.fit(X_scaled, y)
        
        # Store classifier and scaler
        if mask_type.lower() == 'tc':
            self.clf_tc = clf
            self.scaler_tc = scaler
        elif mask_type.lower() == 'ar':
            self.clf_ar = clf
            self.scaler_ar = scaler
        else:
            raise ValueError(f"mask_type must be 'tc' or 'ar', got {mask_type}")
        
        print(f"Trained {mask_type.upper()} classifier with {clf.coef_.shape[1]} features")
    
    def predict_mask(self, image_embeddings: np.ndarray, mask_type: str = 'tc') -> np.ndarray:
        """
        Generate mask predictions from embeddings.
        
        Args:
            image_embeddings: Shape (C, H, W) for single image
            mask_type: 'tc' or 'ar'
            
        Returns:
            Predicted mask of shape (H, W) where values are [0, 1]
        """
        clf = self.clf_tc if mask_type.lower() == 'tc' else self.clf_ar
        scaler = self.scaler_tc if mask_type.lower() == 'tc' else self.scaler_ar
        
        if clf is None:
            raise RuntimeError(f"Classifier for {mask_type} not trained yet")
        
        # Reshape for prediction: (C, H, W) -> (H*W, C)
        C, H, W = image_embeddings.shape
        X = image_embeddings.transpose(1, 2, 0).reshape(-1, C)  # (H*W, C)
        
        # Scale using fitted scaler
        X_scaled = scaler.transform(X)
        
        # Predict and get probability of positive class
        preds_proba = clf.predict_proba(X_scaled)[:, 1]  # (H*W,)
        
        # Reshape back to spatial dimensions
        predicted_mask = preds_proba.reshape(H, W).astype(np.float32)
        
        return predicted_mask
    
    def save_models(self, save_dir: str):
        """Save classifiers and scalers to disk."""
        os.makedirs(save_dir, exist_ok=True)
        
        if self.clf_tc is not None:
            joblib.dump(self.clf_tc, os.path.join(save_dir, 'clf_tc.pkl'))
            joblib.dump(self.scaler_tc, os.path.join(save_dir, 'scaler_tc.pkl'))
        
        if self.clf_ar is not None:
            joblib.dump(self.clf_ar, os.path.join(save_dir, 'clf_ar.pkl'))
            joblib.dump(self.scaler_ar, os.path.join(save_dir, 'scaler_ar.pkl'))
    
    def load_models(self, save_dir: str):
        """Load classifiers and scalers from disk."""
        if os.path.exists(os.path.join(save_dir, 'clf_tc.pkl')):
            self.clf_tc = joblib.load(os.path.join(save_dir, 'clf_tc.pkl'))
            self.scaler_tc = joblib.load(os.path.join(save_dir, 'scaler_tc.pkl'))
        
        if os.path.exists(os.path.join(save_dir, 'clf_ar.pkl')):
            self.clf_ar = joblib.load(os.path.join(save_dir, 'clf_ar.pkl'))
            self.scaler_ar = joblib.load(os.path.join(save_dir, 'scaler_ar.pkl'))


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(epoch, train_dataloader, climatesam, lr_classifier, prompt_maker, device, 
                   local_rank, worker_args, max_epoch_num):
    """
    Train logistic regression classifier on a single epoch of data.
    No backpropagation - just accumulating data for batch training.
    """
    climatesam.eval()
    
    # Create progress bar if you're the main process
    batch_pbar = None
    if local_rank == 0:
        batch_pbar = tqdm(total=len(train_dataloader), desc=f'Epoch {epoch}/{max_epoch_num} - Collecting', 
                         position=0, leave=True)
    
    # Accumulate embeddings and masks for training
    all_embeddings_tc = []
    all_masks_tc = []
    all_embeddings_ar = []
    all_masks_ar = []
    
    with torch.no_grad():
        for train_step, batch in enumerate(train_dataloader):
            batch = batch_to_cuda(batch, device)
            
            # Encode images to get embeddings
            image_embeddings, interm_features, image_input, ori_img_size = climatesam.encode_images(batch['input'])
            
            # Convert to numpy for sklearn
            image_embeddings_np = image_embeddings.cpu().numpy()  # (B, 256, 64, 64)
            
            # Process ground truth masks
            gt_masks = batch['gt_mask']
            
            for b_idx in range(len(gt_masks)):
                mask = gt_masks[b_idx].cpu().numpy()  # Original resolution
                
                # Extract TC mask (label == 1)
                tc_mask = (mask == 1).astype(np.float32)
                all_embeddings_tc.append(image_embeddings_np[b_idx])
                all_masks_tc.append(tc_mask)
                
                # Extract AR mask (label == 2)
                ar_mask = (mask == 2).astype(np.float32)
                all_embeddings_ar.append(image_embeddings_np[b_idx])
                all_masks_ar.append(ar_mask)
            
            if batch_pbar:
                batch_pbar.update(1)
                batch_pbar.set_postfix({
                    'epoch': f"{epoch}/{max_epoch_num}",
                    'batch': f"{train_step + 1}/{len(train_dataloader)}",
                    'samples': len(all_masks_tc)
                })
    
    if batch_pbar:
        batch_pbar.close()
    
    # Train classifiers on accumulated data
    if len(all_embeddings_tc) > 0:
        embeddings_tc_np = np.stack(all_embeddings_tc, axis=0)
        print(f"Training TC classifier with {len(all_masks_tc)} samples...")
        lr_classifier.train_classifier(embeddings_tc_np, all_masks_tc, mask_type='tc', 
                                      max_iter=1000)
    
    if len(all_embeddings_ar) > 0:
        embeddings_ar_np = np.stack(all_embeddings_ar, axis=0)
        print(f"Training AR classifier with {len(all_masks_ar)} samples...")
        lr_classifier.train_classifier(embeddings_ar_np, all_masks_ar, mask_type='ar',
                                      max_iter=1000)
    
    print(f"Epoch {epoch}: Trained classifiers successfully")


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate_one_epoch(epoch, val_dataloader, ar_metrics, tc_metrics, climatesam, 
                       lr_classifier, prompt_maker, device, max_epoch_num, worker_args):
    """Validate using logistic regression predictions."""
    climatesam.eval()
    
    # Metrics for final predictions
    ar_metrics_final = StreamSegMetrics(class_names=['Background', 'Foreground'])
    tc_metrics_final = StreamSegMetrics(class_names=['Background', 'Foreground'])
    
    valid_pbar = tqdm(total=len(val_dataloader), desc='valid', leave=False)
    
    for val_step, batch in enumerate(val_dataloader):
        batch = batch_to_cuda(batch, device)
        
        with torch.no_grad():
            # Encode images
            image_embeddings, interm_features, image_input, ori_img_size = climatesam.encode_images(batch['input'])
            image_embeddings_np = image_embeddings.cpu().numpy()
            
            # Get ground truth
            gt_masks = batch['gt_mask']
            masks_ar_gts = [(mask == 2).to(torch.uint8) for mask in gt_masks]
            masks_tc_gts = [(mask == 1).to(torch.uint8) for mask in gt_masks]
            gt_masks_tensor = torch.stack(gt_masks, dim=0).to(device)
            
            # Generate predictions from LR classifiers
            ar_pred_masks = []
            tc_pred_masks = []
            
            for b_idx in range(image_embeddings_np.shape[0]):
                # Predict TC mask
                tc_pred = lr_classifier.predict_mask(image_embeddings_np[b_idx], mask_type='tc')
                tc_pred_tensor = torch.from_numpy(tc_pred).to(device).unsqueeze(0).unsqueeze(0)
                # Resize to original image size
                tc_pred_tensor = F.interpolate(tc_pred_tensor, size=ori_img_size[0], 
                                              mode='bilinear', align_corners=False)
                tc_pred_masks.append(tc_pred_tensor)
                
                # Predict AR mask
                ar_pred = lr_classifier.predict_mask(image_embeddings_np[b_idx], mask_type='ar')
                ar_pred_tensor = torch.from_numpy(ar_pred).to(device).unsqueeze(0).unsqueeze(0)
                ar_pred_tensor = F.interpolate(ar_pred_tensor, size=ori_img_size[0],
                                              mode='bilinear', align_corners=False)
                ar_pred_masks.append(ar_pred_tensor)
            
            # Prepare prompts for full model inference
            # Create binary multiclass mask from LR predictions for prompts
            multiclass_pred = torch.zeros_like(gt_masks_tensor)
            for b_idx in range(len(tc_pred_masks)):
                tc_binary = (tc_pred_masks[b_idx].squeeze() > 0.5).long()
                ar_binary = (ar_pred_masks[b_idx].squeeze() > 0.5).long()
                multiclass_pred[b_idx] = tc_binary * 1 + ar_binary * 2
            
            prompt_dict = prompt_maker.make_prompts(multiclass_pred)
            prompt_dict = batch_to_cuda(prompt_dict, device)
            
            # Forward through full model with generated prompts
            tc_pred_masks_full, ar_pred_masks_full, _ = climatesam.forward(
                image_input=image_input,
                image_embeddings=image_embeddings,
                interm_embeddings=interm_features,
                ori_img_size=ori_img_size,
                ar_point_prompts=prompt_dict['ar_point_prompts'],
                tc_point_prompts=prompt_dict['tc_point_prompts'],
                ar_bbox_prompts=prompt_dict['ar_bbox_prompts'],
                tc_bbox_prompts=prompt_dict['tc_bbox_prompts'],
                ar_mask_prompts=prompt_dict['ar_mask_prompts'],
                tc_mask_prompts=prompt_dict['tc_mask_prompts']
            )
            
            # Convert to list format for metrics
            ar_masks = [mask for mask in ar_pred_masks_full]
            tc_masks = [mask for mask in tc_pred_masks_full]
            
            # Ensure masks are in correct shape for metrics
            for masks in [masks_ar_gts, masks_tc_gts, ar_masks, tc_masks]:
                for i in range(len(masks)):
                    if len(masks[i].shape) == 2:
                        masks[i] = masks[i][None, None, :]
                    if len(masks[i].shape) == 3:
                        masks[i] = masks[i][:, None, :]
        
        # Visualization for first validation step
        if val_step == 0 and worker_args.wandb:
            wandb_images = {}
            
            ar_point_prompts_copy = copy.deepcopy(prompt_dict['ar_point_prompts'])
            tc_point_prompts_copy = copy.deepcopy(prompt_dict['tc_point_prompts'])
            ar_bbox_prompts_copy = copy.deepcopy(prompt_dict['ar_bbox_prompts'])
            tc_bbox_prompts_copy = copy.deepcopy(prompt_dict['tc_bbox_prompts'])
            
            for i in range(min(len(gt_masks), 2)):  # Limit visualization to first 2 images
                mask = gt_masks[i].cpu().numpy()
                ar_points = ar_point_prompts_copy[i] if i < len(ar_point_prompts_copy) else None
                tc_points = tc_point_prompts_copy[i] if i < len(tc_point_prompts_copy) else None
                ar_bbox = ar_bbox_prompts_copy[i] if i < len(ar_bbox_prompts_copy) else None
                tc_bbox = tc_bbox_prompts_copy[i] if i < len(tc_bbox_prompts_copy) else None
                tc_pred_mask = tc_masks[i] if i < len(tc_masks) else None
                ar_pred_mask = ar_masks[i] if i < len(ar_masks) else None
                
                save_path = os.path.join(worker_args.exp_dir, worker_args.run_name, 'images', 
                                        f"epoch_{epoch}_step_{val_step}_image_{i}.png")
                fig = plot_mask_with_points_and_bbox(
                    mask, ar_points, tc_points, ar_bbox, tc_bbox,
                    tc_pred_mask, ar_pred_mask, radius=8, save_path=save_path, 
                    axis=True, title=f"Epoch {epoch} - LR Prediction {i}"
                )
                
                wandb_images[f"valid/val_step_{val_step}_pred_image_{i}"] = \
                    wandb.Image(fig, caption=f"Validation Step {val_step} LR Predictions - Image {i}")
            
            if wandb_images:
                wandb_images["epoch"] = epoch
                wandb.log(wandb_images, step=epoch)
                print(f"Epoch {epoch} - Logged {len(wandb_images)-1} validation images to W&B")
        
        # Update metrics
        tc_metrics_final.update(tc_masks, masks_tc_gts, batch['index_name'])
        ar_metrics_final.update(ar_masks, masks_ar_gts, batch['index_name'])
        
        valid_pbar.update(1)
    
    # Compute metrics
    ar_metrict_dict, _ = ar_metrics_final.compute()
    tc_metric_dict, _ = tc_metrics_final.compute()
    
    miou_ar = ar_metrict_dict['Mean Foreground IoU']
    miou_tc = tc_metric_dict['Mean Foreground IoU']
    
    # Reset metrics
    ar_metrics_final.reset()
    tc_metrics_final.reset()
    
    # Log to wandb
    if worker_args.wandb:
        wandb.log({
            "valid/miou_ar": miou_ar,
            "valid/miou_tc": miou_tc,
            "valid/mean_acc_ar": ar_metrict_dict['Mean Acc'],
            "valid/mean_acc_tc": tc_metric_dict['Mean Acc'],
            "valid/overall_acc_ar": ar_metrict_dict['Overall Acc'],
            "valid/overall_acc_tc": tc_metric_dict['Overall Acc'],
            "epoch": epoch,
        }, step=epoch)
    
    valid_pbar.close()
    return miou_tc, miou_ar


# ============================================================
# DATASET
# ============================================================

def set_up_dataset(worker_args):
    """Setup training and validation datasets."""
    dataset_dir = worker_args.data_dir
    train_bs = worker_args.train_bs
    val_bs = worker_args.val_bs
    
    train_dataset = ClimateDataset(
        data_dir=dataset_dir, train_flag=True, shot_num=worker_args.shot_num,
        augmented=worker_args.augmented, generate_prompt=True
    )
    val_dataset = ClimateDataset(
        data_dir=dataset_dir, train_flag=False, augmented=False, generate_prompt=True
    )
    
    train_collate_fn = train_dataset.collate_fn
    val_collate_fn = val_dataset.collate_fn
    
    # Debug mode
    if hasattr(worker_args, 'debugging') and worker_args.debugging:
        debug_size = getattr(worker_args, 'debug_size', 10)
        indices = list(range(min(debug_size, len(train_dataset))))
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        print(f"Debug mode: Using only {len(train_dataset)} training samples")

        debug_val_size = getattr(worker_args, 'debug_val_size', 5)
        val_indices = list(range(min(debug_val_size, len(val_dataset))))
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
        print(f"Debug mode: Using only {len(val_dataset)} validation samples")
        
        worker_args.max_epoch_num = 2
        worker_args.valid_per_epochs = 2
    
    train_workers, val_workers = 4, 2
    
    g = torch.Generator()
    g.manual_seed(3407)
    
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=train_bs, shuffle=True, num_workers=train_workers,
        drop_last=False, collate_fn=train_collate_fn,
        worker_init_fn=partial(worker_init_fn, base_seed=3407), generator=g
    )
    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=val_bs, shuffle=False, num_workers=val_workers,
        drop_last=False, collate_fn=val_collate_fn, 
        worker_init_fn=partial(worker_init_fn, base_seed=3407), generator=g
    )
    
    return train_dataloader, val_dataloader


# ============================================================
# MODELS
# ============================================================

def set_up_model(worker_args, device):
    """Setup ClimateSAM model (frozen) and logistic regression classifier."""
    climatesam = ClimateSAM(
        model_type=worker_args.sam_type,
        mlp_ratio=worker_args.image_encoder_mlp_ratio,
        enable_wandb_logging=getattr(worker_args, 'debugging', False)
    ).to(device)
    
    # Load pretrained weights
    image_encoder_path = os.path.join(worker_args.exp_dir, f"{worker_args.encoder_weights_name}.pth")
    if not os.path.exists(image_encoder_path):
        raise FileNotFoundError(f"Pretrained weights not found at {image_encoder_path}")
    
    phase_1_checkpoint = torch.load(image_encoder_path, map_location=device)
    print(f"Pretrained weights from phase 1 loaded from {image_encoder_path}")
    
    if 'image_encoder' in phase_1_checkpoint:
        climatesam.image_encoder.load_state_dict(phase_1_checkpoint['image_encoder'])
        print(f"Image encoder weights loaded")
    
    if 'mask_decoder' in phase_1_checkpoint:
        climatesam.mask_decoder.load_state_dict(phase_1_checkpoint['mask_decoder'])
        print(f"Mask decoder weights loaded")
    
    if 'input_adapter' in phase_1_checkpoint:
        climatesam.input_adapter.load_state_dict(phase_1_checkpoint['input_adapter'])
        print(f"Input adapter weights loaded")
    
    # Freeze ClimateSAM
    for params in climatesam.parameters():
        params.requires_grad = False
    
    # Initialize logistic regression classifier
    lr_classifier = PixelWiseLogisticRegression(embedding_dim=256, device=str(device))
    
    return climatesam, lr_classifier


# ============================================================
# MAIN WORKER
# ============================================================

def main_worker(worker_id, worker_args):
    """Main training loop."""
    set_randomness()
    max_epoch_num = worker_args.max_epoch_num
    train_dataloader, val_dataloader = set_up_dataset(worker_args)

    if isinstance(worker_id, str):
        worker_id = int(worker_id)
    device, local_rank = setup_device_and_distributed(worker_id, worker_args)
    print(f"Worker {worker_id} initialized on device {device} with local_rank {local_rank}.")

    climatesam, lr_classifier = set_up_model(worker_args, device)
    
    prompt_maker = PromptMaker(
        prompt_type='point',
        positive_point_num=worker_args.positive_point_num,
        negative_point_num=worker_args.negative_point_num
    )
    
    best_miou_tc = 0
    best_miou_ar = 0
    ar_metrics = StreamSegMetrics(class_names=['Background', 'Foreground'])
    tc_metrics = StreamSegMetrics(class_names=['Background', 'Foreground'])
    
    print(f"Validation will be performed every {worker_args.valid_per_epochs} epochs.")
    
    # Print model info
    print("Model Info:")
    total_params = sum(p.numel() for p in climatesam.parameters())
    trainable_params = sum(p.numel() for p in climatesam.parameters() if p.requires_grad)
    print(f"ClimateSAM - Total: {total_params:,}, Trainable: {trainable_params:,}")
    print(f"Logistic Regression - Non-parametric sklearn model\n")
    
    # Training loop
    for epoch in range(1, max_epoch_num + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{max_epoch_num}")
        print(f"{'='*60}")
        
        # Training phase - collect data and train LR classifier
        train_one_epoch(
            epoch=epoch, train_dataloader=train_dataloader, climatesam=climatesam,
            lr_classifier=lr_classifier, prompt_maker=prompt_maker, device=device,
            local_rank=local_rank, worker_args=worker_args, max_epoch_num=max_epoch_num
        )
        
        # Validation phase
        if epoch % worker_args.valid_per_epochs == 0 or epoch == max_epoch_num:
            miou_tc, miou_ar = validate_one_epoch(
                epoch=epoch, val_dataloader=val_dataloader, ar_metrics=ar_metrics,
                tc_metrics=tc_metrics, climatesam=climatesam, lr_classifier=lr_classifier,
                prompt_maker=prompt_maker, device=device, max_epoch_num=max_epoch_num,
                worker_args=worker_args
            )
            
            print(f"Epoch {epoch} - mIoU TC: {miou_tc:.2%}, mIoU AR: {miou_ar:.2%}")
            
            if miou_tc > best_miou_tc:
                best_miou_tc = miou_tc
                print(f'Best mIoU TC updated to {best_miou_tc:.2%}!')
            
            if miou_ar > best_miou_ar:
                best_miou_ar = miou_ar
                print(f'Best mIoU AR updated to {best_miou_ar:.2%}!')
            
            # Save best model
            if worker_args.save_model and epoch > 4:
                best_weights_dir = os.path.join(worker_args.exp_dir, 'best_weights_lr')
                os.makedirs(best_weights_dir, exist_ok=True)
                
                lr_classifier.save_models(best_weights_dir)
                print(f"Logistic regression models saved to {best_weights_dir}")
                
                if worker_args.wandb:
                    wandb.log({
                        'best_weights_saved': True,
                        'epoch': epoch
                    }, step=epoch)
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best mIoU TC: {best_miou_tc:.2%}")
    print(f"Best mIoU AR: {best_miou_ar:.2%}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    print("Starting logistic regression training process...")
    args = parse()
    set_randomness()
    
    if hasattr(args, 'wandb') and args.wandb:
        project_name = args.project_name if hasattr(args, 'project_name') else "climate-sam"
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

    # Launch the experiment process
    if len(args.used_gpu) == 1:
        main_worker(worker_id=0, worker_args=args)
    else:
        print(f"Multi-GPU training not implemented for logistic regression variant")
