import random
import numpy as np
import torch
import os
import torch.nn.functional as F
from functools import partial
from torch.utils.data import DataLoader
from train_util import batch_to_cuda, get_idle_gpu, get_idle_port, set_randomness, plot_with_projection, plot_mask_with_points_and_bbox, prompt_debug
from loss_function import ClimateLoss, compute_climate_loss, GeneratorLoss
from tqdm import tqdm
from contextlib import nullcontext
from train_parser import parse
from climatesam import ClimateSAM
from model.prompt_generator import PromptGenerator
from dataset.climatenet import ClimateDataset
from evaluator import StreamSegMetrics
import copy
import wandb

def worker_init_fn(worker_id: int, base_seed: int, same_worker_seed: bool = True):
    """
    Set random seed for each worker in DataLoader to ensure the reproducibility.
    """
    seed = base_seed if same_worker_seed else base_seed + worker_id

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def setup_optimizer_and_scheduler(model, worker_args):
    """
    Sets up optimizer and scheduler for the prompt generator.
    """
    lr = getattr(worker_args, 'lr', 1e-4)
    weight_decay = getattr(worker_args, 'weight_decay', 1e-4)

    all_trainable_params = list(p for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(
        params=all_trainable_params, lr=lr, weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=worker_args.max_epoch_num, eta_min=1e-5
    )
    return optimizer, scheduler

def setup_device():
    """
    Setup device for training (single GPU only).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def train_one_epoch(epoch, embeddings_file_path, model, optimizer, scheduler, device, worker_args, max_epoch_num, scaler):
    """
    Train the prompt generator for one epoch using precomputed embeddings.
    """
    model.train()
    
    epoch_loss_dict = {}
    epoch_loss_count = 0
    
    if hasattr(worker_args, 'verbose') and worker_args.verbose:
        pbar = tqdm(embeddings_file_path, desc=f'Epoch {epoch}/{max_epoch_num}')
    else:
        pbar = embeddings_file_path
    
    for embedding_file in pbar:
        embeddings = torch.load(embedding_file, map_location='cpu')
        imgs, img_features, interm_embeddings, gt_masks, index = (
            embeddings['imgs'], 
            embeddings['img_features'], 
            embeddings['interm_features'], 
            embeddings['gt_mask'], 
            embeddings['index_name']
        )

        interm_embeddings = [e.to(device) for e in interm_embeddings]
        # Forward pass through prompt generator
        tc_masks, ar_masks = model(interm_embeddings)

        # Compute losses
        gen_loss = GeneratorLoss(device)
        losses = gen_loss.compute_loss(ar_masks, tc_masks, gt_masks)
        total_loss = losses['total_loss']
        
        # Accumulate losses for logging
        for key, value in losses.items():
            if key not in epoch_loss_dict:
                epoch_loss_dict[key] = 0
            epoch_loss_dict[key] += value.item() if torch.is_tensor(value) else value

        # Backward pass
        backward_context = nullcontext
        with backward_context():
            scaler.scale(total_loss).backward()
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        epoch_loss_count += 1
        
        # Update progress bar
        if hasattr(pbar, 'set_postfix'):
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
    
    # Calculate average losses for the epoch
    if epoch_loss_count > 0:
        avg_epoch_losses = {key: epoch_loss_dict[key] / epoch_loss_count for key in epoch_loss_dict.keys()}
        
        # Log to wandb
        if hasattr(worker_args, 'wandb') and worker_args.wandb:
            log_dict = {f"train/{key}": avg_epoch_losses[key] for key in avg_epoch_losses.keys()}
            log_dict["epoch"] = epoch
            log_dict["learning_rate"] = scheduler.get_last_lr()[0]
            wandb.log(log_dict, step=epoch)
            
    del embeddings, imgs, img_features, interm_embeddings, gt_masks, index
    
    
    scheduler.step()

@torch.no_grad()
def log_prompter_predictions(epoch, embeddings_file_path, prompt_generator, device, worker_args, sample_limit=5):
    """
    Log AR and TC predictions from prompt generator every 5 epochs.
    
    Args:
        epoch: Current epoch number
        embeddings_file_path: List of embedding file paths
        prompt_generator: The prompt generator model
        device: Training device
        worker_args: Training arguments
        sample_limit: Number of samples to log (default: 5)
    """
    if epoch % 5 != 0:
        return
        
    prompt_generator.eval()
    
    # Select a subset of embedding files for logging
    selected_files = embeddings_file_path[:sample_limit]
    
    wandb_images = {}
    
    for i, embedding_file in enumerate(selected_files):
        try:
            # Load embeddings
            embeddings = torch.load(embedding_file, map_location='cpu')
            imgs, img_features, interm_embeddings, gt_masks, index = (
                embeddings['imgs'], 
                embeddings['img_features'], 
                embeddings['interm_features'], 
                embeddings['gt_mask'], 
                embeddings['index_name']
            )
            
            # Move to device
            interm_embeddings = [e.to(device) for e in interm_embeddings]
            
            # Generate predictions
            tc_masks, ar_masks = prompt_generator(interm_embeddings)
            
            # Convert predictions to numpy for visualization
            tc_pred = torch.sigmoid(tc_masks).cpu().numpy()
            ar_pred = torch.sigmoid(ar_masks).cpu().numpy()
            gt_mask = gt_masks.cpu().numpy() if torch.is_tensor(gt_masks) else gt_masks
            
            # Create visualization
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Original image (if available in embeddings)
            if imgs is not None:
                img = imgs.cpu().numpy() if torch.is_tensor(imgs) else imgs
                if len(img.shape) == 4:
                    img = img[0]  # Take first batch item
                if len(img.shape) == 3 and img.shape[0] <= 3:
                    img = img.transpose(1, 2, 0)  # CHW to HWC
                axes[0, 0].imshow(img.squeeze() if img.shape[-1] == 1 else img)
                axes[0, 0].set_title('Input Image')
            else:
                axes[0, 0].text(0.5, 0.5, 'Image not available', ha='center', va='center')
                axes[0, 0].set_title('Input Image')
            
            # Ground truth masks
            if len(gt_mask.shape) == 4:
                gt_mask = gt_mask[0, 0]  # Take first batch and channel
            elif len(gt_mask.shape) == 3:
                gt_mask = gt_mask[0]
                
            # GT AR mask (class 2)
            gt_ar = (gt_mask == 2).astype(float)
            axes[0, 1].imshow(gt_ar, cmap='Reds', alpha=0.7)
            axes[0, 1].set_title('GT AR Mask')
            
            # GT TC mask (class 1)  
            gt_tc = (gt_mask == 1).astype(float)
            axes[0, 2].imshow(gt_tc, cmap='Blues', alpha=0.7)
            axes[0, 2].set_title('GT TC Mask')
            
            # Predicted masks
            if len(ar_pred.shape) == 4:
                ar_pred = ar_pred[0, 0]  # Take first batch and channel
            elif len(ar_pred.shape) == 3:
                ar_pred = ar_pred[0]
                
            if len(tc_pred.shape) == 4:
                tc_pred = tc_pred[0, 0]
            elif len(tc_pred.shape) == 3:
                tc_pred = tc_pred[0]
            
            axes[1, 0].imshow(ar_pred, cmap='Reds', alpha=0.7, vmin=0, vmax=1)
            axes[1, 0].set_title(f'Pred AR Mask (max: {ar_pred.max():.3f})')
            
            axes[1, 1].imshow(tc_pred, cmap='Blues', alpha=0.7, vmin=0, vmax=1)
            axes[1, 1].set_title(f'Pred TC Mask (max: {tc_pred.max():.3f})')
            
            # Combined overlay
            combined = np.zeros((*ar_pred.shape, 3))
            combined[..., 0] = ar_pred  # Red channel for AR
            combined[..., 2] = tc_pred  # Blue channel for TC
            axes[1, 2].imshow(combined, alpha=0.7)
            axes[1, 2].set_title('Combined Prediction')
            
            # Remove axes
            for ax in axes.flat:
                ax.set_xticks([])
                ax.set_yticks([])
            
            plt.tight_layout()
            
            # Save figure if needed
            if hasattr(worker_args, 'exp_dir'):
                save_dir = os.path.join(worker_args.exp_dir, 'prompter_predictions')
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"epoch_{epoch}_sample_{i}.png")
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
            # Add to wandb logging
            if hasattr(worker_args, 'wandb') and worker_args.wandb:
                wandb_images[f"prompter/epoch_{epoch}_sample_{i}"] = wandb.Image(
                    fig, 
                    caption=f"Epoch {epoch} - Sample {i} - {index}"
                )
            
            plt.close(fig)
            
            # Clean up
            del embeddings, imgs, img_features, interm_embeddings, gt_masks
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    # Log all images to wandb at once
    if wandb_images and hasattr(worker_args, 'wandb') and worker_args.wandb:
        wandb_images["epoch"] = epoch
        wandb.log(wandb_images, step=epoch)
        print(f"Epoch {epoch} - Logged {len(wandb_images)-1} prompter prediction samples to W&B")
    
    prompt_generator.train()

@torch.no_grad()
def validate_prompter(epoch, embeddings_file_path, prompt_generator, device, worker_args, sample_limit=None):
    """
    Validate prompt generator using mIoU metrics every 5 epochs.
    
    Args:
        epoch: Current epoch number
        embeddings_file_path: List of embedding file paths
        prompt_generator: The prompt generator model
        device: Training device
        worker_args: Training arguments
        sample_limit: Number of samples to evaluate (None for all)
    """
    if epoch % 5 != 0:
        return None, None
        
    prompt_generator.eval()
    
    # Initialize metrics
    ar_metrics = StreamSegMetrics(class_names=['Background', 'Foreground'])
    tc_metrics = StreamSegMetrics(class_names=['Background', 'Foreground'])
    
    # Select files for validation
    validation_files = embeddings_file_path if sample_limit is None else embeddings_file_path[:sample_limit]
    
    if hasattr(worker_args, 'verbose') and worker_args.verbose:
        pbar = tqdm(validation_files, desc=f'Validation Epoch {epoch}')
    else:
        pbar = validation_files
    
    total_samples = 0
    
    for embedding_file in pbar:
        try:
            # Load embeddings
            embeddings = torch.load(embedding_file, map_location='cpu')
            imgs, img_features, interm_embeddings, gt_masks, index = (
                embeddings['imgs'], 
                embeddings['img_features'], 
                embeddings['interm_features'], 
                embeddings['gt_mask'], 
                embeddings['index_name']
            )
            
            # Move to device
            interm_embeddings = [e.to(device) for e in interm_embeddings]
            
            # Generate predictions
            tc_masks, ar_masks = prompt_generator(interm_embeddings)
            
            # Apply sigmoid and threshold predictions
            tc_pred = torch.sigmoid(tc_masks) > 0.5
            ar_pred = torch.sigmoid(ar_masks) > 0.5
            
            # Convert to proper format for metrics
            # Ground truth masks
            if torch.is_tensor(gt_masks):
                gt_mask = gt_masks
            else:
                gt_mask = torch.tensor(gt_masks)
            
            # Create binary masks for AR (class 2) and TC (class 1)
            ar_gt = (gt_mask == 2).float()
            tc_gt = (gt_mask == 1).float()
            
            # Ensure proper dimensions for metrics
            def prepare_mask_for_metrics(mask):
                """Prepare mask for StreamSegMetrics"""
                if len(mask.shape) == 4:
                    mask = mask.squeeze(1)  # Remove channel dimension if present
                if len(mask.shape) == 3:
                    # Keep batch dimension, metrics expects [B, H, W]
                    pass
                elif len(mask.shape) == 2:
                    mask = mask.unsqueeze(0)  # Add batch dimension
                return mask.cpu().numpy().astype(np.uint8)
            
            # Prepare masks for metrics computation
            ar_pred_np = prepare_mask_for_metrics(ar_pred)
            tc_pred_np = prepare_mask_for_metrics(tc_pred)
            ar_gt_np = prepare_mask_for_metrics(ar_gt)
            tc_gt_np = prepare_mask_for_metrics(tc_gt)
            
            # Update metrics
            ar_metrics.update(ar_pred_np, ar_gt_np, [f"{index}_ar"])
            tc_metrics.update(tc_pred_np, tc_gt_np, [f"{index}_tc"])
            
            total_samples += 1
            
            # Update progress bar
            if hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({
                    'samples': total_samples,
                    'file': os.path.basename(embedding_file)
                })
            
            # Clean up
            del embeddings, imgs, img_features, interm_embeddings, gt_masks
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing validation file {embedding_file}: {e}")
            continue
    
    # Compute final metrics
    ar_metric_dict, _ = ar_metrics.compute()
    tc_metric_dict, _ = tc_metrics.compute()
    
    # Extract key metrics
    miou_ar = ar_metric_dict['Mean Foreground IoU']
    mean_acc_ar = ar_metric_dict['Mean Acc']
    overall_acc_ar = ar_metric_dict['Overall Acc']
    
    miou_tc = tc_metric_dict['Mean Foreground IoU']
    mean_acc_tc = tc_metric_dict['Mean Acc']
    overall_acc_tc = tc_metric_dict['Overall Acc']
    
    # Calculate combined metrics
    miou_combined = (miou_ar + miou_tc) / 2
    
    # Log metrics to wandb
    if hasattr(worker_args, 'wandb') and worker_args.wandb:
        wandb.log({
            "valid_prompter/miou_ar": miou_ar,
            "valid_prompter/miou_tc": miou_tc,
            "valid_prompter/miou_combined": miou_combined,
            "valid_prompter/mean_acc_ar": mean_acc_ar,
            "valid_prompter/mean_acc_tc": mean_acc_tc,
            "valid_prompter/overall_acc_ar": overall_acc_ar,
            "valid_prompter/overall_acc_tc": overall_acc_tc,
            "epoch": epoch,
        }, step=epoch)
    
    # Print results
    print(f"Epoch {epoch} Validation Results:")
    print(f"  AR  - mIoU: {miou_ar:.4f}, Mean Acc: {mean_acc_ar:.4f}, Overall Acc: {overall_acc_ar:.4f}")
    print(f"  TC  - mIoU: {miou_tc:.4f}, Mean Acc: {mean_acc_tc:.4f}, Overall Acc: {overall_acc_tc:.4f}")
    print(f"  Combined mIoU: {miou_combined:.4f}")
    
    # Reset metrics
    ar_metrics.reset()
    tc_metrics.reset()
    
    prompt_generator.train()
    
    return miou_ar, miou_tc

def main_worker(worker_args):
    """
    Main worker function for training the prompt generator.
    """
    set_randomness()
    max_epoch_num = worker_args.max_epoch_num
    device = setup_device()
    print(f"Training initialized on device {device}.")
    
    # Setup embeddings paths
    embedding_dir_path = './embeddings'
    embeddings_file_path = [
        os.path.join(embedding_dir_path, f) 
        for f in os.listdir(embedding_dir_path) 
        if f.endswith('.pth')
    ]
    
    if not embeddings_file_path:
        raise ValueError(f"No embedding files found in {embedding_dir_path}")
    
    print(f"Found {len(embeddings_file_path)} embedding files.")
    
    # Split embeddings into train/val if needed
    if hasattr(worker_args, 'val_split') and worker_args.val_split > 0:
        split_idx = int(len(embeddings_file_path) * (1 - worker_args.val_split))
        train_embeddings = embeddings_file_path[:split_idx]
        val_embeddings = embeddings_file_path[split_idx:]
        print(f"Split: {len(train_embeddings)} train, {len(val_embeddings)} validation files")
    else:
        train_embeddings = embeddings_file_path
        val_embeddings = embeddings_file_path  # Use all for validation if no split
    
    # Model configuration
    num_features_map = {
        'vit_b': 12,
        'vit_l': 24,
        'vit_h': 32
    }
    features_per_block = {
        'vit_b': 3,
        'vit_l': 6,
        'vit_h': 9
    }
    
    # Get model type from args or default to vit_l
    model_type = getattr(worker_args, 'sam_type', 'vit_l')
    
    # Initialize prompt generator
    prompt_generator = PromptGenerator(
        in_channels=1024,
        fused_channels=64,
        num_features=num_features_map[model_type],
        features_per_block=features_per_block[model_type]
    ).to(device)
    
    # Setup optimizer and scheduler
    optimizer, scheduler = setup_optimizer_and_scheduler(prompt_generator, worker_args)
    
    # Initialize scaler
    scaler = torch.amp.GradScaler('cuda')
    
    # Track best metrics
    best_miou_ar = 0
    best_miou_tc = 0
    best_miou_combined = 0
    
    print(f"Starting training for {max_epoch_num} epochs...")
    
    for epoch in range(1, max_epoch_num + 1):
        train_one_epoch(
            epoch, train_embeddings, prompt_generator, 
            optimizer, scheduler, device, worker_args, max_epoch_num, scaler
        )
        
        # Validation and logging every 5 epochs
        if epoch % 5 == 0:
            # Validate with metrics
            val_sample_limit = getattr(worker_args, 'val_sample_limit', None)
            miou_ar, miou_tc = validate_prompter(
                epoch, val_embeddings, prompt_generator, device, worker_args, val_sample_limit
            )
            
            if miou_ar is not None and miou_tc is not None:
                miou_combined = (miou_ar + miou_tc) / 2
                
                # Track best metrics
                if miou_ar > best_miou_ar:
                    best_miou_ar = miou_ar
                    print(f'Best AR mIoU updated to {best_miou_ar:.4f}!')
                
                if miou_tc > best_miou_tc:
                    best_miou_tc = miou_tc
                    print(f'Best TC mIoU updated to {best_miou_tc:.4f}!')
                
                if miou_combined > best_miou_combined:
                    best_miou_combined = miou_combined
                    print(f'Best Combined mIoU updated to {best_miou_combined:.4f}!')
                    
                    # Save best model
                    if hasattr(worker_args, 'save_model') and worker_args.save_model:
                        save_path = os.path.join(worker_args.exp_dir, f"prompt_generator_best.pth")
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': prompt_generator.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'miou_ar': miou_ar,
                            'miou_tc': miou_tc,
                            'miou_combined': miou_combined,
                        }, save_path)
                        print(f"Best model checkpoint saved to {save_path}")
                        
                        if hasattr(worker_args, 'wandb') and worker_args.wandb:
                            wandb.save(save_path)
            
            # Log prompter predictions for visualization
            log_prompter_predictions(epoch, val_embeddings, prompt_generator, device, worker_args)
        
        # Save regular checkpoint
        if hasattr(worker_args, 'save_model') and worker_args.save_model and epoch % 10 == 0:
            save_path = os.path.join(worker_args.exp_dir, f"prompt_generator_epoch_{epoch}.pth")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': prompt_generator.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, save_path)
            print(f"Model checkpoint saved to {save_path}")
            
            if hasattr(worker_args, 'wandb') and worker_args.wandb:
                wandb.save(save_path)
    
    # Final validation
    print("\nFinal validation...")
    miou_ar, miou_tc = validate_prompter(
        max_epoch_num, val_embeddings, prompt_generator, device, worker_args
    )
    
    print(f"\nTraining completed!")
    print(f"Best AR mIoU: {best_miou_ar:.4f}")
    print(f"Best TC mIoU: {best_miou_tc:.4f}")
    print(f"Best Combined mIoU: {best_miou_combined:.4f}")
