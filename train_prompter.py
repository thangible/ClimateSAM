import random
import numpy as np
import torch
import os
import torch.nn.functional as F
from functools import partial
from torch.utils.data import DataLoader
from train_util import batch_to_cuda, get_idle_gpu, get_idle_port, set_randomness, plot_with_projection, plot_mask_with_points_and_bbox, prompt_debug, worker_init_fn, setup_optimizer_and_scheduler, setup_device
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
import matplotlib.pyplot as plt



def train_one_epoch(epoch, embeddings_file_path, model, optimizer, scheduler, device, worker_args, max_epoch_num, scaler):
    """
    Train the prompt generator for one epoch using precomputed embeddings with 
    multi-level segmentation loss.
    """
    model.train()
    
    epoch_loss_dict = {}
    epoch_loss_count = 0
    
    # Use a dummy pbar if tqdm is not available or verbose is false
    if hasattr(worker_args, 'verbose') and worker_args.verbose:
        try:
            pbar = tqdm(embeddings_file_path, desc=f'Epoch {epoch}/{max_epoch_num}')
        except NameError:
            pbar = embeddings_file_path
    else:
        pbar = embeddings_file_path
    
    # Initialize loss module (adjusting num_blocks based on model config)
    num_blocks = model.num_blocks if hasattr(model, 'num_blocks') else 4
    gen_loss = GeneratorLoss(device, num_blocks=num_blocks)

    for embedding_file in pbar:
        # Load precomputed embeddings data
        embeddings = torch.load(embedding_file, map_location='cpu')
        # We only need the features and the mask
        interm_embeddings = embeddings['interm_features'] 
        gt_masks = embeddings['gt_mask'] 

        # Prepare data for device
        interm_embeddings = [e.to(device) for e in interm_embeddings]
         

        optimizer.zero_grad()
        
        # Forward pass through prompt generator
        # Model now returns the final multi-class logit mask and a list of intermediate logits
        final_logit, intermediate_logits = model(interm_embeddings)

        # Compute losses using the new loss function structure
        losses = gen_loss.compute_loss(final_logit, intermediate_logits, gt_masks)
        total_loss = losses['total_loss']
        
        # Accumulate losses for logging
        for key, value in losses.items():
            if key not in epoch_loss_dict:
                epoch_loss_dict[key] = 0
            # Values from losses dict are already native Python numbers
            epoch_loss_dict[key] += value

        # Backward pass with AMP context
        backward_context = nullcontext 
        with backward_context():
            scaler.scale(total_loss).backward()
        
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss_count += 1
        
        # Update progress bar
        if hasattr(pbar, 'set_postfix'):
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'final': f"{losses['final_loss']:.4f}",
                'interm': f"{losses['intermediate_loss_sum']:.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
    
    # Calculate average losses for the epoch
    if epoch_loss_count > 0:
        avg_epoch_losses = {key: epoch_loss_dict[key] / epoch_loss_count for key in epoch_loss_dict.keys()}
        
        # Log to wandb
        if hasattr(worker_args, 'wandb') and worker_args.wandb:
            log_dict = {f"train/{key}": avg_epoch_losses[key] for key in avg_epoch_losses.keys()}
            log_dict["epoch"] = epoch
            # Ensure scheduler has get_last_lr method
            log_dict["learning_rate"] = scheduler.get_last_lr()[0] 
            wandb.log(log_dict, step=epoch)
            
    # Clean up (Note: deletion is often unnecessary in Python but kept for user's style)
    del embeddings, interm_embeddings, gt_masks
    
    scheduler.step()
    
    
@torch.no_grad()
def log_prompter_predictions(epoch, embeddings_file_path, prompt_generator, device, worker_args, sample_limit=5):
    """
    Log AR and TC predictions from prompt generator every 5 epochs.
    Adapted to use the single multi-class output (Class 1=TC, Class 2=AR).
    
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
            
            # Generate predictions (final_logit is BxCxHxW)
            final_logit, _ = prompt_generator(interm_embeddings)
            
            # Use Softmax to get probability maps for visualization
            probs = F.softmax(final_logit, dim=1) # B, C=3, H, W
            
            # Extract AR (Class 2) and TC (Class 1) probabilities
            # Assuming B=1 for visualization simplicity
            tc_pred = probs[0, 1, :, :].cpu().numpy() # Class 1 (TC) probability map
            ar_pred = probs[0, 2, :, :].cpu().numpy() # Class 2 (AR) probability map

            # Convert GT mask to numpy for visualization
            gt_masks = torch.stack(gt_masks, dim=0).to(device) # B, H, W
            gt_mask = gt_masks.cpu().numpy() if torch.is_tensor(gt_masks) else gt_masks
            if len(gt_mask.shape) == 3:
                 gt_mask = gt_mask[0] # Take first batch item (H, W)
            
            # Create visualization
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Original image 
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
            # GT AR mask (class 2)
            gt_ar = (gt_mask == 2).astype(float)
            axes[0, 1].imshow(gt_ar, cmap='Reds', alpha=0.7)
            axes[0, 1].set_title('GT AR Mask (Class 2)')
            
            # GT TC mask (class 1)  
            gt_tc = (gt_mask == 1).astype(float)
            axes[0, 2].imshow(gt_tc, cmap='Blues', alpha=0.7)
            axes[0, 2].set_title('GT TC Mask (Class 1)')
            
            # Predicted masks (using probability maps)
            axes[1, 0].imshow(ar_pred, cmap='Reds', alpha=0.7, vmin=0, vmax=1)
            axes[1, 0].set_title(f'Pred AR Prob (max: {ar_pred.max():.3f})')
            
            axes[1, 1].imshow(tc_pred, cmap='Blues', alpha=0.7, vmin=0, vmax=1)
            axes[1, 1].set_title(f'Pred TC Prob (max: {tc_pred.max():.3f})')
            
            # Combined overlay
            combined = np.zeros((*ar_pred.shape, 3))
            combined[..., 0] = ar_pred  # Red channel for AR
            combined[..., 2] = tc_pred  # Blue channel for TC
            axes[1, 2].imshow(combined, alpha=0.7)
            axes[1, 2].set_title('Combined Prediction Prob')
            
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
        wandb.log(wandb_images, step=epoch)
        print(f"Epoch {epoch} - Logged {len(wandb_images)} prompter prediction samples to W&B")
    
    prompt_generator.train()

@torch.no_grad()
def validate_prompter(epoch, val_source, prompt_generator, device, worker_args, sample_limit=None):
    """
    Validate prompt generator every 5 epochs.

    Accepts either:
      - a DataLoader yielding batches with keys:
            'input', 'ar_point_prompts', 'tc_point_prompts',
            'ar_bbox_prompts', 'tc_bbox_prompts', 'gt_mask', 'index_name'
      - OR the previous list-of-embedding-filepaths behavior (backward compatible).
    """
    if epoch % 5 != 0:
        return None, None

    prompt_generator.eval()

    ar_metrics = StreamSegMetrics(class_names=['Background', 'Foreground'])
    tc_metrics = StreamSegMetrics(class_names=['Background', 'Foreground'])

    # If a DataLoader is provided, run the new batched validation loop
    if isinstance(val_source, DataLoader):
        val_dataloader = val_source
        valid_pbar = tqdm(total=len(val_dataloader), desc='valid', leave=False)

        total_samples = 0
        for val_step, batch in enumerate(val_dataloader):
            batch = batch_to_cuda(batch, device)

            # Set inference images once
            _ = prompt_generator.set_infer_img(batch['input'])

            ar_point_prompts_copy = copy.deepcopy(batch['ar_point_prompts'])
            tc_point_prompts_copy = copy.deepcopy(batch['tc_point_prompts'])
            ar_bbox_prompts_copy = copy.deepcopy(batch['ar_bbox_prompts'])
            tc_bbox_prompts_copy = copy.deepcopy(batch['tc_bbox_prompts'])

            # Perform inference with prompts (model returns tc_masks, ar_masks)
            tc_masks, ar_masks = prompt_generator.infer(
                ar_point_prompts=batch['ar_point_prompts'],
                tc_point_prompts=batch['tc_point_prompts'],
                ar_bbox_prompts=batch['ar_bbox_prompts'],
                tc_bbox_prompts=batch['tc_bbox_prompts']
            )

            masks_gt = batch['gt_mask']
            # Support both tensor (B,H,W) or list of masks
            if torch.is_tensor(masks_gt):
                masks_gt_list = [masks_gt[i] for i in range(masks_gt.shape[0])]
            else:
                masks_gt_list = masks_gt

            masks_ar_gts = [(mask == 2).to(torch.uint8) for mask in masks_gt_list]
            masks_tc_gts = [(mask == 1).to(torch.uint8) for mask in masks_gt_list]

            # ensure masks have shape (B,1,H,W) when provided as list of tensors
            for masks in [masks_ar_gts, masks_tc_gts, ar_masks, tc_masks]:
                for i in range(len(masks)):
                    m = masks[i]
                    if not torch.is_tensor(m):
                        m = torch.tensor(m)
                    if len(m.shape) == 2:
                        m = m[None, None, :]
                    elif len(m.shape) == 3:
                        # (B, H, W) -> (B,1,H,W)
                        if m.shape[0] != 1:
                            m = m[:, None, :]
                        else:
                            m = m[:, None, :]
                    if len(m.shape) != 4:
                        raise RuntimeError("Mask has unexpected shape during validation.")
                    masks[i] = m

            # LOG first batch images for this epoch
            if val_step == 0:
                wandb_images = {}
                masks_gt_copy = copy.deepcopy(masks_gt_list)
                tc_masks_copy = copy.deepcopy(tc_masks)
                ar_masks_copy = copy.deepcopy(ar_masks)
                for i in range(len(masks_gt_copy)):
                    mask = masks_gt_copy[i]
                    ar_points = ar_point_prompts_copy[i]
                    tc_points = tc_point_prompts_copy[i]
                    ar_bbox = ar_bbox_prompts_copy[i]
                    tc_bbox = tc_bbox_prompts_copy[i]
                    tc_pred_mask = tc_masks_copy[i]
                    ar_pred_mask = ar_masks_copy[i]
                    save_dir = os.path.join(worker_args.exp_dir, getattr(worker_args, 'run_name', ''), 'images')
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"epoch_{epoch}_step_{val_step}_image_{i}.png")
                    fig = plot_mask_with_points_and_bbox(mask, ar_points, tc_points, ar_bbox, tc_bbox, tc_pred_mask, ar_pred_mask, radius=8, save_path=save_path, axis=True)

                    if getattr(worker_args, 'wandb', False):
                        wandb_images[f"valid/val_step_{val_step}_image_{i}"] = wandb.Image(fig, caption=f"Validation Step {val_step} Image {i}")

                if getattr(worker_args, 'wandb', False) and wandb_images:
                    wandb_images["epoch"] = epoch
                    wandb.log(wandb_images, step=epoch)

                del ar_point_prompts_copy, tc_point_prompts_copy, ar_bbox_prompts_copy, tc_bbox_prompts_copy, masks_gt_copy, tc_masks_copy, ar_masks_copy
                torch.cuda.empty_cache()

            # Update metrics (StreamSegMetrics accepts lists of tensors)
            tc_metrics.update(tc_masks, masks_tc_gts, batch['index_name'])
            ar_metrics.update(ar_masks, masks_ar_gts, batch['index_name'])

            total_samples += 1
            valid_pbar.update(1)
            valid_pbar.set_postfix_str(f"Epoch: {epoch}/{getattr(worker_args, 'max_epoch_num', '?')}.")

        valid_pbar.close()

    else:
        # Backward compatible: val_source is list of embedding file paths (original behavior)
        validation_files = val_source if sample_limit is None else val_source[:sample_limit]
        if hasattr(worker_args, 'verbose') and worker_args.verbose:
            try:
                pbar = tqdm(validation_files, desc=f'Validation Epoch {epoch}')
            except NameError:
                pbar = validation_files
        else:
            pbar = validation_files

        total_samples = 0
        for embedding_file in pbar:
            try:
                embeddings = torch.load(embedding_file, map_location='cpu')
                imgs, img_features, interm_embeddings, gt_masks, index = (
                    embeddings['imgs'],
                    embeddings['img_features'],
                    embeddings['interm_features'],
                    embeddings['gt_mask'],
                    embeddings['index_name']
                )

                interm_embeddings = [e.to(device) for e in interm_embeddings]
                gt_masks = gt_masks.to(device)

                final_logit, _ = prompt_generator(interm_embeddings)
                pred_mask = torch.argmax(final_logit, dim=1)

                tc_pred = (pred_mask == 1).long()
                ar_pred = (pred_mask == 2).long()
                tc_gt = (gt_masks == 1).long()
                ar_gt = (gt_masks == 2).long()

                def prepare_mask_for_metrics(mask):
                    if len(mask.shape) == 4:
                        mask = mask.squeeze(1)
                    return mask.cpu().numpy().astype(np.uint8)

                ar_pred_np = prepare_mask_for_metrics(ar_pred)
                tc_pred_np = prepare_mask_for_metrics(tc_pred)
                ar_gt_np = prepare_mask_for_metrics(ar_gt)
                tc_gt_np = prepare_mask_for_metrics(tc_gt)

                ar_metrics.update(ar_pred_np, ar_gt_np, [f"{index}_ar"])
                tc_metrics.update(tc_pred_np, tc_gt_np, [f"{index}_tc"])

                total_samples += 1

                if hasattr(pbar, 'set_postfix'):
                    pbar.set_postfix({
                        'samples': total_samples,
                        'file': os.path.basename(embedding_file)
                    })

                del embeddings, imgs, img_features, interm_embeddings, gt_masks
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error processing validation file {embedding_file}: {e}")
                continue

    # Compute final metrics
    ar_metric_dict, _ = ar_metrics.compute()
    tc_metric_dict, _ = tc_metrics.compute()

    miou_ar = ar_metric_dict['Mean Foreground IoU']
    mean_acc_ar = ar_metric_dict['Mean Acc']
    overall_acc_ar = ar_metric_dict['Overall Acc']
    freqw_acc_ar = ar_metric_dict.get('FreqW Acc', 0.0)
    miout_including_bg_ar = ar_metric_dict.get('Mean IoU', 0.0)

    miou_tc = tc_metric_dict['Mean Foreground IoU']
    mean_acc_tc = tc_metric_dict['Mean Acc']
    overall_acc_tc = tc_metric_dict['Overall Acc']
    freqw_acc_tc = tc_metric_dict.get('FreqW Acc', 0.0)
    miout_including_bg_tc = tc_metric_dict.get('Mean IoU', 0.0)

    # Log to wandb
    if getattr(worker_args, 'wandb', False):
        wandb.log({
            "valid_prompter/miou_ar": miou_ar,
            "valid_prompter/miou_tc": miou_tc,
            "valid_prompter/miou_combined": (miou_ar + miou_tc) / 2,
            "valid_prompter/mean_acc_ar": mean_acc_ar,
            "valid_prompter/mean_acc_tc": mean_acc_tc,
            "valid_prompter/overall_acc_ar": overall_acc_ar,
            "valid_prompter/overall_acc_tc": overall_acc_tc,
            "epoch": epoch,
        }, step=epoch)

    print(f"Epoch {epoch} Validation Results:")
    print(f"  AR  - mIoU: {miou_ar:.4f}, Mean Acc: {mean_acc_ar:.4f}, Overall Acc: {overall_acc_ar:.4f}")
    print(f"  TC  - mIoU: {miou_tc:.4f}, Mean Acc: {mean_acc_tc:.4f}, Overall Acc: {overall_acc_tc:.4f}")
    print(f"  Combined mIoU: {(miou_ar + miou_tc) / 2:.4f}")

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
    embedding_dir_path = './embeddings/train'
    embedding_valid_dir_path = './embeddings/valid'
    embeddings_file_path = [
        os.path.join(embedding_dir_path, f) 
        for f in os.listdir(embedding_dir_path) 
        if f.endswith('.pth')
    ]
    
    embeddings_valid_file_path = [
        os.path.join(embedding_valid_dir_path, f) 
        for f in os.listdir(embedding_valid_dir_path) 
        if f.endswith('.pth')
    ]
    
    
    
    if not embeddings_file_path:
        raise ValueError(f"No embedding files found in {embedding_dir_path}")
    
    print(f"Found {len(embeddings_file_path)} embedding files.")
    

    train_embeddings = embeddings_file_path
    val_embeddings = embeddings_valid_file_path  # Use all for validation if no split
    
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
    
    in_channels = {
        'vit_b': 512,
        'vit_l': 1024,
        'vit_h': 1280
    }
    # Get model type from args or default to vit_l
    model_type = getattr(worker_args, 'sam_type', 'vit_l')
    
    # Initialize prompt generator
    prompt_generator = PromptGenerator(
        in_channels=in_channels[model_type],
        fused_channels=worker_args.fuse_channels,
        num_features=num_features_map[model_type],
        features_per_block=features_per_block[model_type]
    ).to(device)
    
    # Setup optimizer and scheduler
    optimizer, scheduler = setup_optimizer_and_scheduler(model=None, prompter=prompt_generator, worker_args=worker_args)
    
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
                        save_path = os.path.join(worker_args.exp_dir, f"generator_fuse_channels_{worker_args.fuse_channels}_sam_type_{worker_args.sam_type}.pth")
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

if __name__ == '__main__':
    print("Starting prompt generator training process...")
    args = parse()
    
    # Set default values for prompt generator training
    if not hasattr(args, 'max_epoch_num'):
        args.max_epoch_num = 100
    if not hasattr(args, 'lr'):
        args.lr = 1e-4
    if not hasattr(args, 'weight_decay'):
        args.weight_decay = 1e-4
    if not hasattr(args, 'verbose'):
        args.verbose = True
    
    # Initialize wandb if enabled
    if hasattr(args, 'wandb') and args.wandb:
        project_name = getattr(args, 'project_name', "climate-sam-prompt-generator")
        run_name = getattr(args, 'run_name', None)
        wandb.init(project=project_name, name=run_name, config=vars(args))

    # Setup GPU
    if torch.cuda.is_available():
        if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
            used_gpu = os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0]
        else:
            used_gpu = get_idle_gpu(gpu_num=1)[0]
            os.environ['CUDA_VISIBLE_DEVICES'] = str(used_gpu)
        print(f"Using GPU: {used_gpu}")
    else:
        print("Using CPU")

    # Launch training
    main_worker(worker_args=args)