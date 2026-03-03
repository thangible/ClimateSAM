import random
import numpy as np
import torch
import os
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from functools import partial
from torch.utils.data import DataLoader
from train_util import batch_to_cuda, get_idle_gpu, get_idle_port, set_randomness,  plot_with_projection, plot_mask_with_points_and_bbox, prompt_debug
from loss_function import ClimateLoss, compute_climate_loss
from tqdm import tqdm
from contextlib import nullcontext
from train_parser import parse
# Remove ClimateSAM import and add vanilla SAM
from model.segment_anything_ext.modeling.sam import Sam
from model.segment_anything_ext.build_sam import sam_model_registry
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
    Sets up optimizer and scheduler for vanilla SAM model.
    """
    # Learning rate and weight decay with defaults
    lr = worker_args.lr if hasattr(worker_args, 'lr') else 1e-3
    weight_decay = worker_args.weight_decay if hasattr(worker_args, 'weight_decay') else 1e-4

    # Get all trainable parameters
    all_trainable_params = list(p for p in model.parameters() if p.requires_grad) 

    optimizer = torch.optim.AdamW(
        params=all_trainable_params, lr=lr, weight_decay=weight_decay
    )

    # Cosine Annealing Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=worker_args.max_epoch_num, eta_min=1e-5
    )
    return optimizer, scheduler

def setup_device_and_distributed(worker_id, worker_args):
    gpu_num = len(worker_args.used_gpu)
    world_size = os.environ['WORLD_SIZE'] if 'WORLD_SIZE' in os.environ.keys() else gpu_num
    base_rank = os.environ['RANK'] if 'RANK' in os.environ.keys() else 0
    local_rank = (base_rank * gpu_num) + worker_id
    if gpu_num > 1:
        dist.init_process_group(backend='nccl', init_method=worker_args.dist_url,
                                world_size=world_size, rank=local_rank)
    device = torch.device(f"cuda:{worker_id}")
    torch.cuda.set_device(device)
    return device, local_rank

def prepare_batch_for_vanilla_sam(batch, data_type='ar'):
    """
    Prepare batch data for vanilla SAM forward pass.
    Args:
        batch: Input batch
        data_type: 'ar' for atmospheric rivers or 'tc' for tropical cyclones
    """
    batch_size = len(batch['input'])
    batched_input = []
    
    for i in range(batch_size):
        input_dict = {
            'image': batch['input'][i],
            'original_size': batch['input'][i].shape[-2:],  # (H, W)
        }
        
        # Add prompts based on data type
        if data_type == 'ar':
            if batch['ar_point_prompts'][i] is not None:
                input_dict['point_coords'] = batch['ar_point_prompts'][i][:, :2]  # x, y coordinates
                input_dict['point_labels'] = batch['ar_point_prompts'][i][:, 2]   # labels
            if batch['ar_bbox_prompts'][i] is not None:
                input_dict['boxes'] = batch['ar_bbox_prompts'][i]
        elif data_type == 'tc':
            if batch['tc_point_prompts'][i] is not None:
                input_dict['point_coords'] = batch['tc_point_prompts'][i][:, :2]  # x, y coordinates
                input_dict['point_labels'] = batch['tc_point_prompts'][i][:, 2]   # labels
            if batch['tc_bbox_prompts'][i] is not None:
                input_dict['boxes'] = batch['tc_bbox_prompts'][i]
                
        batched_input.append(input_dict)
    
    return batched_input

def compute_vanilla_sam_loss(pred_masks, gt_masks, device):
    """
    Compute loss for vanilla SAM predictions.
    """
    total_loss = 0.0
    focal_loss = 0.0
    dice_loss = 0.0
    
    for i, (pred, gt) in enumerate(zip(pred_masks, gt_masks)):
        # Get the best mask from multimask output
        if pred.shape[1] > 1:  # Multiple masks
            # Choose mask with highest IoU score or use first mask
            pred = pred[:, 0:1, :, :]  # Use first mask
        
        # Ensure same shape
        if pred.shape != gt.shape:
            pred = F.interpolate(pred.float(), size=gt.shape[-2:], mode='bilinear', align_corners=False)
        
        # Convert to probabilities
        pred_sigmoid = torch.sigmoid(pred)
        
        # Focal loss
        alpha = 0.25
        gamma = 2.0
        ce_loss = F.binary_cross_entropy_with_logits(pred, gt.float(), reduction='none')
        p_t = pred_sigmoid * gt + (1 - pred_sigmoid) * (1 - gt)
        focal = alpha * (1 - p_t) ** gamma * ce_loss
        focal_loss += focal.mean()
        
        # Dice loss
        smooth = 1.0
        intersection = (pred_sigmoid * gt).sum()
        dice = (2.0 * intersection + smooth) / (pred_sigmoid.sum() + gt.sum() + smooth)
        dice_loss += (1 - dice)
    
    focal_loss /= len(pred_masks)
    dice_loss /= len(pred_masks)
    total_loss = focal_loss + dice_loss
    
    return {
        'total_loss': total_loss,
        'focal_loss': focal_loss,
        'dice_loss': dice_loss,
        'total_loss_for_backward': total_loss
    }

def train_one_epoch(epoch, train_dataloader, model, optimizer, scheduler, device, local_rank, worker_args, max_epoch_num, scaler):
    model.train()
    
    # Get gradient accumulation steps
    gradient_accumulation_steps = getattr(worker_args, 'gradient_accumulation_steps', 1)
    data_type = getattr(worker_args, 'data_type', 'ar')  # Default to AR
    
    # Create progress bar if main process
    if local_rank == 0:
        batch_pbar = tqdm(total=len(train_dataloader), desc=f'Epoch {epoch}/{max_epoch_num} - Batches', position=0, leave=True)
    
    # Initialize accumulated loss for logging
    epoch_loss_dict = {}
    epoch_loss_count = 0
    
    for train_step, batch in enumerate(train_dataloader):
        batch = batch_to_cuda(batch, device)
        
        with torch.amp.autocast('cuda'):
            # Prepare batch for vanilla SAM
            batched_input = prepare_batch_for_vanilla_sam(batch, data_type)
            
            # Forward pass through vanilla SAM
            outputs, _ = model(batched_input, multimask_output=True)
            
            # Extract predictions
            pred_masks = [output['masks'] for output in outputs]
            
            # Get ground truth masks based on data type
            if data_type == 'ar':
                gt_masks = batch['ar_object_masks']
            else:  # tc
                gt_masks = batch['tc_object_masks']
            
            # Compute loss
            loss_dict = compute_vanilla_sam_loss(pred_masks, gt_masks, device)
        
        total_loss = loss_dict.pop('total_loss_for_backward')
        
        # Scale loss by gradient accumulation steps
        total_loss = total_loss / gradient_accumulation_steps
        
        # Accumulate losses for epoch-level logging
        for key, value in loss_dict.items():
            if key not in epoch_loss_dict:
                epoch_loss_dict[key] = 0
            epoch_loss_dict[key] += value.item() / gradient_accumulation_steps

        backward_context = nullcontext
        if torch.distributed.is_initialized():
            # Only sync gradients on the last accumulation step
            if (train_step + 1) % gradient_accumulation_steps != 0:
                backward_context = model.no_sync
            else:
                backward_context = nullcontext

        with backward_context():
            scaler.scale(total_loss).backward()
        
        # Update batch progress bar
        if local_rank == 0:
            batch_pbar.update(1)
            batch_pbar.set_postfix({
                'epoch': f"{epoch}/{max_epoch_num}",
                'batch': f"{train_step + 1}/{len(train_dataloader)}",
                'loss': f"{total_loss.item():.4f}"
            })
        
        # Only update optimizer every gradient_accumulation_steps
        if (train_step + 1) % gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            epoch_loss_count += 1

    # Handle any remaining gradients
    if len(train_dataloader) % gradient_accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        epoch_loss_count += 1
    
    # Calculate average losses for the entire epoch
    if epoch_loss_count > 0:
        avg_epoch_losses = {key: epoch_loss_dict[key] / epoch_loss_count for key in epoch_loss_dict.keys()}
        
        # Log to wandb once per epoch
        if worker_args.wandb and local_rank == 0:
            log_dict = {f"train/{key}": avg_epoch_losses[key] for key in avg_epoch_losses.keys()}
            log_dict["epoch"] = epoch
            log_dict["learning_rate"] = scheduler.get_last_lr()[0]
            wandb.log(log_dict, step=epoch)
    
    # Close progress bar
    if local_rank == 0:
        batch_pbar.close()
            
    scheduler.step()

@torch.no_grad()
def validate_one_epoch(epoch, val_dataloader, metrics, model, device, max_epoch_num, worker_args):
    model.eval()
    valid_pbar = tqdm(total=len(val_dataloader), desc='valid', leave=False)
    
    data_type = getattr(worker_args, 'data_type', 'ar')  # Default to AR
    
    for val_step, batch in enumerate(val_dataloader):
        batch = batch_to_cuda(batch, device)
        
        # Prepare batch for vanilla SAM
        batched_input = prepare_batch_for_vanilla_sam(batch, data_type)
        
        # Forward pass
        outputs, _ = model(batched_input, multimask_output=False)
        
        # Extract predictions
        pred_masks = [output['masks'] for output in outputs]
        
        # Get ground truth masks
        masks_gt = batch['gt_mask']
        if data_type == 'ar':
            masks_gts = [(mask == 2).to(torch.uint8) for mask in masks_gt]
        else:  # tc
            masks_gts = [(mask == 1).to(torch.uint8) for mask in masks_gt]
        
        # Process masks for evaluation
        for masks in [masks_gts, pred_masks]:
            for i in range(len(masks)):
                if len(masks[i].shape) == 2:
                    masks[i] = masks[i][None, None, :]
                if len(masks[i].shape) == 3:
                    masks[i] = masks[i][:, None, :]
                if len(masks[i].shape) != 4:
                    raise RuntimeError(f"Unexpected mask shape: {masks[i].shape}")
        
        # Convert predictions to binary masks
        for i in range(len(pred_masks)):
            pred_masks[i] = (pred_masks[i] > 0.5).to(torch.uint8)
        
        # Update metrics
        metrics.update(pred_masks, masks_gts, batch['index_name'])
        valid_pbar.update(1)
        
        str_step_info = f"Epoch: {epoch}/{max_epoch_num}"
        valid_pbar.set_postfix_str(str_step_info)
    
    # Compute metrics
    metric_dict, _ = metrics.compute()
    
    miou = metric_dict['Mean Foreground IoU']
    mean_acc = metric_dict['Mean Acc']
    overall_acc = metric_dict['Overall Acc']
    freqw_acc = metric_dict['FreqW Acc']
    miou_including_bg = metric_dict['Mean IoU']
    
    metrics.reset()
    
    if worker_args.wandb:
        wandb.log({
            f"valid/miou_{data_type}": miou,
            f"valid/mean_acc_{data_type}": mean_acc,
            f"valid/overall_acc_{data_type}": overall_acc,
            f"valid/freqw_acc_{data_type}": freqw_acc,
            f"valid/miou_including_bg_{data_type}": miou_including_bg,
            "epoch": epoch,
        }, step=epoch)
        
    return miou

def main_worker(worker_id, worker_args):
    set_randomness()
    max_epoch_num = worker_args.max_epoch_num 
    if isinstance(worker_id, str):
        worker_id = int(worker_id)
    device, local_rank = setup_device_and_distributed(worker_id, worker_args)
    print(f"Worker {worker_id} initialized on device {device} with local_rank {local_rank}.")
    
    # Get data type from arguments
    data_type = getattr(worker_args, 'data_type', 'ar')
    print(f"Training on {data_type.upper()} data")
    
    # PREPARE DATASET
    dataset_dir = worker_args.data_dir
    train_dataset = ClimateDataset(
        data_dir=dataset_dir, train_flag=True, shot_num=worker_args.shot_num,
        augmented=worker_args.augmented, generate_prompt=True
    )
    val_dataset = ClimateDataset(data_dir=dataset_dir, train_flag=False, augmented=False, generate_prompt=True)

    train_collate_fn = train_dataset.collate_fn
    val_collate_fn = val_dataset.collate_fn

    if hasattr(worker_args, 'debugging') and worker_args.debugging:
        debug_size = getattr(worker_args, 'debug_size', 10)
        indices = list(range(min(debug_size, len(train_dataset))))
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        print(f"Debug mode: Using only {len(train_dataset)} training samples")

        debug_val_size = getattr(worker_args, 'debug_val_size', 5)
        val_indices = list(range(min(debug_val_size, len(val_dataset))))
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
        print(f"Debug mode: Using only {len(val_dataset)} validation samples")
        
        max_epoch_num = 2
        worker_args.valid_per_epochs = 1
        print(f"Debug mode: Setting max_epoch_num to {max_epoch_num} and valid_per_epochs to {worker_args.valid_per_epochs}")
    
    # DataLoader setup
    train_bs = worker_args.train_bs if worker_args.train_bs else (1 if worker_args.shot_num == 1 else 4)
    gradient_accumulation_steps = getattr(worker_args, 'gradient_accumulation_steps', 1)
    
    actual_train_bs = train_bs // gradient_accumulation_steps
    if actual_train_bs < 1:
        actual_train_bs = 1
        print(f"Warning: gradient_accumulation_steps ({gradient_accumulation_steps}) is larger than train_bs ({train_bs}). Setting actual batch size to 1.")
    
    val_bs = worker_args.val_bs if worker_args.val_bs else 2
    train_workers, val_workers = 1 if worker_args.shot_num == 1 else 4, 2
    if worker_args.num_workers is not None:
        train_workers, val_workers = worker_args.num_workers, worker_args.num_workers
        
    sampler = None
    if torch.distributed.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        actual_train_bs = int(actual_train_bs / torch.distributed.get_world_size())
        
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=actual_train_bs, shuffle=sampler is None, num_workers=train_workers,
        sampler=sampler, drop_last=False, collate_fn=train_collate_fn,
        worker_init_fn=partial(worker_init_fn, base_seed=3407)
    )
    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=val_bs, shuffle=False, num_workers=val_workers,
        drop_last=False, collate_fn=val_collate_fn, worker_init_fn=partial(worker_init_fn, base_seed=3407)
    )
    
    # SET UP VANILLA SAM MODEL
    sam_type = getattr(worker_args, 'sam_type', 'vit_b')
    model = sam_model_registry[sam_type]().to(device=device)
    
    if torch.distributed.is_initialized():
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        try:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
            )
        except Exception as e:
            print(f"Error initializing DistributedDataParallel: {e}")
            model = model.to(device=device)
    
    # Optimizer and scheduler
    optimizer, scheduler = setup_optimizer_and_scheduler(model, worker_args)
    
    best_miou = 0
    metrics = StreamSegMetrics(class_names=['Background', 'Foreground'])
    
    scaler = torch.amp.GradScaler('cuda') 
    print(f"Validation will be performed every {worker_args.valid_per_epochs} epochs.")
    
    for epoch in range(1, max_epoch_num + 1):
        if epoch % worker_args.valid_per_epochs == 1 or epoch == max_epoch_num:
            miou = validate_one_epoch(epoch, val_dataloader, metrics, model, device, max_epoch_num, worker_args)
            print(f"Epoch {epoch} - mIoU {data_type.upper()}: {miou:.2%}")
            
            if miou > best_miou:
                best_miou = miou
                print(f'Best mIoU {data_type.upper()} has been updated to {best_miou:.2%}!')
                
                if worker_args.save_model and epoch > 4:
                    save_path = os.path.join(worker_args.exp_dir, f"vanilla_sam_{data_type}_best.pth")
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'best_miou': best_miou,
                    }, save_path)
                    print(f"Model saved to {save_path}")
                    if worker_args.wandb:
                        wandb.save(save_path)
        
        train_one_epoch(epoch, train_dataloader, model, optimizer, scheduler, device, local_rank, worker_args, max_epoch_num, scaler)

if __name__ == '__main__':
    print("Starting vanilla SAM training process...")
    args = parse()
    
    # Add data_type argument if not present
    if not hasattr(args, 'data_type'):
        args.data_type = 'ar'  # Default to AR
    
    if hasattr(args, 'wandb') and args.wandb:
        project_name = args.project_name if hasattr(args, 'project_name') else "vanilla-sam-climate"
        run_name = args.run_name if hasattr(args, 'run_name') else f"vanilla_sam_{args.data_type}"
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

