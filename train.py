import random
import numpy as np
import torch
import os
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from functools import partial
from torch.utils.data import DataLoader
from train_util import batch_to_cuda, get_idle_gpu, get_idle_port, set_randomness,  plot_with_projection
from loss_function import ClimateLoss, compute_climate_loss
from tqdm import tqdm
from contextlib import nullcontext
from train_parser import parse
from climatesam import ClimateSAM
from dataset.climatenet import ClimateDataset
from evaluator import StreamSegMetrics
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
    Sets up a joint optimizer and scheduler for CAT-SAM and U-Net models.
    """
    # Learning rate and weight decay with defaults
    lr = worker_args.lr if hasattr(worker_args, 'lr') else 1e-3
    weight_decay = worker_args.weight_decay if hasattr(worker_args, 'weight_decay') else 1e-4

    # Combine parameters from both models
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

    
def train_one_epoch(epoch, train_dataloader, model, optimizer, scheduler, device, local_rank, worker_args, max_epoch_num, scaler):
    model.train(mode = True, phase = worker_args.phase, verbose = False)
    
    # Get gradient accumulation steps
    gradient_accumulation_steps = getattr(worker_args, 'gradient_accumulation_steps', 1)
    
    # Calculate effective number of optimizer steps
    effective_steps = len(train_dataloader) // gradient_accumulation_steps
    if len(train_dataloader) % gradient_accumulation_steps != 0:
        effective_steps += 1
    
    # Create nested progress bars if you're the main process
    if local_rank == 0:
        # Outer progress bar for batches
        batch_pbar = tqdm(total=len(train_dataloader), desc='Batches', position=0, leave=True)
        # Inner progress bar for optimizer steps
        step_pbar = tqdm(total=effective_steps, desc='Optimizer Steps', position=1, leave=True)
    else:
        batch_pbar = None
        step_pbar = None
    
    step_count = 0 
    # Initialize accumulated loss for logging
    accumulated_loss_dict = {}
    
    for train_step, batch in enumerate(train_dataloader):
        batch = batch_to_cuda(batch, device)
        
        with torch.amp.autocast('cuda'):
            tc_mask, ar_mask, _ = model(batch['input'],
                                    ar_point_prompts = batch['ar_point_prompts'],
                                    tc_point_prompts = batch['tc_point_prompts'], 
                                    ar_bbox_prompts = batch['ar_bbox_prompts'], 
                                    tc_bbox_prompts= batch['tc_bbox_prompts'],
                                    ar_mask_prompts = batch['ar_mask_prompts'],
                                    tc_mask_prompts = batch['tc_mask_prompts']
                                    )
            
            masks_ar_gt = batch['ar_object_masks']
            masks_tc_gt = batch['tc_object_masks']
            
            # Compute loss using the new loss function
            loss_dict = compute_climate_loss(
                ar_masks=ar_mask,
                tc_masks=tc_mask,
                ar_masks_gt=masks_ar_gt,
                tc_masks_gt=masks_tc_gt,
                device=device,
                worker_args=worker_args
            )
        
        total_loss = loss_dict.pop('total_loss_for_backward')
        
        # Scale loss by gradient accumulation steps
        total_loss = total_loss / gradient_accumulation_steps
        
        # Accumulate losses for logging
        for key, value in loss_dict.items():
            if key not in accumulated_loss_dict:
                accumulated_loss_dict[key] = 0
            accumulated_loss_dict[key] += value.item() / gradient_accumulation_steps

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
        if batch_pbar:
            batch_pbar.update(1)
            batch_pbar.set_postfix({
                'batch': f"{train_step + 1}/{len(train_dataloader)}",
                'loss': f"{total_loss.item():.4f}"
            })
        
        # Only update optimizer every gradient_accumulation_steps
        if (train_step + 1) % gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Calculate effective step for logging
            effective_step = (train_step + 1) // gradient_accumulation_steps
            
            # Log accumulated losses
            if worker_args.wandb:
                log_dict = {f"train/{key}": accumulated_loss_dict[key] for key in accumulated_loss_dict.keys()}
                log_dict["epoch"] = epoch
                log_dict["effective_step"] = effective_step
                # Use a consistent step calculation across training and validation
                wandb.log(log_dict, step=epoch * (len(train_dataloader) // gradient_accumulation_steps) + effective_step)
            
            # Distributed reduction of accumulated losses
            if torch.distributed.is_initialized():
                for key in accumulated_loss_dict.keys():
                    tensor_loss = torch.tensor(accumulated_loss_dict[key], device=device)
                    torch.distributed.reduce(tensor_loss, dst=0, op=torch.distributed.ReduceOp.SUM)
                    accumulated_loss_dict[key] = (tensor_loss / torch.distributed.get_world_size()).item()
            
            # Update step progress bar with accumulated losses
            if step_pbar:
                step_pbar.update(1)
                step_pbar.set_postfix({
                    'step': f"{effective_step}/{effective_steps}",
                    'total_loss': f"{accumulated_loss_dict.get('total_loss', 0):.4f}",
                    'focal': f"{accumulated_loss_dict.get('focal_loss', 0):.4f}",
                    'tversky': f"{accumulated_loss_dict.get('tversky_loss', 0):.4f}"
                })
                
            step_count += 1
            # Reset accumulated losses
            accumulated_loss_dict = {}

    # Handle any remaining gradients if the last batch doesn't complete a full accumulation
    if len(train_dataloader) % gradient_accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        step_count += 1
        if step_pbar:
            step_pbar.update(1)
            
    # Close progress bars
    if batch_pbar:
        batch_pbar.close()
    if step_pbar:
        step_pbar.close()
            
    scheduler.step()

@torch.no_grad()
def validate_one_epoch(epoch, val_dataloader, ar_metrics, tc_metrics, model, device, max_epoch_num, worker_args):
    model.eval()
    print(f"Starting validation for epoch {epoch}...")
    valid_pbar = tqdm(total=len(val_dataloader), desc='valid', leave=False)
    
    # Get gradient accumulation steps for proper step calculation
    gradient_accumulation_steps = getattr(worker_args, 'gradient_accumulation_steps', 1)
    
    for val_step, batch in enumerate(val_dataloader):
        batch = batch_to_cuda(batch, device)
        val_model = model
        with torch.no_grad():
            
            tc_masks, ar_masks, images = val_model(batch['input'],
                                ar_point_prompts = batch['ar_point_prompts'],
                                tc_point_prompts = batch['tc_point_prompts'], 
                                ar_bbox_prompts = batch['ar_bbox_prompts'], 
                                tc_bbox_prompts= batch['tc_bbox_prompts'],
                                )
            
            masks_gt = batch['gt_mask']
            masks_ar_gts = [ (mask == 2).to(torch.uint8) for mask in masks_gt ]
            masks_tc_gts = [ (mask == 1).to(torch.uint8) for mask in masks_gt ]
            # some processing to make sure the masks are in the right shape
            for masks in [masks_ar_gts, masks_tc_gts, ar_masks, tc_masks]:
                    for i in range(len(masks)):
                        if len(masks[i].shape) == 2:
                            masks[i] = masks[i][None, None, :]
                        if len(masks[i].shape) == 3:
                            masks[i] = masks[i][:, None, :]
                        if len(masks[i].shape) != 4:
                            raise RuntimeError
            
            # LOG - Adjust logging frequency based on effective validation steps
            # Use a similar logic as training but for validation
            effective_val_step = val_step // max(1, gradient_accumulation_steps // 2)  # Log more frequently in validation
            if effective_val_step == 1:  # Changed from val_step == 2 to be more consistent
                imges = [images[i].cpu().numpy() for i in range(len(images))]
                masks_ar = [ar_masks[i].cpu().numpy() for i in range(len(ar_masks))]
                masks_tc = [tc_masks[i].cpu().numpy() for i in range(len(tc_masks))]
                masks_ar_gt = [masks_ar_gts[i].cpu().numpy() for i in range(len(masks_ar_gts))]
                masks_tc_gt = [masks_tc_gts[i].cpu().numpy() for i in range(len(masks_tc_gts))]
                for i in range(len(imges)):
                    save_path=os.path.join(worker_args.exp_dir, worker_args.run_name, 'images', f"epoch_{epoch}_step_{val_step}_image_{i}.png")
                    
                    plot, titel = plot_with_projection(imges[i], masks_ar[i], masks_tc[i], masks_ar_gt[i], masks_tc_gt[i], save_path = save_path, epoch=epoch)
                
                    print(f"Epoch {epoch}- Image {i} saved.")
                    if worker_args.wandb:
                        # Use epoch-based step for validation logging to align with training
                        wandb.log({f"valid/image_{i}": wandb.Image(plot, caption=titel), "epoch": epoch}, step = epoch)
                del imges, masks_ar, masks_tc, masks_ar_gt, masks_tc_gt
                torch.cuda.empty_cache()
                
            ar_metrics.update(tc_masks, masks_ar_gts,  batch['index_name'])
            tc_metrics.update(ar_masks, masks_tc_gts,  batch['index_name'])
            valid_pbar.update(1)
            str_step_info = "Epoch: {epoch}/{epochs:4}.".format(
                epoch=epoch, epochs=max_epoch_num
            )
            valid_pbar.set_postfix_str(str_step_info)
            
    ar_metrict_dict, _ = ar_metrics.compute()
    tc_metric_dict, _ = tc_metrics.compute()
    
    miou_ar = ar_metrict_dict['Mean Foreground IoU']
    mean_acc_ar = ar_metrict_dict['Mean Acc']
    overall_acc_ar = ar_metrict_dict['Overall Acc']
    freqw_acc_ar = ar_metrict_dict['FreqW Acc']
    miout_including_bg_ar = ar_metrict_dict['Mean IoU']
    miou_tc = tc_metric_dict['Mean Foreground IoU']
    mean_acc_tc = tc_metric_dict['Mean Acc']
    overall_acc_tc = tc_metric_dict['Overall Acc']
    freqw_acc_tc = tc_metric_dict['FreqW Acc']
    miout_including_bg_tc = tc_metric_dict['Mean IoU']
    ar_metrics.reset()
    tc_metrics.reset()
    
    if worker_args.wandb:
        wandb.log({
            "valid/miou_ar": miou_ar,
            "valid/miou_tc": miou_tc,
            "valid/mean_acc_ar": mean_acc_ar,
            "valid/mean_acc_tc": mean_acc_tc,
            "valid/overall_acc_ar": overall_acc_ar,
            "valid/overall_acc_tc": overall_acc_tc,
            "valid/freqw_acc_ar": freqw_acc_ar,
            "valid/freqw_acc_tc": freqw_acc_tc,
            "valid/miout_including_bg_ar": miout_including_bg_ar,
            "valid/miout_including_bg_tc": miout_including_bg_tc,
            "epoch": epoch,
        },
            step = epoch)
        
    return miou_tc, miou_ar
        
            
            
        
    
        
def main_worker(worker_id, worker_args):
    set_randomness()
    max_epoch_num = worker_args.max_epoch_num 
    if isinstance(worker_id, str):
        worker_id = int(worker_id)
    device, local_rank = setup_device_and_distributed(worker_id, worker_args)
    print(f"Worker {worker_id} initialized on device {device} with local_rank {local_rank}.")
    
    # PREPARE DATASET
    dataset_dir = worker_args.data_dir
    train_dataset = ClimateDataset(
        data_dir=dataset_dir, train_flag=True, shot_num=worker_args.shot_num,
        transforms=None
    )
    val_dataset = ClimateDataset(data_dir=dataset_dir, train_flag=False)
    
    train_collate_fn = train_dataset.collate_fn
    val_collate_fn = val_dataset.collate_fn

    if hasattr(worker_args, 'debugging') and worker_args.debugging:
        debug_size = getattr(worker_args, 'debug_size', 50)  # Default to 50 samples
        indices = list(range(min(debug_size, len(train_dataset))))
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        print(f"Debug mode: Using only {len(train_dataset)} training samples")

        debug_val_size = getattr(worker_args, 'debug_val_size', 10)  # Default to 10 samples
        val_indices = list(range(min(debug_val_size, len(val_dataset))))
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
        print(f"Debug mode: Using only {len(val_dataset)} validation samples")
        
        max_epoch_num = 10
        worker_args.valid_per_epochs = 2
        print(f"Debug mode: Setting max_epoch_num to {max_epoch_num} and valid_per_epochs to {worker_args.valid_per_epochs}")
        
    
    # DataLoader
    train_bs = worker_args.train_bs if worker_args.train_bs else (1 if worker_args.shot_num == 1 else 4)
    gradient_accumulation_steps = getattr(worker_args, 'gradient_accumulation_steps', 1)
    
    # Adjust batch size for gradient accumulation
    actual_train_bs = train_bs // gradient_accumulation_steps
    if actual_train_bs < 1:
        actual_train_bs = 1
        print(f"Warning: gradient_accumulation_steps ({gradient_accumulation_steps}) is larger than train_bs ({train_bs}). Setting actual batch size to 1.")
    
    effective_batch_size = actual_train_bs * gradient_accumulation_steps
    if torch.distributed.is_initialized():
        effective_batch_size *= torch.distributed.get_world_size()
    
    print(f"Effective batch size: {effective_batch_size} (actual_bs: {actual_train_bs}, accumulation: {gradient_accumulation_steps})")
    
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
        drop_last=False, collate_fn=val_collate_fn
    )
    
    # SET UP MODEL
    model = ClimateSAM(model_type=worker_args.sam_type, mlp_ratio=worker_args.image_encoder_mlp_ratio).to(device=device)
    if torch.distributed.is_initialized():
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        try:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
            )
        except Exception as e:
            print(f"Error initializing DistributedDataParallel: {e}")
            model = model.to(device=device)
    
    # Load pretrained weights
    if worker_args.load_pretrained:
        if worker_args.phase == 1:
            image_encoder_path = os.path.join(worker_args.exp_dir, f"phase_1_weights.pth")
            phase_1_checkpoint = torch.load(image_encoder_path, map_location=device)
            print(f"Pretrained weights from phase 1 loaded from {image_encoder_path}")
            model.image_encoder.load_state_dict(phase_1_checkpoint['image_encoder'])
            print(f"Image encoder weights loaded from {image_encoder_path}")
            model.mask_decoder.load_state_dict(phase_1_checkpoint['mask_decoder'])
            print(f"Mask decoder weights loaded from {image_encoder_path}")
            
    # Optimizer and scheduler
    optimizer, scheduler = setup_optimizer_and_scheduler(model, worker_args)
    if worker_args.phase == 2:
        model.enable_prompt_generator()
        optimizer.add_param_group({'params': model.prompt_generator.parameters()})
    best_miou_tc = 0
    best_miou_ar = 0
    best_miou_total = 0
    ar_metrics = StreamSegMetrics(class_names=['Background', 'Foreground'])
    tc_metrics = StreamSegMetrics(class_names=['Background', 'Foreground'])
    
    scaler = torch.amp.GradScaler('cuda') 
    print(f"Validation will be performed every {worker_args.valid_per_epochs} epochs.")
    model.train(mode = True, phase = worker_args.phase, verbose=True)
    for epoch in range(1, max_epoch_num + 1):
        train_one_epoch(epoch, train_dataloader, model, optimizer, scheduler, device, local_rank, worker_args, max_epoch_num, scaler)
        if epoch % worker_args.valid_per_epochs == 0 or epoch == max_epoch_num:
            miou_tc, miou_ar = validate_one_epoch(epoch, val_dataloader, ar_metrics, tc_metrics, model, device, max_epoch_num, worker_args)
            print(f"Epoch {epoch} - mIoU TC: {miou_tc:.2%}, mIoU AR: {miou_ar:.2%}")
            if miou_tc > best_miou_tc:
                best_miou_tc = miou_tc
                print(f'Best mIoU TC has been updated to {best_miou_tc:.2%}!')
            if miou_ar > best_miou_ar:
                best_miou_ar = miou_ar
                print(f'Best mIoU AR has been updated to {best_miou_ar:.2%}!')
            if (miou_tc + miou_ar) / 2 > best_miou_total:
                best_miou_total = (miou_tc + miou_ar) / 2
                print(f'Best mIoU Total has been updated to {best_miou_total:.2%}!')
                if worker_args.save_model and epoch > 5:
                    if worker_args.phase == 1:
                        save_path = os.path.join(worker_args.exp_dir, f"phase_1_weights.pth")
                        phase_1_weights = {
                            'image_encoder': model.image_encoder.state_dict(),
                            'mask_decoder': model.mask_decoder.state_dict(),
                        }
                        torch.save(phase_1_weights, save_path)
                        print(f"Image encoder saved to {save_path}")
                        wandb.save(save_path)
                        print(f"Image encoder saved to wandb: {save_path}")
        
if __name__ == '__main__':
    print("Starting training process...")
    args = parse()
    
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
        args.used_gpu, args.gpu_num = [], 0

    # launch the experiment process for both single-GPU and multi-GPU settings
    if len(args.used_gpu) == 1:
        main_worker(worker_id=0, worker_args=args)

