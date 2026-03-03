import random
import numpy as np
import torch
import os
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from functools import partial
from torch.utils.data import DataLoader
from train_util import batch_to_cuda, get_idle_gpu, get_idle_port, set_randomness,  plot_with_projection, plot_mask_with_points_and_bbox, prompt_debug, worker_init_fn, setup_optimizer_and_scheduler, setup_device
from loss_function import ClimateLoss, compute_climate_loss
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
        augmented=False, generate_prompt=False
    )
    val_dataset = ClimateDataset(data_dir=dataset_dir, train_flag=False, augmented=False)
    
    train_collate_fn = train_dataset.collate_fn
    val_collate_fn = val_dataset.collate_fn

    if hasattr(worker_args, 'debugging') and worker_args.debugging:
        debug_size = getattr(worker_args, 'debug_size', 10)  # Default to 50 samples
        indices = list(range(min(debug_size, len(train_dataset))))
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        print(f"Debug mode: Using only {len(train_dataset)} training samples")

        debug_val_size = getattr(worker_args, 'debug_val_size', 5)  # Default to 10 samples
        val_indices = list(range(min(debug_val_size, len(val_dataset))))
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
        print(f"Debug mode: Using only {len(val_dataset)} validation samples")
        
        max_epoch_num = 2
        worker_args.valid_per_epochs = 1
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
        drop_last=False, collate_fn=val_collate_fn, worker_init_fn=partial(worker_init_fn, base_seed=3407)
    )
    
    
    ##########################
    ###########################
    
    climatesam = ClimateSAM(
        model_type=worker_args.sam_type, 
        mlp_ratio=worker_args.image_encoder_mlp_ratio,
        enable_wandb_logging=getattr(worker_args, 'debugging', False)  # Only log if debugging=True
    ).to(device=device)
    
    
    image_encoder_path = os.path.join(worker_args.exp_dir, f"phase_2_weights.pth")
    phase_2_checkpoint = torch.load(image_encoder_path, map_location='cpu')
    print(f"Pretrained weights from phase 2 loaded from {image_encoder_path}")
    climatesam.image_encoder.load_state_dict(phase_2_checkpoint['image_encoder'])
    print(f"Image encoder weights loaded from {image_encoder_path}")
    climatesam.mask_decoder.load_state_dict(phase_2_checkpoint['mask_decoder'])
    print(f"Mask decoder weights loaded from {image_encoder_path}")
    climatesam.input_adapter.load_state_dict(phase_2_checkpoint['input_adapter'])
    print(f"Input adapter weights loaded from {image_encoder_path}")
    
    for param in climatesam.parameters():
        param.requires_grad = False
    
    
    num_features_map = {
                'vit_b': 12,
                'vit_l': 24,
                'vit_h': 32  # Assuming ViT-H has 32 layers
            }
    feature_per_block = {
        'vit_b': 3,
        'vit_l': 6,
        'vit_h': 9  # Assuming ViT-H has 4 features per block
    }
    in_channels = {
        'vit_b': 768,
        'vit_l': 1024,
        'vit_h': 1280
    }
    
    prompt_generator = PromptGenerator(num_features=num_features_map[worker_args.sam_type],
                                       features_per_block=feature_per_block[worker_args.sam_type],
                                       in_channels= in_channels[worker_args.sam_type]
                                      ).to(device=device)
    
    optimizer, scheduler = setup_optimizer_and_scheduler(prompt_generator, worker_args)
    
    ###################################
    ###################################
    
    best_miou_tc = 0
    best_miou_ar = 0
    best_miou_total = 0
    ar_metrics = StreamSegMetrics(class_names=['Background', 'Foreground'])
    tc_metrics = StreamSegMetrics(class_names=['Background', 'Foreground'])
    scaler = torch.amp.GradScaler('cuda') 
    print(f"Validation will be performed every {worker_args.valid_per_epochs} epochs.")
    
    for epoch in range(1, max_epoch_num + 1):
        train_one_epoch(epoch, train_dataloader, climatesam, prompt_generator, optimizer, scheduler, device, local_rank, worker_args, max_epoch_num, scaler)
        if epoch % worker_args.valid_per_epochs == 0 or epoch == max_epoch_num:
            miou_tc, miou_ar = validate_one_epoch(epoch, val_dataloader, climatesam, prompt_generator, device, ar_metrics, tc_metrics, max_epoch_num, worker_args)
            print(f"Validation results - mIoU (TC): {miou_tc}, mIoU (AR): {miou_ar}")
            if miou_tc > best_miou_tc:
                best_miou_tc = miou_tc
                print(f'Best mIoU TC has been updated to {best_miou_tc:.2%}!')
            if miou_ar > best_miou_ar:
                best_miou_ar = miou_ar
                print(f'Best mIoU AR has been updated to {best_miou_ar:.2%}!')
            if (miou_tc + miou_ar) / 2 > best_miou_total:
                best_miou_total = (miou_tc + miou_ar) / 2
                print(f'Best mIoU Total has been updated to {best_miou_total:.2%}!')
                if worker_args.save_model and epoch > 4:
                    save_path = os.path.join(worker_args.exp_dir, f"prompter_weights.pth")
                    phase_2_weights = {
                        'prompt_generator': prompt_generator.state_dict(),
                    }
                    torch.save(phase_2_weights, save_path)
                    print(f"Prompt generator saved to {save_path}")
                    wandb.save(save_path)
                    print(f"Prompt generator saved to wandb: {save_path}")


def train_one_epoch(epoch, train_dataloader, climatesam, prompt_generator, optimizer, scheduler, device, local_rank, worker_args, max_epoch_num, scaler):
    # Get gradient accumulation steps
    gradient_accumulation_steps = getattr(worker_args, 'gradient_accumulation_steps', 1)
    # Calculate effective number of optimizer steps
    effective_steps = len(train_dataloader) // gradient_accumulation_steps
    if len(train_dataloader) % gradient_accumulation_steps != 0:
        effective_steps += 1
    # Create nested progress bars if you're the main process
    if local_rank == 0:
        # Outer progress bar for batches
        batch_pbar = tqdm(total=len(train_dataloader), desc=f'Epoch {epoch}/{max_epoch_num} - Batches', position=0, leave=True)
        
    # LOSS
    step_count = 0 
    # Initialize accumulated loss for logging across the entire epoch
    epoch_loss_dict = {}
    epoch_loss_count = 0
    
    for train_step, batch in enumerate(train_dataloader):
        batch = batch_to_cuda(batch, device)
        _, _, interm_features = climatesam.set_infer_img(batch['input'])
        masks_gt = batch['gt_mask']
        masks_ar_gts = [(mask == 2).to(torch.uint8) for mask in masks_gt]
        masks_tc_gts = [(mask == 1).to(torch.uint8) for mask in masks_gt]
        with torch.amp.autocast('cuda'):
            
            tc_masks, ar_masks = prompt_generator(interm_features)
            
            # some processing to make sure the masks are in the right shape
            # for masks in [masks_ar_gts, masks_tc_gts, ar_masks, tc_masks]:
            #         for i in range(len(masks)):
            #             if len(masks[i].shape) == 2:
            #                 masks[i] = masks[i][None, None, :]
            #             if len(masks[i].shape) == 3:
            #                 masks[i] = masks[i][:, None, :]
            #             if len(masks[i].shape) != 4:
            #                 raise RuntimeError
                        
            loss_dict = compute_climate_loss(
                ar_masks=ar_masks,
                tc_masks=tc_masks,
                ar_masks_gt=masks_ar_gts,
                tc_masks_gt=masks_tc_gts,
                device=device,
                worker_args=worker_args
            )
            
        total_loss = loss_dict.pop('total_loss_for_backward')
        
        # Scale loss by gradient accumulation steps
        total_loss = total_loss / gradient_accumulation_steps
        
        # Accumulate losses for epoch-level logging
        for key, value in loss_dict.items():
            if key not in epoch_loss_dict:
                epoch_loss_dict[key] = 0
            epoch_loss_dict[key] += value.item() / gradient_accumulation_steps
        
        backward_context = nullcontext
        with backward_context():
            scaler.scale(total_loss).backward()
            
            
        if batch_pbar:
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
            
            # Calculate effective step for logging
            effective_step = (train_step + 1) // gradient_accumulation_steps
            epoch_loss_count += 1
            
            # Distributed reduction of losses for current step
            if torch.distributed.is_initialized():
                step_loss_dict = {}
                for key in loss_dict.keys():
                    step_loss_dict[key] = epoch_loss_dict[key] / epoch_loss_count
                    tensor_loss = torch.tensor(step_loss_dict[key], device=device)
                    torch.distributed.reduce(tensor_loss, dst=0, op=torch.distributed.ReduceOp.SUM)
                    step_loss_dict[key] = (tensor_loss / torch.distributed.get_world_size()).item()
            else:
                step_loss_dict = {key: epoch_loss_dict[key] / epoch_loss_count for key in epoch_loss_dict.keys()}
            step_count += 1
            
    if len(train_dataloader) % gradient_accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        step_count += 1
        epoch_loss_count += 1
        
    # Calculate average losses for the entire epoch
    if epoch_loss_count > 0:
        avg_epoch_losses = {key: epoch_loss_dict[key] / epoch_loss_count for key in epoch_loss_dict.keys()}
        
        # Final distributed reduction for epoch averages
        if torch.distributed.is_initialized():
            for key in avg_epoch_losses.keys():
                tensor_loss = torch.tensor(avg_epoch_losses[key], device=device)
                torch.distributed.reduce(tensor_loss, dst=0, op=torch.distributed.ReduceOp.SUM)
                avg_epoch_losses[key] = (tensor_loss / torch.distributed.get_world_size()).item()
        
        # Log to wandb once per epoch
        if worker_args.wandb and local_rank == 0:
            log_dict = {f"train/{key}": avg_epoch_losses[key] for key in avg_epoch_losses.keys()}
            log_dict["epoch"] = epoch
            log_dict["learning_rate"] = scheduler.get_last_lr()[0]
            wandb.log(log_dict, step=epoch)
            
    # Close progress bars
    if batch_pbar:
        batch_pbar.close()

    scheduler.step()

# def get_infer_features(climatesam, batch_input, device):
#     # Temporarily move to GPU for inference
#     climatesam.to(device)
#     _, _, interm_features = climatesam.set_infer_img(batch_input)
#     # Move back to CPU to free GPU memory
#     climatesam.to('cpu')
#     return interm_features
    
@torch.no_grad()
def validate_one_epoch(epoch, val_dataloader, climatesam, prompt_generator, device,  ar_metrics, tc_metrics, max_epoch_num, worker_args):
    # prompt_generator.eval()
    valid_pbar = tqdm(total=len(val_dataloader), desc='valid', leave=False)
    
    for val_step, batch in enumerate(val_dataloader):
        batch = batch_to_cuda(batch, device)
        
        # Set inference images once

        interm_features = get_infer_features(climatesam, batch['input'], device)
        tc_masks, ar_masks = prompt_generator(interm_features)
        masks_gt = batch['gt_mask']
        masks_ar_gts = [(mask == 2).to(torch.uint8) for mask in masks_gt]
        masks_tc_gts = [(mask == 1).to(torch.uint8) for mask in masks_gt]
        
        # some processing to make sure the masks are in the right shape
        # for masks in [masks_ar_gts, masks_tc_gts, ar_masks, tc_masks]:
        #         for i in range(len(masks)):
        #             if len(masks[i].shape) == 2:
        #                 masks[i] = masks[i][None, None, :]
        #             if len(masks[i].shape) == 3:
        #                 masks[i] = masks[i][:, None, :]
        #             if len(masks[i].shape) != 4:
        #                 raise RuntimeError
                    
        if val_step == 0:
            wandb_images = {}
            masks_gt_copy = copy.deepcopy(masks_gt)
            tc_masks_copy = copy.deepcopy(tc_masks)
            ar_masks_copy = copy.deepcopy(ar_masks)
            for i in range(len(masks_gt)):
                mask = masks_gt_copy[i]
                ar_points = None
                tc_points = None
                ar_bbox = None
                tc_bbox = None
                tc_pred_mask = tc_masks_copy[i]
                ar_pred_mask = ar_masks_copy[i]
                save_path = os.path.join(worker_args.exp_dir, worker_args.run_name, 'images', f"epoch_{epoch}_step_{val_step}_image_{i}.png")
                fig = plot_mask_with_points_and_bbox(mask, ar_points, tc_points, ar_bbox, tc_bbox, tc_pred_mask, ar_pred_mask, radius=8, save_path=save_path, axis=True)
                
                # Collect images for batch logging
                if worker_args.wandb:
                    wandb_images[f"valid/val_step_{val_step}_image_{i}"] = wandb.Image(fig, caption=f"Validation Step {val_step} Image {i}")
                    print(f"Epoch {epoch} - Image {i} prepared for logging.")
            
            # Log all images at once for the same epoch
            if worker_args.wandb and wandb_images:
                wandb_images["epoch"] = epoch
                wandb.log(wandb_images, step=epoch)
                print(f"Epoch {epoch} - All {len(wandb_images)-1} images logged to W&B together.")

            del ar_point_prompts_copy, tc_point_prompts_copy, ar_bbox_prompts_copy, tc_bbox_prompts_copy, masks_gt_copy, tc_masks_copy, ar_masks_copy
            torch.cuda.empty_cache()
                
            
                    
        
    
    tc_metrics.update(tc_masks, masks_tc_gts,  batch['index_name'])
    ar_metrics.update(ar_masks, masks_ar_gts,  batch['index_name'])
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
        args.used_gpu, args.gpu_num = [], 1

    # launch the experiment process for both single-GPU and multi-GPU settings
    if len(args.used_gpu) == 1:
        main_worker(worker_id=0, worker_args=args)

