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
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from functools import partial
from torch.utils.data import DataLoader
from utility import batch_to_cuda, get_idle_gpu, get_idle_port, set_randomness,  plot_with_projection, plot_mask_with_points_and_bbox, prompt_debug, setup_optimizer_and_scheduler, worker_init_fn, setup_device_and_distributed
from loss_function import ClimateLoss, compute_climate_loss
from tqdm import tqdm
from contextlib import nullcontext
from parser_config import parse
from model.climatesam import ClimateSAM
from model.prompt.cgnet import CGNetPrompter
from model.prompt.prompt_maker import PromptMaker
from dataset.climatenet import ClimateDataset
from evaluator import StreamSegMetrics
import copy
import wandb




    
def train_one_epoch(epoch, train_dataloader, model, optimizer, scheduler, device, local_rank, worker_args, max_epoch_num, scaler, prompter=None):
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
        batch_pbar = tqdm(total=len(train_dataloader), desc=f'Epoch {epoch}/{max_epoch_num} - Batches', position=0, leave=True)
    
    step_count = 0 
    # Initialize accumulated loss for logging across the entire epoch
    epoch_loss_dict = {}
    epoch_loss_count = 0
    
    for train_step, batch in enumerate(train_dataloader):
        batch = batch_to_cuda(batch, device)
        
        # Use CGNet prompter to generate bbox prompts (fallback to dataloader prompts if missing)
        ar_bbox_prompts = None
        tc_bbox_prompts = None
        try:
            if prompter is not None and 'cgnet_input' in batch:
                features = batch['cgnet_input'].to(device=device, dtype=torch.float32)
                aux_mask = prompter.get_aux_mask(features)
                prompt_maker = PromptMaker(prompt_type='bbox', positive_point_num=0, negative_point_num=0, centroid_ratio=0)
                prompt_dict = prompt_maker.make_prompts(
                    multiclass_mask=aux_mask,
                    prompt_type='bbox',
                    positive_point_num=0,
                    negative_point_num=0,
                    enlarge_ratio=0.0,
                    centroid_ratio=0
                )
                ar_bbox_prompts = [item.to(device=device, dtype=torch.float32) if item is not None else None for item in prompt_dict.get('ar_bbox_prompts', [None]*len(batch['input']))]
                tc_bbox_prompts = [item.to(device=device, dtype=torch.float32) if item is not None else None for item in prompt_dict.get('tc_bbox_prompts', [None]*len(batch['input']))]
        except Exception as e:
            print(f"Warning: CGNet prompt generation failed, falling back to dataloader prompts. Error: {e}")
            ar_bbox_prompts = batch.get('ar_bbox_prompts')
            tc_bbox_prompts = batch.get('tc_bbox_prompts')
        
        with torch.amp.autocast('cuda'):
            image_embeddings, interm_features, image_input, ori_img_size = model.encode_images(batch['input'])
            tc_mask, ar_mask, _ = model.forward(
                image_input=image_input,
                image_embeddings=image_embeddings,
                interm_embeddings=interm_features,
                ori_img_size=ori_img_size,
                ar_point_prompts=None,
                tc_point_prompts=None,
                ar_bbox_prompts=ar_bbox_prompts,
                tc_bbox_prompts=tc_bbox_prompts,
                ar_mask_prompts=None,
                tc_mask_prompts=None
            )
            
            # prompt_debug(batch, 'Train Step {train_step}')
            # Align ground-truth masks to prediction shapes to avoid size mismatches
            raw_ar_gt = batch.get('ar_object_masks', [])
            raw_tc_gt = batch.get('tc_object_masks', [])

            def _align_gt_to_pred(pred_list, raw_gt_list):
                aligned = []
                for i, pred in enumerate(pred_list):
                    raw = raw_gt_list[i] if (isinstance(raw_gt_list, (list, tuple)) and i < len(raw_gt_list)) else raw_gt_list[i] if (hasattr(raw_gt_list, '__getitem__') and i < len(raw_gt_list)) else None
                    if raw is None:
                        aligned.append(None)
                        continue
                    # Convert to tensor on correct device
                    if not torch.is_tensor(raw):
                        raw = torch.as_tensor(raw, device=device)
                    else:
                        raw = raw.to(device)

                    # Try to coerce dtype to match pred
                    try:
                        raw = raw.to(dtype=pred.dtype)
                    except Exception:
                        raw = raw.float()

                    # Normalize common cases to match pred's batch/channel dims
                    try:
                        if pred.dim() == 4:
                            # want raw shape (B, C, H, W) or (B,1,H,W)
                            if raw.dim() == 2:
                                raw = raw.unsqueeze(0).unsqueeze(0)
                            elif raw.dim() == 3:
                                raw = raw.unsqueeze(1)
                            elif raw.dim() == 4:
                                pass
                        elif pred.dim() == 3:
                            # want raw shape (B, H, W)
                            if raw.dim() == 2:
                                raw = raw.unsqueeze(0)
                            elif raw.dim() == 4 and raw.size(1) == 1:
                                raw = raw.squeeze(1)
                        elif pred.dim() == 2:
                            # want raw shape (H, W)
                            raw = raw.squeeze()
                    except Exception:
                        pass

                    # Final attempt to match shapes
                    if raw.shape != pred.shape:
                        try:
                            raw = raw.reshape(pred.shape)
                        except Exception:
                            # Fallback: broadcast or expand first dimension if possible
                            try:
                                if raw.dim() == pred.dim() - 1:
                                    raw = raw.unsqueeze(0)
                                else:
                                    raw = raw.expand(pred.shape)
                            except Exception:
                                # as last resort convert to zeros of pred shape
                                raw = torch.zeros_like(pred, device=device)

                    aligned.append(raw)
                return aligned

            masks_ar_gt = _align_gt_to_pred(ar_mask, raw_ar_gt)
            masks_tc_gt = _align_gt_to_pred(tc_mask, raw_tc_gt)
            
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

    # Handle any remaining gradients if the last batch doesn't complete a full accumulation
    if len(train_dataloader) % gradient_accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        step_count += 1
        epoch_loss_count += 1
    
    # Step the scheduler once per epoch after all optimizer steps
    scheduler.step()
    
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
    # if step_pbar:
    #     step_pbar.close()
    
    # Update scaler exactly once per epoch
    # scaler.update()

@torch.no_grad()
def validate_one_epoch(epoch, val_dataloader, ar_metrics, tc_metrics, model, device, max_epoch_num, worker_args, prompter):

    # Example usage inside validation loop:
    # plot_mask_with_points(batch['gt_mask'][0], batch['tc_point_prompts'])
    model.eval()
    valid_pbar = tqdm(total=len(val_dataloader), desc='valid', leave=False)
    
    for val_step, batch in enumerate(val_dataloader):
        batch = batch_to_cuda(batch, device)
        
        # Set inference images once
        images = model.set_infer_img(batch['input'])

        # ar_point_prompts_copy = copy.deepcopy(batch['ar_point_prompts'])
        # tc_point_prompts_copy = copy.deepcopy(batch['tc_point_prompts'])
        # ar_bbox_prompts_copy = copy.deepcopy(batch['ar_bbox_prompts'])
        # tc_bbox_prompts_copy = copy.deepcopy(batch['tc_bbox_prompts'])

        # # prompt_debug(batch, text=f"Validation Step {val_step}")
        
        # # If a CGNet prompter is provided, generate bbox prompts (enlarge_ratio=0.0)
        # ar_point_prompts_to_use = batch.get('ar_point_prompts')
        # tc_point_prompts_to_use = batch.get('tc_point_prompts')
        # ar_bbox_prompts_to_use = batch.get('ar_bbox_prompts')
        # tc_bbox_prompts_to_use = batch.get('tc_bbox_prompts')

        features = batch['cgnet_input'].to(device=device, dtype=torch.float32)
        aux_mask = prompter.get_aux_mask(features)
        prompt_maker = PromptMaker(prompt_type='bbox', positive_point_num=0, negative_point_num=0, centroid_ratio=0)
        prompt_dict = prompt_maker.make_prompts(
            multiclass_mask=aux_mask,
            prompt_type='bbox',
            positive_point_num=0,
            negative_point_num=0,
            enlarge_ratio=0.0,
            centroid_ratio=0
        )
        ar_point_prompts_copy = None
        tc_point_prompts_copy = None
        ar_bbox_prompts_copy = copy.deepcopy(prompt_dict['ar_bbox_prompts'])
        tc_bbox_prompts_copy = copy.deepcopy(prompt_dict['tc_bbox_prompts'])
        # Move bbox prompts to device/dtype
        if 'ar_bbox_prompts' in prompt_dict and prompt_dict['ar_bbox_prompts'] is not None:
            ar_bbox_prompts_to_use = [item.to(device=device, dtype=torch.float32) if item is not None else None for item in prompt_dict['ar_bbox_prompts']]
        if 'tc_bbox_prompts' in prompt_dict and prompt_dict['tc_bbox_prompts'] is not None:
            tc_bbox_prompts_to_use = [item.to(device=device, dtype=torch.float32) if item is not None else None for item in prompt_dict['tc_bbox_prompts']]
        # override point prompts (keep None)
        ar_point_prompts_to_use = None
        tc_point_prompts_to_use = None


        # Perform inference with prompts
        tc_masks, ar_masks = model.infer(
            ar_point_prompts=ar_point_prompts_to_use,
            tc_point_prompts=tc_point_prompts_to_use,
            ar_bbox_prompts=ar_bbox_prompts_to_use,
            tc_bbox_prompts=tc_bbox_prompts_to_use
        )
        
        masks_gt = batch['gt_mask']
        masks_ar_gts = [(mask == 2).to(torch.uint8) for mask in masks_gt]
        masks_tc_gts = [(mask == 1).to(torch.uint8) for mask in masks_gt]
        
        # some processing to make sure the masks are in the right shape
        for masks in [masks_ar_gts, masks_tc_gts, ar_masks, tc_masks]:
                for i in range(len(masks)):
                    if len(masks[i].shape) == 2:
                        masks[i] = masks[i][None, None, :]
                    if len(masks[i].shape) == 3:
                        masks[i] = masks[i][:, None, :]
                    if len(masks[i].shape) != 4:
                        raise RuntimeError
        # LOG
        if val_step == 0:
            # Collect all images for this epoch
            wandb_images = {}
            masks_gt_copy = copy.deepcopy(masks_gt)
            tc_masks_copy = copy.deepcopy(tc_masks)
            ar_masks_copy = copy.deepcopy(ar_masks)
            for i in range(len(masks_gt)):
                mask = masks_gt_copy[i]
                ar_points = ar_point_prompts_copy[i]
                tc_points = tc_point_prompts_copy[i]
                ar_bbox = ar_bbox_prompts_copy[i]
                tc_bbox = tc_bbox_prompts_copy[i]
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
        augmented=worker_args.augmented, generate_prompt=True, enlarge_ratio=worker_args.gt_prompt_enlarge_ratio, prompt_type=worker_args.prompt_type
    )
    val_dataset = ClimateDataset(data_dir=dataset_dir, train_flag=False, augmented=False, generate_prompt=True, worker_args=worker_args, enlarge_ratio=worker_args.gt_prompt_enlarge_ratio, prompt_type=worker_args.prompt_type)

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
        
    if hasattr(worker_args, 'hp_mode') and worker_args.hp_mode:
        # Use 20% of the training set for hyperparameter tuning by default
        orig_train_len = len(train_dataset)
        # Allow override via worker_args.hp_size, otherwise use 20% (at least 1)
        hp_size = getattr(worker_args, 'hp_size', max(1, int(orig_train_len * 0.1)))
        hp_size = min(hp_size, orig_train_len)
        # Deterministic sampling for reproducibility; seed can be overridden with hp_seed
        rng = random.Random(getattr(worker_args, 'hp_seed', 3407))
        indices = rng.sample(range(orig_train_len), k=hp_size)
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        print(f"Hyperparameter mode enabled: using {hp_size}/{orig_train_len} training samples (~{hp_size / orig_train_len * 100:.2f}%).")
        
        orig_val_len = len(val_dataset)
        hp_val_size = getattr(worker_args, 'hp_val_size', max(1, int(orig_val_len * 0.1)))
        hp_val_size = min(hp_val_size, orig_val_len)
        rng_val = random.Random(getattr(worker_args, 'hp_seed', 3407))
        val_indices = rng_val.sample(range(orig_val_len), k=hp_val_size)
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
        print(f"Hyperparameter mode: using {hp_val_size}/{orig_val_len} validation samples (~{hp_val_size / orig_val_len * 100:.2f}%).")
        
        # max_epoch_num = 
        # worker_args.valid_per_epochs = 5
        # print(f"Hyperparameter mode: Setting max_epoch_num to {max_epoch_num} and valid_per_epochs to {worker_args.valid_per_epochs}") 

        
        
        


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
        
    g = torch.Generator()
    g.manual_seed(3407)
        
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=actual_train_bs, shuffle=sampler is None, num_workers=train_workers,
        sampler=sampler, drop_last=False, collate_fn=train_collate_fn,
        worker_init_fn=partial(worker_init_fn, base_seed=3407), generator=g
    )
    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=val_bs, shuffle=False, num_workers=val_workers,
        drop_last=False, collate_fn=val_collate_fn, worker_init_fn=partial(worker_init_fn, base_seed=3407), generator=g
    )
    
    # SET UP MODEL - enable W&B logging only if debugging is True
    model = ClimateSAM(
        model_type=worker_args.sam_type, 
        mlp_ratio=worker_args.image_encoder_mlp_ratio,
        enable_wandb_logging=getattr(worker_args, 'debugging', False)  # Only log if debugging=True
    ).to(device=device)
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
        if worker_args.pretrained_name is not None:
            image_encoder_path = os.path.join(worker_args.exp_dir, f"{worker_args.pretrained_name}.pth") 
        else:
            image_encoder_path = os.path.join(worker_args.exp_dir, f"infused_token_{worker_args.sam_type}_{worker_args.image_encoder_mlp_ratio}_{worker_args.run_name}.pth")
        if not os.path.exists(image_encoder_path):
            print(f"Pretrained weights not found at {image_encoder_path}. Please check the path and try again.")
        else:
            phase_1_checkpoint = torch.load(image_encoder_path, map_location=device)
            print(f"Pretrained weights from phase 1 loaded from {image_encoder_path}")
            if 'image_encoder' not in phase_1_checkpoint:
                print(f"Image encoder weights not found in checkpoint.")
            else:
                model.image_encoder.load_state_dict(phase_1_checkpoint['image_encoder'])
                print(f"Image encoder weights loaded from {image_encoder_path}")
            if 'mask_decoder' not in phase_1_checkpoint:
                print(f"Mask decoder weights not found in checkpoint.")
            else:
                model.mask_decoder.load_state_dict(phase_1_checkpoint['mask_decoder'])
                print(f"Mask decoder weights loaded from {image_encoder_path}")
            if 'input_adapter' not in phase_1_checkpoint:
                print(f"Input adapter weights not found in checkpoint.")
            else:
                model.input_adapter.load_state_dict(phase_1_checkpoint['input_adapter'])
                print(f"Input adapter weights loaded from {image_encoder_path}")

    
            
    # Optimizer and scheduler
    optimizer, scheduler = setup_optimizer_and_scheduler(model, worker_args)
    
    # Initialize CGNet prompter to generate bbox prompts with enlarge_ratio=0.0 during training
    save_path = os.path.join(worker_args.exp_dir, "cgnet_weight.pth")
    prompter = None
    try:
        prompter = CGNetPrompter(weights_path=save_path, device=device, worker_args=worker_args)
        print(f"CGNetPrompter initialized with weights: {save_path}")
    except Exception as e:
        print(f"Warning: Failed to initialize CGNetPrompter from {save_path}: {e}")
        prompter = None
    
    best_miou_tc = 0
    best_miou_ar = 0
    best_miou_total = 0
    ar_metrics = StreamSegMetrics(class_names=['Background', 'Foreground'])
    tc_metrics = StreamSegMetrics(class_names=['Background', 'Foreground'])
    
    scaler = torch.amp.GradScaler('cuda') 
    print(f"Validation will be performed every {worker_args.valid_per_epochs} epochs.")
    model.train(mode = True, phase = worker_args.phase, verbose=True)
    for epoch in range(1, max_epoch_num + 1):
        
        if sampler is not None:
            sampler.set_epoch(epoch)
        
        if epoch % worker_args.valid_per_epochs == 1 or epoch == max_epoch_num:
            if worker_args.load_pretrained or epoch > 1: 
                miou_tc, miou_ar = validate_one_epoch(epoch, val_dataloader, ar_metrics, tc_metrics, model, device, max_epoch_num, worker_args, prompter=prompter)
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
                    if worker_args.save_model and epoch > 4:
                        save_path = os.path.join(worker_args.exp_dir, f"infused_token_{worker_args.sam_type}_{worker_args.image_encoder_mlp_ratio}_{worker_args.run_name}.pth")
                        phase_1_weights = {
                            'image_encoder': model.image_encoder.state_dict(),
                            'mask_decoder': model.mask_decoder.state_dict(),
                            'input_adapter': model.input_adapter.state_dict(),
                        }
                        torch.save(phase_1_weights, save_path)
                        print(f"Image encoder saved to {save_path}")
                        wandb.save(save_path)
                        print(f"Image encoder saved to wandb: {save_path}")
               
        train_one_epoch(epoch, train_dataloader, model, optimizer, scheduler, device, local_rank, worker_args, max_epoch_num, scaler, prompter=prompter)
        
        
        
if __name__ == '__main__':
    print("Starting training process...")
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

    # launch the experiment process for both single-GPU and multi-GPU settings
    if len(args.used_gpu) == 1:
        main_worker(worker_id=0, worker_args=args)
