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

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from functools import partial
from torch.utils.data import DataLoader
from tqdm import tqdm
from contextlib import nullcontext
from evaluator import StreamSegMetrics


from utility import batch_to_cuda, get_idle_gpu, get_idle_port, set_randomness,  plot_with_projection, plot_mask_with_points_and_bbox, prompt_debug, setup_device_and_distributed, setup_optimizer_and_scheduler_for_generator, worker_init_fn
from loss_function import ClimateLoss, compute_climate_loss, compute_generator_loss, calculate_generator_token_loss
from parser_config import parse
from model.climatesam import ClimateSAM
from dataset.climatenet import ClimateDataset
from model.prompt_generator_token_cgblock import PromptGenerator
from model.prompt.prompt_maker import PromptMaker

# ------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------


def train_one_epoch(epoch, train_dataloader, climatesam, prompter, prompt_maker, optimizer, scheduler, device, local_rank, worker_args, max_epoch_num, scaler, gradient_accumulation_steps):
    climatesam.eval()
    prompter.train()
    # Monitor initial memory
    # monitor_gpu_memory("Training start")
    
    # Calculate effective number of optimizer steps
    effective_steps = len(train_dataloader) // gradient_accumulation_steps
    if len(train_dataloader) % gradient_accumulation_steps != 0:
        effective_steps += 1
    
    # Create progress bar if you're the main process
    batch_pbar = None
    if local_rank == 0:
        batch_pbar = tqdm(total=len(train_dataloader), desc=f'Epoch {epoch}/{max_epoch_num} - Batches', position=0, leave=True)
        
    step_count = 0 
    # Initialize accumulated loss for logging across the entire epoch
    epoch_loss_dict = {}
    epoch_loss_count = 0
    
    for train_step, batch in enumerate(train_dataloader):
        batch = batch_to_cuda(batch, device)

        # Encode images with no_grad (we don't train ClimateSAM)
        with torch.no_grad():
            image_embeddings, interm_features, image_input, ori_img_size = climatesam.encode_images(batch['input'])
            # detach to be 100% sure no graph links back to ClimateSAM
            image_embeddings = image_embeddings.detach()
            interm_features = [f.detach() for f in interm_features]
            image_input = image_input.detach()

        # Only the prompter needs gradients
        with torch.amp.autocast('cuda'):
            # Build MLP-refined tokens from the mask decoder (use no_grad to avoid updating mask decoder)
            with torch.no_grad():
                ar_refined = climatesam.mask_decoder.hf_mlp_ar(climatesam.mask_decoder.hf_token_ar.weight.to(device))
                tc_refined = climatesam.mask_decoder.hf_mlp_tc(climatesam.mask_decoder.hf_token_tc.weight.to(device))

            # Pass refined tokens into the prompt generator for gating
            final_logit, interm_masks, ar_mask, tc_mask = prompter(
                interm_features,
                ar_refined=ar_refined,
                tc_refined=tc_refined
            )
            
            softmax_final_logit = F.softmax(final_logit, dim=1)
            multiclass_mask = torch.argmax(softmax_final_logit, dim=1)
            
            # Ground truth masks (multiclass for generator loss)
            gt_masks = torch.stack(batch['gt_mask'], dim=0).to(device)  # B, H, W
            
            # Build prompts (must be done before computing any GT-based losses)
            # Use AR/TC binary heads if available, otherwise fall back to multiclass mask
            if ar_mask is not None and tc_mask is not None:
                # pass sigmoid probabilities to PromptMaker
                ar_mask_sig = torch.sigmoid(ar_mask.detach())
                tc_mask_sig = torch.sigmoid(tc_mask.detach())
                prompt_dict = prompt_maker.make_prompts(ar_mask=ar_mask_sig, tc_mask=tc_mask_sig, prompt_type=worker_args.prompt_type, enlarge_ratio=worker_args.prompt_enlarge_ratio)
            else:
                # Use detached multiclass_mask to avoid accidental gradient flow into prompt creation
                prompt_dict = prompt_maker.make_prompts(multiclass_mask.detach(), prompt_type=worker_args.prompt_type, enlarge_ratio=worker_args.prompt_enlarge_ratio)
            prompt_dict = batch_to_cuda(prompt_dict, device)

            # Compute combined generator + optional AR/TC binary loss via helper
            ar_masks_pred = None
            tc_masks_pred = None
            ar_masks_gt = None
            tc_masks_gt = None
            if ar_mask is not None and tc_mask is not None:
                b = ar_mask.shape[0]
                ar_masks_pred = [ar_mask[i:i+1] for i in range(b)]
                tc_masks_pred = [tc_mask[i:i+1] for i in range(b)]

                # Use true per-image binary GT masks for AR/TC head supervision.
                # Each entry shape: [1, 1, H, W] to match pred list entries.
                ar_masks_gt = [((gt_masks[i:i+1] == 2).float()).unsqueeze(1) for i in range(b)]
                tc_masks_gt = [((gt_masks[i:i+1] == 1).float()).unsqueeze(1) for i in range(b)]
            
            merged_loss = calculate_generator_token_loss(
                multiclass_mask=final_logit,
                interm_masks=interm_masks,
                gt_masks=gt_masks,
                device=device,
                worker_args=worker_args,
                ar_masks_pred=ar_masks_pred,    # Logits from your PromptGenerator
                tc_masks_pred=tc_masks_pred,    # Logits from your PromptGenerator
                ar_masks_gt=ar_masks_gt,
                tc_masks_gt=tc_masks_gt
            )
            
        # ClimateSAM forward also doesn't require gradients (we don't train it)
        with torch.no_grad():
            tc_pred_masks, ar_pred_masks, _ = climatesam.forward(
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
        
        # Combine losses
        loss_dict = {}
        loss_dict.update({f"gen_{k}": v for k, v in merged_loss.items()})
        
        # Total loss for backward (already combined in merged_loss)
        total_loss = merged_loss.pop('total_loss_for_backward')
        loss_dict['total_loss_for_backward'] = total_loss

        # Scale loss by gradient accumulation steps
        total_loss = loss_dict['total_loss_for_backward'] / gradient_accumulation_steps
        
        # Accumulate losses for epoch-level logging
        for key, value in loss_dict.items():
            if key not in epoch_loss_dict:
                epoch_loss_dict[key] = 0
            epoch_loss_dict[key] += value.item() / gradient_accumulation_steps

        # Backward pass
        with nullcontext():
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
            
            step_count += 1
            epoch_loss_count += 1

            step_loss_dict = {key: epoch_loss_dict[key] / epoch_loss_count for key in epoch_loss_dict.keys()}

    if len(train_dataloader) % gradient_accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        step_count += 1
        epoch_loss_count += 1
        
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
    if local_rank == 0 and batch_pbar:
        batch_pbar.close()
            
    scheduler.step()
            
    # Close progress bar
    if batch_pbar:
        batch_pbar.close()
            
    scheduler.step()


# ------------------------------------------------------------
# EVAL 
# ------------------------------------------------------------
@torch.no_grad()
def validate_one_epoch(epoch, val_dataloader, ar_metrics, tc_metrics, climatesam, prompter, prompt_maker, device, max_epoch_num, worker_args):
    climatesam.eval()
    prompter.eval()
    
    # Add metrics for intermediate predictions
    interm_ar_metrics = StreamSegMetrics(class_names=['Background', 'Foreground'])
    interm_tc_metrics = StreamSegMetrics(class_names=['Background', 'Foreground'])
    final_logit_metrics = StreamSegMetrics(class_names=['Background', 'TC', 'AR'])  # 3-class for multiclass
    
    valid_pbar = tqdm(total=len(val_dataloader), desc='valid', leave=False)
    
    for val_step, batch in enumerate(val_dataloader):
        batch = batch_to_cuda(batch, device)
        
        with torch.no_grad():
            # Step 1: Encode images to get intermediate features (same as training)
            image_embeddings, interm_features, image_input, ori_img_size = climatesam.encode_images(batch['input'])
            # detach to be 100% sure no graph links back to ClimateSAM
            image_embeddings = image_embeddings.detach()
            interm_features = [f.detach() for f in interm_features]
            image_input = image_input.detach()
            
            
            # final_logit, interm_masks = prompter(interm_features)
            # interm_masks = [F.interpolate(mask, size=ori_img_size[0], mode='bilinear', align_corners=False) for mask in interm_masks]
            # 
            # softmax_final_logit = F.softmax(final_logit, dim=1) # B, 3, H, W
            # multiclass_mask = torch.argmax(softmax_final_logit, dim=1) # B, H, W
            # Use MLP-refined tokens from the mask decoder and pass them into the prompter
            # so that the generator can produce class-specific binary masks (AR/TC)
            ar_refined = climatesam.mask_decoder.hf_mlp_ar(climatesam.mask_decoder.hf_token_ar.weight.to(device))
            tc_refined = climatesam.mask_decoder.hf_mlp_tc(climatesam.mask_decoder.hf_token_tc.weight.to(device))

            final_logit, interm_masks, ar_mask, tc_mask = prompter(
                interm_features,
                ar_refined=ar_refined,
                tc_refined=tc_refined
            )

            if interm_masks is not None:
                interm_masks = [F.interpolate(mask, size=ori_img_size[0], mode='bilinear', align_corners=False) for mask in interm_masks]

            softmax_final_logit = F.softmax(final_logit, dim=1) # B, 3, H, W
            multiclass_mask = torch.argmax(softmax_final_logit, dim=1) # B, H, W

            # Build a multiclass mask for prompt generation using the AR/TC binary heads if available.
            # Priority: AR overrides TC in case of overlap.
            use_binary_heads = (ar_mask is not None) and (tc_mask is not None)
            if use_binary_heads:
                # ar_mask / tc_mask expected shape: (B, 1, H, W)
                ar_prob = torch.sigmoid(ar_mask).squeeze(1)  # B, H, W
                tc_prob = torch.sigmoid(tc_mask).squeeze(1)  # B, H, W

                threshold = 0.5
                pred_multiclass_for_prompts = torch.zeros_like(multiclass_mask, dtype=torch.long)
                tc_pos = (tc_prob > threshold)
                ar_pos = (ar_prob > threshold)
                # set TC then AR to allow AR override
                pred_multiclass_for_prompts[tc_pos] = 1
                pred_multiclass_for_prompts[ar_pos] = 2
            else:
                pred_multiclass_for_prompts = multiclass_mask

            prompt_dict = prompt_maker.make_prompts(pred_multiclass_for_prompts, prompt_type=worker_args.prompt_type, enlarge_ratio=worker_args.prompt_enlarge_ratio)
            prompt_dict = batch_to_cuda(prompt_dict, device)
            
            ar_point_prompts_copy = copy.deepcopy(prompt_dict['ar_point_prompts'])
            tc_point_prompts_copy = copy.deepcopy(prompt_dict['tc_point_prompts'])
            ar_bbox_prompts_copy = copy.deepcopy(prompt_dict['ar_bbox_prompts'])
            tc_bbox_prompts_copy = copy.deepcopy(prompt_dict['tc_bbox_prompts'])


            # Step 4: Forward through the rest of the model with generated prompts
            tc_pred_masks, ar_pred_masks, _ = climatesam.forward(
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
        
        # Ground truth masks
        masks_gt = batch['gt_mask']
        gt_masks_tensor = torch.stack(masks_gt, dim=0).to(device)  # B, H, W for multiclass evaluation
        masks_ar_gts = [(mask == 2).to(torch.uint8) for mask in masks_gt]
        masks_tc_gts = [(mask == 1).to(torch.uint8) for mask in masks_gt]
        
        # Convert predicted masks to list format for consistency
        ar_masks = [mask for mask in ar_pred_masks]
        tc_masks = [mask for mask in tc_pred_masks]
        
        # Process intermediate masks for evaluation
        interm_ar_masks = []
        interm_tc_masks = []
        if interm_masks is not None:
            for i in range(len(interm_masks)):
                # Assuming interm_masks are in format [batch_size, 2, H, W] where 2 = [AR, TC]
                interm_ar_mask = interm_masks[i][0, 2, :, :]  # AR channel 
                interm_tc_mask = interm_masks[i][0, 1, :, :]  # TC channel
                interm_ar_masks.append(interm_ar_mask)
                interm_tc_masks.append(interm_tc_mask)
        
        # Process final logit for evaluation (multiclass)
        final_logit_pred = torch.argmax(final_logit, dim=1)  # B, H, W
        final_logit_pred_list = [final_logit_pred[i:i+1].unsqueeze(0) for i in range(final_logit_pred.shape[0])]
        gt_masks_list = [gt_masks_tensor[i:i+1].unsqueeze(0) for i in range(gt_masks_tensor.shape[0])]
        
        # Ensure masks are in the right shape
        all_mask_lists = [masks_ar_gts, masks_tc_gts, ar_masks, tc_masks]
        if interm_ar_masks:
            all_mask_lists.extend([interm_ar_masks, interm_tc_masks])
        
        for masks in all_mask_lists:
            for i in range(len(masks)):
                if len(masks[i].shape) == 2:
                    masks[i] = masks[i][None, None, :]
                if len(masks[i].shape) == 3:
                    masks[i] = masks[i][:, None, :]
                if len(masks[i].shape) != 4:
                    raise RuntimeError(f"Unexpected mask shape: {masks[i].shape}")
        
        # LOG - Visualization for first validation step
        if val_step == 0:
            wandb_images = {}
            masks_gt_copy = copy.deepcopy(masks_gt)
            tc_masks_copy = copy.deepcopy(tc_masks)
            ar_masks_copy = copy.deepcopy(ar_masks)
            
            # Copy intermediate predictions for visualization
            interm_ar_masks_copy = copy.deepcopy(interm_ar_masks) if interm_ar_masks else None
            interm_tc_masks_copy = copy.deepcopy(interm_tc_masks) if interm_tc_masks else None
            final_logit_copy = copy.deepcopy(final_logit_pred_list)
            
            # Use generated prompts for visualization
            # ar_point_prompts_copy = copy.deepcopy(prompt_dict['ar_point_prompts'])
            # tc_point_prompts_copy = copy.deepcopy(prompt_dict['tc_point_prompts'])
            # ar_bbox_prompts_copy = copy.deepcopy(prompt_dict['ar_bbox_prompts'])
            # tc_bbox_prompts_copy = copy.deepcopy(prompt_dict['tc_bbox_prompts'])
            
            for i in range(len(masks_gt)):
                mask = masks_gt_copy[i]
                ar_points = ar_point_prompts_copy[i] if i < len(ar_point_prompts_copy) else None
                tc_points = tc_point_prompts_copy[i] if i < len(tc_point_prompts_copy) else None
                ar_bbox = ar_bbox_prompts_copy[i] if i < len(ar_bbox_prompts_copy) else None
                tc_bbox = tc_bbox_prompts_copy[i] if i < len(tc_bbox_prompts_copy) else None
                tc_pred_mask = tc_masks_copy[i]
                ar_pred_mask = ar_masks_copy[i]
                
                # Main prediction visualization
                save_path = os.path.join(worker_args.exp_dir, worker_args.run_name, 'images', f"epoch_{epoch}_step_{val_step}_image_{i}.png")
                fig = plot_mask_with_points_and_bbox(
                    mask, ar_points, tc_points, ar_bbox, tc_bbox, 
                    tc_pred_mask, ar_pred_mask, radius=8, save_path=save_path, axis=True, title = f"Epoch {epoch} - Prediction {i}"
                )
                
                if worker_args.wandb:
                    wandb_images[f"valid/val_step_{val_step}_final_pred_image_{i}"] = wandb.Image(fig, caption=f"Validation Step {val_step} Final Predictions - Image {i}")
                    print(f"Epoch {epoch} - Final image {i} logged to W&B.")
                
                # Intermediate masks visualization
                if interm_ar_masks_copy and interm_tc_masks_copy and i < len(interm_ar_masks_copy):
                    interm_ar_pred = interm_ar_masks_copy[i]
                    interm_tc_pred = interm_tc_masks_copy[i]
                    
                    interm_save_path = os.path.join(worker_args.exp_dir, worker_args.run_name, 'images', f"epoch_{epoch}_step_{val_step}_interm_{i}.png")
                    interm_fig = plot_mask_with_points_and_bbox(
                        mask, ar_points, tc_points, ar_bbox, tc_bbox,
                        interm_tc_pred, interm_ar_pred, radius=8, save_path=interm_save_path, axis=True, title = f"Epoch {epoch} - Intermediate Prediction {i}"
                    )
                    
                    if worker_args.wandb:
                        wandb_images[f"valid/val_step_{val_step}_interm_pred_image_{i}"] = wandb.Image(interm_fig, caption=f"Validation Step {val_step} Intermediate Predictions - Image {i}")
                        print(f"Epoch {epoch} - Generator Intermediate Prediction {i} logged to W&B.")

                # Final logit visualization (multiclass)
                if final_logit_copy and i < len(final_logit_copy):
                    final_logit_pred_mask = final_logit_copy[i]
                    
                    logit_save_path = os.path.join(worker_args.exp_dir, worker_args.run_name, 'images', f"epoch_{epoch}_step_{val_step}_logit_{i}.png")
                    # For multiclass, we can visualize it as a single mask with different values
                    logit_fig = plot_mask_with_points_and_bbox(
                        mask, ar_points, tc_points, ar_bbox, tc_bbox,
                        final_logit_pred_mask, final_logit_pred_mask, radius=8, save_path=logit_save_path, axis=True, title = f"Epoch {epoch} - Generator Prediction {i}"
                    )
                    
                    if worker_args.wandb:
                        wandb_images[f"valid/val_step_{val_step}_logit_pred_image_{i}"] = wandb.Image(logit_fig, caption=f"Validation Step {val_step} Final Logit Predictions - Image {i}")
                        print(f"Epoch {epoch} - Logit image {i} logged to W&B.")

            # Log all images at once for the same epoch
            if worker_args.wandb and wandb_images:
                wandb_images["epoch"] = epoch
                wandb.log(wandb_images, step=epoch)
                print(f"Epoch {epoch} - All {len(wandb_images)-1} images logged to W&B together.")

            # Clean up copies
            del ar_point_prompts_copy, tc_point_prompts_copy, ar_bbox_prompts_copy, tc_bbox_prompts_copy
            del masks_gt_copy, tc_masks_copy, ar_masks_copy
            if interm_ar_masks_copy:
                del interm_ar_masks_copy, interm_tc_masks_copy
            del final_logit_copy
            torch.cuda.empty_cache()
            
        # Update metrics - Final predictions
        tc_metrics.update(tc_masks, masks_tc_gts, batch['index_name'])
        ar_metrics.update(ar_masks, masks_ar_gts, batch['index_name'])
        
        # # Update metrics - Intermediate predictions
        # if interm_ar_masks and interm_tc_masks:
        #     interm_ar_metrics.update(interm_ar_masks, masks_ar_gts, batch['index_name'])
        #     interm_tc_metrics.update(interm_tc_masks, masks_tc_gts, batch['index_name'])
        
        # Update metrics - Final logit (multiclass)
        final_logit_metrics.update(final_logit_pred_list, gt_masks_list, batch['index_name'])
        
        # Update progress bar
        valid_pbar.update(1)
        str_step_info = "Epoch: {epoch}/{epochs:4}.".format(
            epoch=epoch, epochs=max_epoch_num
        )
        valid_pbar.set_postfix_str(str_step_info)
    
    # Compute metrics - Final predictions
    ar_metrict_dict, _ = ar_metrics.compute()
    tc_metric_dict, _ = tc_metrics.compute()
    
    # # Compute metrics - Intermediate predictions
    # interm_ar_dict, _ = interm_ar_metrics.compute()
    # interm_tc_dict, _ = interm_tc_metrics.compute()
    
    # Compute metrics - Final logit
    final_logit_dict, _ = final_logit_metrics.compute()
    
    # Extract final prediction metrics
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
    
    # # Extract intermediate prediction metrics
    # interm_miou_ar = interm_ar_dict['Mean Foreground IoU']
    # interm_mean_acc_ar = interm_ar_dict['Mean Acc']
    # interm_overall_acc_ar = interm_ar_dict['Overall Acc']
    
    # interm_miou_tc = interm_tc_dict['Mean Foreground IoU']
    # interm_mean_acc_tc = interm_tc_dict['Mean Acc']
    # interm_overall_acc_tc = interm_tc_dict['Overall Acc']
    
    # Extract final logit metrics
    logit_mean_iou = final_logit_dict['Mean IoU']
    logit_mean_acc = final_logit_dict['Mean Acc']
    logit_overall_acc = final_logit_dict['Overall Acc']
    
    # Also extract per-class IoUs from final_logit_metrics (Background, TC, AR)
    # These keys are added by StreamSegMetrics.compute() as '{class_name} IoU'
    logit_bg_iou = final_logit_dict.get('Background IoU', None)
    logit_tc_iou = final_logit_dict.get('TC IoU', None)
    logit_ar_iou = final_logit_dict.get('AR IoU', None)
    
    # Reset metrics for next epoch
    ar_metrics.reset()
    tc_metrics.reset()
    interm_ar_metrics.reset()
    interm_tc_metrics.reset()
    final_logit_metrics.reset()
     
    # Log metrics to wandb
    if worker_args.wandb:
        wandb.log({
            # Final predictions
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
            
            # # Intermediate predictions
            # "valid/interm_miou_ar": interm_miou_ar,
            # "valid/interm_miou_tc": interm_miou_tc,
            # "valid/interm_mean_acc_ar": interm_mean_acc_ar,
            # "valid/interm_mean_acc_tc": interm_mean_acc_tc,
            # "valid/interm_overall_acc_ar": interm_overall_acc_ar,
            # "valid/interm_overall_acc_tc": interm_overall_acc_tc,
            
            # Final logit (multiclass)
            "valid/logit_mean_iou": logit_mean_iou,
            "valid/logit_mean_acc": logit_mean_acc,
            "valid/logit_overall_acc": logit_overall_acc,
            # Per-class IoU for final logits
            "valid/logit_bg_iou": logit_bg_iou,
            "valid/logit_tc_iou": logit_tc_iou,
            "valid/logit_ar_iou": logit_ar_iou,
             
            "epoch": epoch,
        }, step=epoch)
    
    valid_pbar.close()
    return miou_tc, miou_ar, logit_mean_iou, logit_tc_iou, logit_ar_iou

#-----------------------------------------------------------
# DATA
#-----------------------------------------------------------
def set_up_dataset(worker_args):
    dataset_dir = worker_args.data_dir
    train_bs = worker_args.train_bs 
    val_bs = worker_args.val_bs
    gradient_accumulation_steps = worker_args.gradient_accumulation_steps
    
    
    train_dataset = ClimateDataset(
        data_dir=dataset_dir, train_flag=True, shot_num=worker_args.shot_num,
        augmented=worker_args.augmented, generate_prompt=True
    )
    val_dataset = ClimateDataset(data_dir=dataset_dir, train_flag=False, augmented=False, generate_prompt=True)
    train_collate_fn = train_dataset.collate_fn
    val_collate_fn = val_dataset.collate_fn
    # Debugging mode - use smaller dataset and fewer epochs
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
        worker_args.valid_per_epochs = 2
        print(f"Debug mode: Setting max_epoch_num to {max_epoch_num} and valid_per_epochs to {worker_args.valid_per_epochs}")
        
    # Adjust batch size for gradient accumulation
    actual_train_bs = train_bs // gradient_accumulation_steps
    if actual_train_bs < 1:
        actual_train_bs = 1
        print(f"Warning: gradient_accumulation_steps ({gradient_accumulation_steps}) is larger than train_bs ({train_bs}). Setting actual batch size to 1.")
    
    effective_batch_size = actual_train_bs * gradient_accumulation_steps
    
    print(f"Effective batch size: {effective_batch_size} (actual_bs: {actual_train_bs}, accumulation: {gradient_accumulation_steps})")
        
    train_workers, val_workers = 4, 2
    
    sampler = None
    
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
    
    return train_dataloader, val_dataloader

    
#-----------------------------------------------------------
# MODELS
#-----------------------------------------------------------
def set_up_model(worker_args, device):
    climatesam = ClimateSAM(
        model_type=worker_args.sam_type, 
        mlp_ratio=worker_args.image_encoder_mlp_ratio,
        enable_wandb_logging=getattr(worker_args, 'debugging', False)  # Only log if debugging=True
    ).to(device)
    
    # Load pretrained weights
    image_encoder_path = os.path.join(worker_args.exp_dir, f"{worker_args.encoder_weights_name}.pth") 
    if not os.path.exists(image_encoder_path):
        raise FileNotFoundError(f"Pretrained weights not found at {image_encoder_path}. Please check the path and try again.")
    phase_1_checkpoint = torch.load(image_encoder_path, map_location=device)
    print(f"Pretrained weights from phase 1 loaded from {image_encoder_path}")
    # ENCODRER WEIGHTS
    if 'image_encoder' not in phase_1_checkpoint:
        raise ValueError(f"Image encoder weights not found in checkpoint.")
    else:
        climatesam.image_encoder.load_state_dict(phase_1_checkpoint['image_encoder'])
        print(f"Image encoder weights loaded from {image_encoder_path}")
        
    # MASK DECODER WEIGHTS
    if 'mask_decoder' not in phase_1_checkpoint:
        raise ValueError(f"Mask decoder weights not found in checkpoint.")
    else:
        climatesam.mask_decoder.load_state_dict(phase_1_checkpoint['mask_decoder'])
        print(f"Mask decoder weights loaded from {image_encoder_path}")
    
    # INPUT ADAPTER WEIGHTS
    if 'input_adapter' not in phase_1_checkpoint:
        raise ValueError(f"Input adapter weights not found in checkpoint.")
    else:
        climatesam.input_adapter.load_state_dict(phase_1_checkpoint['input_adapter'])
        print(f"Input adapter weights loaded from {image_encoder_path}")
            
    ###################################################
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
        'vit_b': 768,
        'vit_l': 1024,
        'vit_h': 1280
    }
    
    prompt_generator = PromptGenerator(
        in_channels=in_channels[worker_args.sam_type],
        fused_channels=worker_args.fuse_channels,
        num_features=num_features_map[worker_args.sam_type],
        features_per_block=features_per_block[worker_args.sam_type]
    ).to(device)
    

    
    for params in climatesam.parameters():
        params.requires_grad = False
    # for params in climatesam.image_encoder.parameters():
    #     params.requires_grad = True
    # for params in climatesam.mask_decoder.parameters():
    #     params.requires_grad = True
        
    for params in prompt_generator.parameters():
        params.requires_grad = True
        
    if worker_args.load_pretrained:
        if worker_args.pretrained_name is not None:
            generator_path = os.path.join(worker_args.exp_dir, worker_args.pretrained_name)
        else:
            best_weights_dir = os.path.join(worker_args.exp_dir, 'best_weights')
            generator_path = os.path.join(best_weights_dir, f"best_generator_token_cg_{worker_args.sam_type}_{worker_args.fuse_channels}_{worker_args.run_name}.pth")
            
        if not os.path.exists(generator_path):
            print(f"Pretrained weights for prompt generator not found at {generator_path}. Starting training from scratch.")

        else:
            phase_2_checkpoint = torch.load(generator_path, map_location=device, weights_only=False)
            print(f"Pretrained weights for prompt generator from phase 2 loaded from {generator_path}")
            prompt_generator.load_state_dict(phase_2_checkpoint['prompt_generator'])
            print(f"Prompt generator weights loaded from {generator_path}")
        

    return climatesam, prompt_generator


#-----------------------------------------------------------
# MAIN WORKER
#-----------------------------------------------------------
def main_worker(worker_id, worker_args):
    set_randomness()
    max_epoch_num = worker_args.max_epoch_num 
    train_dataloader, val_dataloader = set_up_dataset(worker_args)

    if isinstance(worker_id, str):
        worker_id = int(worker_id)
    device, local_rank = setup_device_and_distributed(worker_id, worker_args)
    print(f"Worker {worker_id} initialized on device {device} with local_rank {local_rank}.")

    climatesam, prompt_generator = set_up_model(worker_args, device)
    optimizer, scheduler = setup_optimizer_and_scheduler_for_generator(climatesam, prompt_generator, worker_args)  
    
    prompt_maker = PromptMaker(prompt_type='point', positive_point_num=worker_args.positive_point_num, negative_point_num=worker_args.negative_point_num)
    
    best_miou_tc = 0
    best_miou_ar = 0
    best_miou_total = 0
    best_logit_miou = 0
    best_average_miou = 0
    ar_metrics = StreamSegMetrics(class_names=['Background', 'Foreground'])
    tc_metrics = StreamSegMetrics(class_names=['Background', 'Foreground'])
    
    scaler = torch.amp.GradScaler('cuda') 
    print(f"Validation will be performed every {worker_args.valid_per_epochs} epochs.")
    
    # Set gradient accumulation steps
    gradient_accumulation_steps = getattr(worker_args, 'gradient_accumulation_steps', 1)
    # Print parameter counts for each module (only from main process)
    
    print("Model parameter breakdown:")
    def _print_param_stats(module, phase):
        for n, c in module.named_children():
            total = sum(p.numel() for p in c.parameters())
            trainable = sum(p.numel() for p in c.parameters() if p.requires_grad)
            if total > 0:
                print(f"{n.upper():<25} | train={str(c.training):<5} | {trainable:>9,}/{total:>12,} ({100*trainable/total:>5.2f}%)")
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"Phase {phase}: trainable = {trainable:,} / {total:,}\n")

    _print_param_stats(climatesam, 1)
    _print_param_stats(prompt_generator, 2)
    
    # Training loop
    for epoch in range(1, max_epoch_num + 1):
        
        # Validation
        if epoch % worker_args.valid_per_epochs == 1 or epoch == max_epoch_num:
            if worker_args.load_pretrained or epoch > 1: 
                miou_tc, miou_ar, logit_mean_iou, logit_tc_iou, logit_ar_iou = validate_one_epoch(
                    epoch, val_dataloader, ar_metrics, tc_metrics, 
                    climatesam, prompt_generator, prompt_maker, device, 
                    max_epoch_num, worker_args
                )
                average_miou = (miou_tc + miou_ar) / 2
                print(f"Epoch {epoch} - mIoU TC: {miou_tc:.2%}, mIoU AR: {miou_ar:.2%}, Logit mIoU: {logit_mean_iou:.2%}, Logit TC IoU: {logit_tc_iou:.2%}, Logit AR IoU: {logit_ar_iou:.2%}")

                if miou_tc > best_miou_tc:
                    best_miou_tc = miou_tc
                    print(f'Best mIoU TC has been updated to {best_miou_tc:.2%}!')
                    
                if miou_ar > best_miou_ar:
                    best_miou_ar = miou_ar
                    print(f'Best mIoU AR has been updated to {best_miou_ar:.2%}!')
                    
                if best_average_miou < average_miou:
                    best_average_miou = average_miou
                    print(f'Best Average mIoU has been updated to {best_average_miou:.2%}!')
                    # Save best model (including all components)
                    if worker_args.save_model and epoch > 4:
                        # Create best_weights directory if it doesn't exist
                        best_weights_dir = os.path.join(worker_args.exp_dir, 'best_weights')
                        os.makedirs(best_weights_dir, exist_ok=True)

                        save_path = os.path.join(best_weights_dir, f"best_generator_token_cg_{worker_args.sam_type}_{worker_args.fuse_channels}_{worker_args.run_name}.pth")
                        complete_model_weights = {
                            # 'image_encoder': climatesam.image_encoder.state_dict(),
                            # 'mask_decoder': climatesam.mask_decoder.state_dict(),
                            'prompt_generator': prompt_generator.state_dict()
                            # 'epoch': epoch,
                            # 'best_miou_tc': best_miou_tc,
                            # 'best_miou_ar': best_miou_ar,
                            # 'best_miou_total': best_miou_total,
                            # 'best_logit_miou': best_logit_miou,
                        }
                        
                        
                                
                        torch.save(complete_model_weights, save_path)
                        print(f"Complete model weights saved to {save_path}")
                        
                        if worker_args.wandb:
                            wandb.save(save_path)
                            print(f"Complete model weights saved to wandb: {save_path}")
                
            # if (miou_tc + miou_ar) / 2 > best_miou_total:
            #     best_miou_total = (miou_tc + miou_ar) / 2
            #     print(f'Best mIoU Total has been updated to {best_miou_total:.2%}!')
            #     # Save best model (including all components)
            #     if worker_args.save_model and epoch > 4:
            #         # Create best_weights directory if it doesn't exist
            #         best_weights_dir = os.path.join(worker_args.exp_dir, 'best_weights')
            #         os.makedirs(best_weights_dir, exist_ok=True)

            #         save_path = os.path.join(best_weights_dir, f"best_model_sam_type_{worker_args.sam_type}.pth")
            #         complete_model_weights = {
            #             'image_encoder': climatesam.image_encoder.state_dict(),
            #             'mask_decoder': climatesam.mask_decoder.state_dict(),
            #             # 'prompt_generator': prompt_generator.state_dict(),
            #             'epoch': epoch,
            #             'best_miou_tc': best_miou_tc,
            #             'best_miou_ar': best_miou_ar,
            #             'best_miou_total': best_miou_total,
            #             # 'best_logit_miou': best_logit_miou,
            #         }
                    
                    
                            
            #         torch.save(complete_model_weights, save_path)
            #         print(f"Complete model weights saved to {save_path}")
                    
            #         if worker_args.wandb:
            #             wandb.save(save_path)
            #             print(f"Complete model weights saved to wandb: {save_path}")
                
           
                
                
            
        
        # Training
        train_one_epoch(
            epoch = epoch, train_dataloader=train_dataloader, climatesam=climatesam, prompter=prompt_generator,
            prompt_maker=prompt_maker, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            device=device, max_epoch_num=max_epoch_num, worker_args=worker_args,
            gradient_accumulation_steps=gradient_accumulation_steps, local_rank=local_rank
            
        )
    
    print(f"Training completed!")
    print(f"Best mIoU TC: {best_miou_tc:.2%}")
    print(f"Best mIoU AR: {best_miou_ar:.2%}")
    print(f"Best mIoU Total: {best_miou_total:.2%}")

        
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


def monitor_gpu_memory(step_name=""):
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / 1024**3
        reserved_gb = torch.cuda.memory_reserved() / 1024**3
        print(f"{step_name} - Allocated: {allocated_gb:.2f}GB, Reserved: {reserved_gb:.2f}GB")