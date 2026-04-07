import sys
import os 

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(TRAIN_SCRIPT_DIR)

for p in (PROJECT_ROOT, TRAIN_SCRIPT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
import sys
import os 
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import copy
import wandb
import pandas as pd

from functools import partial
from torch.utils.data import DataLoader
from utility import batch_to_cuda, get_idle_gpu, get_idle_port, set_randomness, plot_with_projection, plot_mask_with_points_and_bbox, prompt_debug, setup_optimizer_and_scheduler, worker_init_fn, setup_device_and_distributed
from loss_function import ClimateLoss, compute_climate_loss
from tqdm import tqdm
from contextlib import nullcontext
from parser_config import parse

from model.climatesam import ClimateSAM as InfusedClimateSAM
from model.climatesam_concat import ClimateSAM as ConcatClimateSAM
from model.climatesam_single import ClimateSAM as SingleClimateSAM
from model.lora_sam import LoRAClimateSAMVanilla as SingleLoRASAM, LoRA_Sam, LoRALinear
from model.lora_sam_dual import LoRAClimateSAMVanilla as DualLoRASAM, LoRA_Sam, LoRALinear
from dataset.climatenet import ClimateDataset
from evaluator import StreamSegMetrics

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(TRAIN_SCRIPT_DIR)

for p in (PROJECT_ROOT, TRAIN_SCRIPT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# Initialize a single metric tracker with all three classes
val_metrics = StreamSegMetrics(class_names=['Background', 'TC', 'AR'])

def validate_one_epoch(epoch, val_dataloader, val_metrics, model, device, max_epoch_num, worker_args):

    model.eval()
    valid_pbar = tqdm(total=len(val_dataloader), desc='valid', leave=False)
    
    for val_step, batch in enumerate(val_dataloader):
        batch = batch_to_cuda(batch, device)
        
        images = model.set_infer_img(batch['input'])

        ar_point_prompts_copy = copy.deepcopy(batch['ar_point_prompts'])
        tc_point_prompts_copy = copy.deepcopy(batch['tc_point_prompts'])
        ar_bbox_prompts_copy = copy.deepcopy(batch['ar_bbox_prompts'])
        tc_bbox_prompts_copy = copy.deepcopy(batch['tc_bbox_prompts'])
        
        tc_masks, ar_masks = model.infer(
            ar_point_prompts=batch['ar_point_prompts'],
            tc_point_prompts=batch['tc_point_prompts'],
            ar_bbox_prompts=batch['ar_bbox_prompts'],
            tc_bbox_prompts=batch['tc_bbox_prompts']
        )
        
        masks_gt = batch['gt_mask']
        
        # Combine the separate predicted masks into a single multi-class prediction mask
        combined_preds = []
        for i in range(len(masks_gt)):
            # Create a background mask (0) matching the ground truth shape
            pred = torch.zeros_like(masks_gt[i])
            
            # Reshape the model outputs to match the ground truth (pred) shape exactly
            # This strips away any extra channel dimensions (e.g., [1, H, W] -> [H, W])
            tc_mask_matched = tc_masks[i].view_as(pred)
            ar_mask_matched = ar_masks[i].view_as(pred)
            
            # Assign class indices matching the ground truth logic
            # TC is class 1, AR is class 2
            pred[tc_mask_matched == 1] = 1
            pred[ar_mask_matched == 1] = 2
            
            combined_preds.append(pred)
        
        # Process shapes for the unified metrics
        for masks in [masks_gt, combined_preds]:
            for i in range(len(masks)):
                if len(masks[i].shape) == 2:
                    masks[i] = masks[i][None, None, :]
                if len(masks[i].shape) == 3:
                    masks[i] = masks[i][:, None, :]
                if len(masks[i].shape) != 4:
                    raise RuntimeError

        # LOG
        if val_step == 0:
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
                
                # Make sure the directory exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                fig = plot_mask_with_points_and_bbox(mask, ar_points, tc_points, ar_bbox, tc_bbox, tc_pred_mask, ar_pred_mask, radius=8, save_path=save_path, axis=True)
                
                if worker_args.wandb:
                    wandb_images[f"valid/val_step_{val_step}_image_{i}"] = wandb.Image(fig, caption=f"Validation Step {val_step} Image {i}")
                    print(f"Epoch {epoch} Image {i} prepared for logging.")
            
            if worker_args.wandb and wandb_images:
                wandb_images["epoch"] = epoch
                wandb.log(wandb_images, step=epoch)
                print(f"Epoch {epoch} All {len(wandb_images)-1} images logged to W&B together.")

            del ar_point_prompts_copy, tc_point_prompts_copy, ar_bbox_prompts_copy, tc_bbox_prompts_copy, masks_gt_copy, tc_masks_copy, ar_masks_copy
            torch.cuda.empty_cache()
            
        # Update the single unified metric tracker
        val_metrics.update(combined_preds, masks_gt, batch['index_name'])
        
        valid_pbar.update(1)
        str_step_info = "Epoch: {epoch}/{epochs:4}.".format(
            epoch=epoch, epochs=max_epoch_num
        )
        valid_pbar.set_postfix_str(str_step_info)
        
    # Extract comprehensive metrics
    metric_dict, _ = val_metrics.compute()
    
    # The dictionary automatically creates keys based on your class names
    iou_bg = metric_dict['Background IoU']
    iou_tc = metric_dict['TC IoU']
    iou_ar = metric_dict['AR IoU']
    
    mean_iou = metric_dict['Mean IoU']
    mean_foreground_iou = metric_dict['Mean Foreground IoU']
    mean_acc = metric_dict['Mean Acc']
    overall_acc = metric_dict['Overall Acc']
    freqw_acc = metric_dict['FreqW Acc']
    
    val_metrics.reset()
    
    if worker_args.wandb:
        wandb.log({
            "valid/iou_bg": iou_bg,
            "valid/iou_tc": iou_tc,
            "valid/iou_ar": iou_ar,
            "valid/mean_iou_all_classes": mean_iou,
            "valid/mean_foreground_iou": mean_foreground_iou,
            "valid/mean_acc": mean_acc,
            "valid/overall_acc": overall_acc,
            "valid/freqw_acc": freqw_acc,
            "epoch": epoch,
        }, step=epoch)
        
    return iou_tc, iou_ar, iou_bg, mean_iou, mean_foreground_iou

def set_up_dataloader(worker_args, prompt_config=None):
    val_dataset = ClimateDataset(data_dir=worker_args.data_dir, train_flag=False, augmented=False, generate_prompt=True, worker_args=worker_args, enlarge_ratio=worker_args.gt_prompt_enlarge_ratio, prompt_type=prompt_config)
    val_collate_fn = val_dataset.collate_fn
    
    g = torch.Generator()
    g.manual_seed(3407)
        
    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=worker_args.val_bs , shuffle=False, num_workers=worker_args.num_workers,
        drop_last=False, collate_fn=val_collate_fn, worker_init_fn=partial(worker_init_fn, base_seed=3407), generator=g, 
    )
    
    return val_dataloader

def set_up_model(config, device): 
    mode = config['mode']
    sam_type = config['sam_type']
    mlp_ratio = config['image_encoder_mlp_ratio']
    debugging = config.get('debugging', False)
    save_path = config['save_path']
    lora_r = config.get('lora_r', None)

    if mode == 'infused':
        model = InfusedClimateSAM(
            model_type=sam_type, 
            mlp_ratio=mlp_ratio,
            enable_wandb_logging=debugging 
        ).to(device=device)

    elif mode == 'concat':
        model = ConcatClimateSAM(
            model_type=sam_type, 
            mlp_ratio=mlp_ratio,
            enable_wandb_logging=debugging 
        ).to(device=device)
        
    elif mode == 'single':
        model = SingleClimateSAM(
            model_type=sam_type, 
            mlp_ratio=mlp_ratio,
            enable_wandb_logging=debugging 
        ).to(device=device)
        
    elif mode == 'lora_single':
        model = SingleLoRASAM(model_type=sam_type, r=lora_r, lora_layers=None, input_weights=None, use_prompt_generator=False, mlp_ratio=mlp_ratio, freeze_base=True, enable_wandb_logging=debugging).to(device=device)

    elif mode == 'lora_dual':
        model = DualLoRASAM(model_type=sam_type, r=lora_r, lora_layers=None, input_weights=None, use_prompt_generator=False, mlp_ratio=mlp_ratio, freeze_base=True, enable_wandb_logging=debugging).to(device=device)

    else:
        raise ValueError(f"Invalid mode type: {mode}")
    
    checkpoint = torch.load(save_path, map_location=device)
    print(f"mode: {mode}, Weight loaded from {save_path}")

    # IMAGE ENCODER
    if 'image_encoder' in checkpoint and hasattr(model, 'image_encoder'):
        model.image_encoder.load_state_dict(checkpoint['image_encoder'])
        print(f"Image encoder weights loaded from {save_path}")
    else:
        print("Image encoder weights not found in checkpoint or model does not have image_encoder.")
        
    ## INPUT ADAPTER
    if 'input_adapter' in checkpoint and hasattr(model, 'input_adapter'):
        model.input_adapter.load_state_dict(checkpoint['input_adapter'])
        print(f"Input adapter weights loaded from {save_path}")
    else:
        print("Input adapter weights not found in checkpoint or model does not have input_adapter.")
        
    # TC MASK DECODER
    if 'mask_decoder_tc' in checkpoint and hasattr(model, 'mask_decoder_tc'):
        model.mask_decoder_tc.load_state_dict(checkpoint['mask_decoder_tc'])
        print(f"TC mask decoder weights loaded from {save_path}")
    else:
        print("TC mask decoder weights not found in checkpoint or model does not have mask_decoder_tc.")

    ## AR MASK DECODER
    if 'mask_decoder_ar' in checkpoint and hasattr(model, 'mask_decoder_ar'):
        model.mask_decoder_ar.load_state_dict(checkpoint['mask_decoder_ar'])
        print(f"AR mask decoder weights loaded from {save_path}")
    else:
        print("AR mask decoder weights not found in checkpoint or model does not have mask_decoder_ar.")
        
    # MASK DECODER (for single decoder models)
    if 'mask_decoder' in checkpoint and hasattr(model, 'mask_decoder'):
        model.mask_decoder.load_state_dict(checkpoint['mask_decoder'])
        print(f"Mask decoder weights loaded from {save_path}")
    else:
        print("Mask decoder weights not found in checkpoint or model does not have mask_decoder.")
        
    model.eval()
    return model

def make_save_path(worker_args, name):
    save_dir = os.path.join(worker_args.exp_dir, 'best_weights')
    save_path = os.path.join(save_dir, f"{name}.pth")
    return save_path

def main_worker(worker_id, worker_args):
    set_randomness()
    max_epoch_num = worker_args.max_epoch_num 
    
    device, local_rank = setup_device_and_distributed(worker_id, worker_args)
    print(f"Worker {worker_id} initialized on device {device} with local_rank {local_rank}.")
    
    configs_linear = [
        {"mode": "infused", "save_path": make_save_path(worker_args, 'infused_vit_b_mlp1_linear'), "sam_type": 'vit_b', "image_encoder_mlp_ratio": 1},
        {"mode": "infused", "save_path": make_save_path(worker_args, 'infused_vit_b_mlp05_linear'), "sam_type": 'vit_b', "image_encoder_mlp_ratio": 0.5},
        {"mode": "concat", "save_path": make_save_path(worker_args, 'concat_vit_b_mlp1_linear'), "sam_type": 'vit_b', "image_encoder_mlp_ratio": 1},
        {"mode": "concat", "save_path": make_save_path(worker_args, 'concat_vit_b_mlp05_linear'), "sam_type": 'vit_b', "image_encoder_mlp_ratio": 0.5},
        {"mode": "single", "save_path": make_save_path(worker_args, 'single_vit_b_mlp1_linear'), "sam_type": 'vit_b', "image_encoder_mlp_ratio": 1},
        {"mode": "single", "save_path": make_save_path(worker_args, 'single_vit_b_mlp05_linear'), "sam_type": 'vit_b', "image_encoder_mlp_ratio": 0.5},
        {"mode": "lora_dual", "save_path": make_save_path(worker_args, 'lora_dual_vit_b_rank_32_linear'), "sam_type": 'vit_b', "image_encoder_mlp_ratio": 1, "lora_r": 32},
        {"mode": "lora_dual", "save_path": make_save_path(worker_args, 'lora_dual_vit_b_rank_64_linear'), "sam_type": 'vit_b', "image_encoder_mlp_ratio": 1, "lora_r": 64},
        {"mode": "lora_single", "save_path": make_save_path(worker_args, 'lora_single_vit_b_rank_32_linear'), "sam_type": 'vit_b', "image_encoder_mlp_ratio": 1, "lora_r": 32},
        {"mode": "lora_single", "save_path": make_save_path(worker_args, 'lora_single_vit_b_rank_64_linear'), "sam_type": 'vit_b', "image_encoder_mlp_ratio": 1, "lora_r": 64}
    ]
    
    configs_nonlinear = [
        {"mode": "infused", "save_path": make_save_path(worker_args, 'infused_vit_b_mlp1_nonlinear'), "sam_type": 'vit_b', "image_encoder_mlp_ratio": 1},
        {"mode": "infused", "save_path": make_save_path(worker_args, 'infused_vit_b_mlp05_nonlinear'), "sam_type": 'vit_b', "image_encoder_mlp_ratio": 0.5},
        {"mode": "concat", "save_path": make_save_path(worker_args, 'concat_vit_b_mlp1_nonlinear'), "sam_type": 'vit_b', "image_encoder_mlp_ratio": 1},
        {"mode": "concat", "save_path": make_save_path(worker_args, 'concat_vit_b_mlp05_nonlinear'), "sam_type": 'vit_b', "image_encoder_mlp_ratio": 0.5},
        {"mode": "single", "save_path": make_save_path(worker_args, 'single_vit_b_mlp1_nonlinear'), "sam_type": 'vit_b', "image_encoder_mlp_ratio": 1},
        {"mode": "single", "save_path": make_save_path(worker_args, 'single_vit_b_mlp05_nonlinear'), "sam_type": 'vit_b', "image_encoder_mlp_ratio": 0.5},
        {"mode": "lora_dual", "save_path": make_save_path(worker_args, 'lora_dual_vit_b_rank_32_nonlinear'), "sam_type": 'vit_b', "image_encoder_mlp_ratio": 1, "lora_r": 32},
        {"mode": "lora_dual", "save_path": make_save_path(worker_args, 'lora_dual_vit_b_rank_64_nonlinear'), "sam_type": 'vit_b', "image_encoder_mlp_ratio": 1, "lora_r": 64},
        {"mode": "lora_single", "save_path": make_save_path(worker_args, 'lora_single_vit_b_rank_32_nonlinear'), "sam_type": 'vit_b', "image_encoder_mlp_ratio": 1, "lora_r": 32},
        {"mode": "lora_single", "save_path": make_save_path(worker_args, 'lora_single_vit_b_rank_64_nonlinear'), "sam_type": 'vit_b', "image_encoder_mlp_ratio": 1, "lora_r": 64}
    ]
    
    prompt_configs = ['bbox', 'point', 'random']
    
    # Store all the metrics here
    metrics_data = []

    # Combine configurations to evaluate everything
    all_configs = configs_linear + configs_nonlinear
    
    for config in all_configs:
        print(f"\nValidating {config['mode']} model..., with MLP ratio {config['image_encoder_mlp_ratio']}, SAM type {config['sam_type']}, and LoRA rank {config.get('lora_r', 'N/A')}")
        
        # Only pass config and device, not the 'mode' string as a separate positional argument
        model = set_up_model(config, device)
        
        for prompt_config in prompt_configs:
            if prompt_config == 'bbox':
                print("Using bbox prompts for validation.")
                val_dataloader = set_up_dataloader(worker_args, prompt_config)
            elif prompt_config == 'point':
                print("Using point prompts for validation.")
                val_dataloader = set_up_dataloader(worker_args, prompt_config)
            elif prompt_config == 'random':
                print("Using random prompts for validation.")
                val_dataloader = set_up_dataloader(worker_args, prompt_config=None)
            else:
                raise ValueError(f"Invalid prompt configuration: {prompt_config}")
            
            iou_tc, iou_ar, iou_bg, mean_iou, mean_foreground_iou = validate_one_epoch(
                0, val_dataloader, val_metrics, model, device, max_epoch_num, worker_args
            )
            
            # Save the information to the metrics data list
            metrics_data.append({
                'Mode': config['mode'],
                'Config Group': 'Linear' if config in configs_linear else 'Non-linear',
                'Prompt Type': prompt_config,
                'MLP Ratio': config['image_encoder_mlp_ratio'],
                'LoRA Rank': config.get('lora_r', 'N/A'),
                'TC IoU': round(iou_tc, 4),
                'AR IoU': round(iou_ar, 4),
                'BG IoU': round(iou_bg, 4),
                'Mean IoU': round(mean_iou, 4),
                'Mean FG IoU': round(mean_foreground_iou, 4)
            })

    # Create the pandas DataFrame and print the final evaluation chart
    df_metrics = pd.DataFrame(metrics_data)
    print("\n--- FINAL EVALUATION METRICS ---")
    print(df_metrics.to_string(index=False))
        
if __name__ == '__main__':
    print("Starting training process...")
    args = parse()
    set_randomness()
    
    project_name = "climate-sam_evaluation"
    
    if args.wandb:
        wandb.init(project=project_name, name=args.run_name, config=vars(args))

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