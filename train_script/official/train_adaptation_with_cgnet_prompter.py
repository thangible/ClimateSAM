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
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from functools import partial
from torch.utils.data import DataLoader
from utility import batch_to_cuda, get_idle_gpu, get_idle_port, set_randomness, plot_with_projection, plot_mask_with_points_and_bbox, prompt_debug, print_param_stats, setup_device_and_distributed
from loss_function import ClimateLoss, compute_climate_loss
from tqdm import tqdm
from contextlib import nullcontext
from parser_config import parse
from model.climatesam import ClimateSAM
from dataset.climatenet import ClimateDataset
from evaluator import StreamSegMetrics
import copy
import wandb

# Import the new BBox Prompter
from model.prompt.cgnet_bbox import CGNetBBoxPrompter 

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
    lr = worker_args.lr if hasattr(worker_args, 'lr') else 1e-3
    weight_decay = worker_args.weight_decay if hasattr(worker_args, 'weight_decay') else 1e-4

    all_trainable_params = list(p for p in model.parameters() if p.requires_grad) 

    optimizer = torch.optim.AdamW(
        params=all_trainable_params, lr=lr, weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=worker_args.max_epoch_num, eta_min=1e-5
    )
    return optimizer, scheduler

def train_one_epoch(epoch, train_dataloader, model, prompter, optimizer, scheduler, device, local_rank, worker_args, max_epoch_num, scaler):
    model.train(mode = True, phase = worker_args.phase, verbose = False)
    
    # Ensure prompter is frozen and in eval mode
    prompter.cgnet_model.eval()
    
    gradient_accumulation_steps = getattr(worker_args, 'gradient_accumulation_steps', 1)
    
    effective_steps = len(train_dataloader) // gradient_accumulation_steps
    if len(train_dataloader) % gradient_accumulation_steps != 0:
        effective_steps += 1
    
    if local_rank == 0:
        batch_pbar = tqdm(total=len(train_dataloader), desc=f'Epoch {epoch}/{max_epoch_num} - Batches', position=0, leave=True)
    else:
        batch_pbar = None
    
    step_count = 0 
    epoch_loss_dict = {}
    epoch_loss_count = 0
    
    for train_step, batch in enumerate(train_dataloader):
        batch = batch_to_cuda(batch, device)
        
        # 1. Generate BBox prompts directly using the frozen CGNetBBoxPrompter
        with torch.no_grad():
            features = batch['cgnet_input'].to(device=device, dtype=torch.float32)
            prompt_dict = prompter.get_prompts(features, conf_threshold=0.5, iou_threshold=0.4, enlarge_ratio=worker_args.gt_prompt_enlarge_ratio if hasattr(worker_args, 'gt_prompt_enlarge_ratio') else 0)
        
        # 2. Forward SAM and compute loss
        with torch.amp.autocast('cuda'):
            image_embeddings, interm_features, image_input, ori_img_size = model.encode_images(batch['input'])
            tc_mask, ar_mask, _ = model.forward(
                image_input=image_input,
                image_embeddings=image_embeddings,
                interm_embeddings=interm_features,
                ori_img_size=ori_img_size,
                ar_point_prompts=None,
                tc_point_prompts=None,
                ar_bbox_prompts=prompt_dict['ar_bbox_prompts'],
                tc_bbox_prompts=prompt_dict['tc_bbox_prompts'],
                ar_mask_prompts=None,
                tc_mask_prompts=None
            )
            
            masks_ar_gt = batch['ar_object_masks']
            masks_tc_gt = batch['tc_object_masks']
            
            loss_dict = compute_climate_loss(
                ar_masks=ar_mask,
                tc_masks=tc_mask,
                ar_masks_gt=masks_ar_gt,
                tc_masks_gt=masks_tc_gt,
                device=device,
                worker_args=worker_args
            )
        
        total_loss = loss_dict.pop('total_loss_for_backward')
        total_loss = total_loss / gradient_accumulation_steps
        
        for key, value in loss_dict.items():
            if key not in epoch_loss_dict:
                epoch_loss_dict[key] = 0
            epoch_loss_dict[key] += value.item() / gradient_accumulation_steps

        backward_context = nullcontext
        if torch.distributed.is_initialized():
            if (train_step + 1) % gradient_accumulation_steps != 0:
                backward_context = model.no_sync
            else:
                backward_context = nullcontext

        with backward_context():
            scaler.scale(total_loss).backward()
        
        if batch_pbar:
            batch_pbar.update(1)
            batch_pbar.set_postfix({
                'epoch': f"{epoch}/{max_epoch_num}",
                'loss': f"{total_loss.item() * gradient_accumulation_steps:.4f}"
            })
        
        if (train_step + 1) % gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            epoch_loss_count += 1
            
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
    
    scheduler.step()
    
    if epoch_loss_count > 0:
        avg_epoch_losses = {key: epoch_loss_dict[key] / epoch_loss_count for key in epoch_loss_dict.keys()}
        
        if torch.distributed.is_initialized():
            for key in avg_epoch_losses.keys():
                tensor_loss = torch.tensor(avg_epoch_losses[key], device=device)
                torch.distributed.reduce(tensor_loss, dst=0, op=torch.distributed.ReduceOp.SUM)
                avg_epoch_losses[key] = (tensor_loss / torch.distributed.get_world_size()).item()
        
        if worker_args.wandb and local_rank == 0:
            log_dict = {f"train/{key}": avg_epoch_losses[key] for key in avg_epoch_losses.keys()}
            log_dict["epoch"] = epoch
            log_dict["learning_rate"] = scheduler.get_last_lr()[0]
            wandb.log(log_dict, step=epoch)
            
    if batch_pbar:
        batch_pbar.close()


@torch.no_grad()
def validate_one_epoch(epoch, val_dataloader, ar_metrics, tc_metrics, model, prompter, device, worker_args):
    model.eval()
    prompter.cgnet_model.eval()
    
    valid_pbar = tqdm(total=len(val_dataloader), desc='valid', leave=False)
    
    for val_step, batch in enumerate(val_dataloader):
        batch = batch_to_cuda(batch, device)
        
        # 1. Generate BBox prompts directly using the frozen CGNetBBoxPrompter
        features = batch['cgnet_input'].to(device=device, dtype=torch.float32)
        prompt_dict = prompter.get_prompts(features, conf_threshold=0.5, iou_threshold=0.4, enlarge_ratio=0)
        
        # 2. Package for SAM
        combined_prompt_dict = {
            'ar_point_prompts': None,
            'tc_point_prompts': None,
            'ar_bbox_prompts': prompt_dict['ar_bbox_prompts'],
            'tc_bbox_prompts': prompt_dict['tc_bbox_prompts']
        }
        
        # 3. Set inference images
        images = model.set_infer_img(batch['input'])
        
        # 4. Perform inference with combined prompts
        tc_masks, ar_masks = model.infer(
            ar_point_prompts=combined_prompt_dict['ar_point_prompts'],
            tc_point_prompts=combined_prompt_dict['tc_point_prompts'],
            ar_bbox_prompts=combined_prompt_dict['ar_bbox_prompts'],
            tc_bbox_prompts=combined_prompt_dict['tc_bbox_prompts']
        )
        
        # 5. Extract ground truth masks
        masks_gt = batch['gt_mask']
        masks_ar_gts = [(mask == 2).to(torch.uint8) for mask in masks_gt]
        masks_tc_gts = [(mask == 1).to(torch.uint8) for mask in masks_gt]
        
        for masks in [masks_ar_gts, masks_tc_gts, ar_masks, tc_masks]:
            for i in range(len(masks)):
                if len(masks[i].shape) == 2:
                    masks[i] = masks[i][None, None, :]
                if len(masks[i].shape) == 3:
                    masks[i] = masks[i][:, None, :]
                if len(masks[i].shape) != 4:
                    raise RuntimeError(f"Unexpected mask shape: {masks[i].shape}")
        
        # 6. Update metrics
        tc_metrics.update(tc_masks, masks_tc_gts, batch['index_name'])
        ar_metrics.update(ar_masks, masks_ar_gts, batch['index_name'])
        valid_pbar.update(1)
        
    valid_pbar.close()
    
    ar_metric_dict, _ = ar_metrics.compute()
    tc_metric_dict, _ = tc_metrics.compute()
    
    miou_ar = ar_metric_dict['Mean Foreground IoU']
    miou_tc = tc_metric_dict['Mean Foreground IoU']
    mean_acc_ar = ar_metric_dict['Mean Acc']
    mean_acc_tc = tc_metric_dict['Mean Acc']
    
    ar_metrics.reset()
    tc_metrics.reset()
    
    if worker_args.wandb:
        wandb.log({
            "valid/miou_ar": miou_ar,
            "valid/miou_tc": miou_tc,
            "valid/mean_acc_ar": mean_acc_ar,
            "valid/mean_acc_tc": mean_acc_tc,
            "epoch": epoch,
        }, step = epoch)
        
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
        augmented=worker_args.augmented, generate_prompt=True, prompt_type='bbox', enlarge_ratio=[0,0]
    )
    val_dataset = ClimateDataset(data_dir=dataset_dir, train_flag=False, augmented=False, generate_prompt=True, prompt_type='bbox', enlarge_ratio=[0,0])
    
    train_collate_fn = train_dataset.collate_fn
    val_collate_fn = val_dataset.collate_fn

    if hasattr(worker_args, 'debugging') and worker_args.debugging:
        debug_size = getattr(worker_args, 'debug_size', 10)  
        indices = list(range(min(debug_size, len(train_dataset))))
        train_dataset = torch.utils.data.Subset(train_dataset, indices)

        debug_val_size = getattr(worker_args, 'debug_val_size', 5)  
        val_indices = list(range(min(debug_val_size, len(val_dataset))))
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
        
        max_epoch_num = 2
        worker_args.valid_per_epochs = 1
        
    
    # DataLoader setup
    train_bs = worker_args.train_bs if worker_args.train_bs else (1 if worker_args.shot_num == 1 else 4)
    gradient_accumulation_steps = getattr(worker_args, 'gradient_accumulation_steps', 1)
    
    actual_train_bs = train_bs // gradient_accumulation_steps
    if actual_train_bs < 1:
        actual_train_bs = 1
    
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

    # 1. Initialize and Freeze CGNet BBox Prompter
    pretrained_name = worker_args.pretrained_name if hasattr(worker_args, 'pretrained_name') else os.path.join(worker_args.exp_dir,'cgnet_bbox_weight.pth')
    cgnetprompter = CGNetBBoxPrompter(
        weights_path=pretrained_name, 
        device=device, 
        worker_args=worker_args,
        num_classes=2 
    )
    # Freeze the model entirely
    cgnetprompter.cgnet_model.eval()
    for param in cgnetprompter.cgnet_model.parameters():
        param.requires_grad = False
    print("CGNet BBox Prompter loaded and frozen.")

    # 2. Initialize ClimateSAM for Training
    model = ClimateSAM(
        model_type=worker_args.sam_type, 
        mlp_ratio=worker_args.image_encoder_mlp_ratio,
        enable_wandb_logging=getattr(worker_args, 'debugging', False)
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

    # Load pretrained weights for ClimateSAM if requested
    if getattr(worker_args, 'load_pretrained', False):
        weights_name = getattr(worker_args, 'encoder_weights_name', 'phase_1_weights')
        image_encoder_path = os.path.join(worker_args.exp_dir, f"{weights_name}.pth")
        
        if os.path.exists(image_encoder_path):
            print(f"Loading SAM weights from {image_encoder_path}")
            checkpoint = torch.load(image_encoder_path, map_location=device)
            if 'image_encoder' in checkpoint: 
                model.module.image_encoder.load_state_dict(checkpoint['image_encoder']) if hasattr(model, 'module') else model.image_encoder.load_state_dict(checkpoint['image_encoder'])
            if 'mask_decoder' in checkpoint: 
                model.module.mask_decoder.load_state_dict(checkpoint['mask_decoder']) if hasattr(model, 'module') else model.mask_decoder.load_state_dict(checkpoint['mask_decoder'])
            if 'input_adapter' in checkpoint: 
                model.module.input_adapter.load_state_dict(checkpoint['input_adapter']) if hasattr(model, 'module') else model.input_adapter.load_state_dict(checkpoint['input_adapter'])
        else:
            print(f"Warning: SAM weights not found at {image_encoder_path}. Using default initialization.")

    # 3. Setup Optimizer and Scheduler for ClimateSAM
    optimizer, scheduler = setup_optimizer_and_scheduler(model, worker_args)
    scaler = torch.amp.GradScaler('cuda')
    
    ar_metrics = StreamSegMetrics(class_names=['Background', 'Foreground'])
    tc_metrics = StreamSegMetrics(class_names=['Background', 'Foreground'])

    best_miou_tc = 0
    best_miou_ar = 0
    best_miou_total = 0

    print("Starting adaptation training for ClimateSAM guided by frozen CGNetBBoxPrompter...")
    model.train(mode=True, phase=worker_args.phase, verbose=True)
    
    for epoch in range(1, max_epoch_num + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        
        if epoch % getattr(worker_args, 'valid_per_epochs', 1) == 0 or epoch == max_epoch_num:
            if getattr(worker_args, 'load_pretrained', False) or epoch > 1: 
                miou_tc, miou_ar = validate_one_epoch(epoch, val_dataloader, ar_metrics, tc_metrics, model, cgnetprompter, device, worker_args)
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
                    if getattr(worker_args, 'save_model', True):
                        save_path = os.path.join(worker_args.exp_dir, f"adapted_sam_cgnet_{worker_args.sam_type}_{worker_args.run_name}.pth")
                        
                        actual_model = model.module if hasattr(model, 'module') else model
                        model_weights = {
                            'image_encoder': actual_model.image_encoder.state_dict(),
                            'mask_decoder': actual_model.mask_decoder.state_dict(),
                            'input_adapter': actual_model.input_adapter.state_dict(),
                        }
                        torch.save(model_weights, save_path)
                        print(f"ClimateSAM saved to {save_path}")
                        if worker_args.wandb:
                            wandb.save(save_path)
               
        train_one_epoch(epoch, train_dataloader, model, cgnetprompter, optimizer, scheduler, device, local_rank, worker_args, max_epoch_num, scaler)
        
if __name__ == '__main__':
    print("Starting training process...")
    args = parse()
    
    if hasattr(args, 'wandb') and args.wandb:
        project_name = args.project_name if hasattr(args, 'project_name') else "climate-sam-cgnet-adaptation"
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

    if len(args.used_gpu) == 1:
        main_worker(worker_id=0, worker_args=args)