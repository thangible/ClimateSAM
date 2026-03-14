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
from utility import batch_to_cuda, get_idle_gpu, get_idle_port, set_randomness,  plot_with_projection, plot_mask_with_points_and_bbox, prompt_debug
from loss_function import ClimateLoss, compute_climate_loss
from tqdm import tqdm
from contextlib import nullcontext
from parser_config import parse
from climatesam import ClimateSAM
from dataset.climatenet import ClimateDataset
from evaluator import StreamSegMetrics
import copy
import wandb

# LoRA helpers (updated to match LoRA module names)
from model.lora_sam import LoRAClimateSAMVanilla, LoRA_Sam, LoRALinear


def worker_init_fn(worker_id: int, base_seed: int, same_worker_seed: bool = True):
    seed = base_seed if same_worker_seed else base_seed + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def setup_optimizer_and_scheduler(model, worker_args):
    lr = worker_args.lr if hasattr(worker_args, 'lr') else 1e-3
    weight_decay = worker_args.weight_decay if hasattr(worker_args, 'weight_decay') else 1e-4

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params=trainable, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=worker_args.max_epoch_num, eta_min=1e-5)
    return optimizer, scheduler


def setup_device_and_distributed(worker_id, worker_args):
    gpu_num = len(worker_args.used_gpu)
    world_size = os.environ['WORLD_SIZE'] if 'WORLD_SIZE' in os.environ.keys() else gpu_num
    base_rank = os.environ['RANK'] if 'RANK' in os.environ.keys() else 0
    local_rank = (int(base_rank) * gpu_num) + int(worker_id)
    if gpu_num > 1:
        dist.init_process_group(backend='nccl', init_method=worker_args.dist_url,
                                world_size=world_size, rank=local_rank)
    device = torch.device(f"cuda:{worker_id}")
    torch.cuda.set_device(device)
    return device, local_rank


def ensure_lora_trainable(model):
    """Freeze all params and unfreeze LoRA adapter params if present.

    Works with LoRAClimateSAMVanilla (base.lora.adapters) or LoRA_Sam instances.
    """
    m = model.module if hasattr(model, 'module') else model

    # resolve wrapper -> base if needed
    base = None
    if hasattr(m, 'base'):
        base = m.base
    elif hasattr(m, 'sam'):
        base = m
    else:
        base = m

    # freeze everything first
    for p in base.parameters():
        p.requires_grad = False

    # find adapter modules
    adapter_modules = []
    # check LoRAClimateSAMVanilla style
    if hasattr(base, 'lora') and hasattr(base.lora, 'adapters'):
        adapter_modules = list(base.lora.adapters)
    # check LoRA_Sam instance
    elif hasattr(base, 'adapters'):
        adapter_modules = list(base.adapters)

    # Unfreeze adapter params
    for mod in adapter_modules:
        for p in mod.parameters():
            p.requires_grad = True

    # As fallback, also unfreeze any parameter that has attribute 'is_lora' on its tensor
    for name, p in base.named_parameters():
        if hasattr(p, 'is_lora') and getattr(p, 'is_lora'):
            p.requires_grad = True

    # Ensure prompt encoder remains frozen unless explicitly part of adapters
    if hasattr(base, 'prompt_encoder'):
        for p in base.prompt_encoder.parameters():
            if not (hasattr(p, 'is_lora') and getattr(p, 'is_lora')):
                p.requires_grad = False


def train_one_epoch(epoch, train_dataloader, model, optimizer, scheduler, device, local_rank, worker_args, max_epoch_num, scaler):
    model.train(mode = True)
    # after calling model.train, ensure LoRA params are still trainable
    ensure_lora_trainable(model)

    gradient_accumulation_steps = getattr(worker_args, 'gradient_accumulation_steps', 1)
    effective_steps = len(train_dataloader) // gradient_accumulation_steps
    if len(train_dataloader) % gradient_accumulation_steps != 0:
        effective_steps += 1

    batch_pbar = tqdm(total=len(train_dataloader), desc=f'Epoch {epoch}/{max_epoch_num} - Batches', position=0, leave=True) if local_rank == 0 else None

    epoch_loss_dict = {}
    epoch_loss_count = 0

    for train_step, batch in enumerate(train_dataloader):
        batch = batch_to_cuda(batch, device)

        with torch.amp.autocast('cuda'):
            image_embeddings, interm_features, image_input, ori_img_size = model.encode_images(batch['input'])
            tc_mask, ar_mask, _ = model.forward(
                image_input=image_input,
                image_embeddings=image_embeddings,
                interm_embeddings=interm_features,
                ori_img_size=ori_img_size,
                ar_point_prompts=batch['ar_point_prompts'],
                tc_point_prompts=batch['tc_point_prompts'],
                ar_bbox_prompts=batch['ar_bbox_prompts'],
                tc_bbox_prompts=batch['tc_bbox_prompts'],
                ar_mask_prompts=batch['ar_mask_prompts'],
                tc_mask_prompts=batch['tc_mask_prompts']
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
            epoch_loss_dict.setdefault(key, 0.0)
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
            batch_pbar.set_postfix({'epoch': f"{epoch}/{max_epoch_num}", 'batch': f"{train_step+1}/{len(train_dataloader)}", 'loss': f"{total_loss.item():.4f}"})

        if (train_step + 1) % gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            epoch_loss_count += 1

    if len(train_dataloader) % gradient_accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        epoch_loss_count += 1

    if epoch_loss_count > 0:
        avg_epoch_losses = {k: epoch_loss_dict[k] / epoch_loss_count for k in epoch_loss_dict.keys()}
        if torch.distributed.is_initialized():
            for key in avg_epoch_losses.keys():
                tensor_loss = torch.tensor(avg_epoch_losses[key], device=device)
                torch.distributed.reduce(tensor_loss, dst=0, op=torch.distributed.ReduceOp.SUM)
                avg_epoch_losses[key] = (tensor_loss / torch.distributed.get_world_size()).item()
        if worker_args.wandb and local_rank == 0:
            log_dict = {f"train/{k}": avg_epoch_losses[k] for k in avg_epoch_losses.keys()}
            log_dict["epoch"] = epoch
            log_dict["learning_rate"] = scheduler.get_last_lr()[0]
            wandb.log(log_dict, step=epoch)

    if batch_pbar:
        batch_pbar.close()
    scheduler.step()


@torch.no_grad()
def validate_one_epoch(epoch, val_dataloader, ar_metrics, tc_metrics, model, device, max_epoch_num, worker_args):
    model.eval()
    valid_pbar = tqdm(total=len(val_dataloader), desc='valid', leave=False)

    for val_step, batch in enumerate(val_dataloader):
        batch = batch_to_cuda(batch, device)
        images = model.set_infer_img(batch['input'])

        tc_masks, ar_masks = model.infer(
            ar_point_prompts=batch['ar_point_prompts'],
            tc_point_prompts=batch['tc_point_prompts'],
            ar_bbox_prompts=batch['ar_bbox_prompts'],
            tc_bbox_prompts=batch['tc_bbox_prompts']
        )

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
                    raise RuntimeError

        if val_step == 0:
            wandb_images = {}
            masks_gt_copy = copy.deepcopy(masks_gt)
            tc_masks_copy = copy.deepcopy(tc_masks)
            ar_masks_copy = copy.deepcopy(ar_masks)
            for i in range(len(masks_gt)):
                mask = masks_gt_copy[i]
                ar_points = batch['ar_point_prompts'][i]
                tc_points = batch['tc_point_prompts'][i]
                ar_bbox = batch['ar_bbox_prompts'][i]
                tc_bbox = batch['tc_bbox_prompts'][i]
                tc_pred_mask = tc_masks_copy[i]
                ar_pred_mask = ar_masks_copy[i]
                save_path = os.path.join(worker_args.exp_dir, worker_args.run_name, 'images', f"epoch_{epoch}_step_{val_step}_image_{i}.png")
                fig = plot_mask_with_points_and_bbox(mask, ar_points, tc_points, ar_bbox, tc_bbox, tc_pred_mask, ar_pred_mask, radius=8, save_path=save_path, axis=True)
                if worker_args.wandb:
                    wandb_images[f"valid/val_step_{val_step}_image_{i}"] = wandb.Image(fig, caption=f"Validation Step {val_step} Image {i}")
            if worker_args.wandb and wandb_images:
                wandb_images["epoch"] = epoch
                wandb.log(wandb_images, step=epoch)
            del masks_gt_copy, tc_masks_copy, ar_masks_copy
            torch.cuda.empty_cache()

        tc_metrics.update(tc_masks, masks_tc_gts,  batch['index_name'])
        ar_metrics.update(ar_masks, masks_ar_gts,  batch['index_name'])
        valid_pbar.update(1)
        valid_pbar.set_postfix_str(f"Epoch: {epoch}/{max_epoch_num}.")

    ar_metrict_dict, _ = ar_metrics.compute()
    tc_metric_dict, _ = tc_metrics.compute()
    ar_metrics.reset()
    tc_metrics.reset()

    if worker_args.wandb:
        wandb.log({
            "valid/miou_ar": ar_metrict_dict['Mean Foreground IoU'],
            "valid/miou_tc": tc_metric_dict['Mean Foreground IoU'],
            "epoch": epoch,
        }, step=epoch)

    return tc_metric_dict['Mean Foreground IoU'], ar_metrict_dict['Mean Foreground IoU']


def main_worker(worker_id, worker_args):
    set_randomness()
    max_epoch_num = worker_args.max_epoch_num
    if isinstance(worker_id, str):
        worker_id = int(worker_id)
    device, local_rank = setup_device_and_distributed(worker_id, worker_args)

    dataset_dir = worker_args.data_dir
    train_dataset = ClimateDataset(data_dir=dataset_dir, train_flag=True, shot_num=worker_args.shot_num, augmented=worker_args.augmented, generate_prompt=True)
    val_dataset = ClimateDataset(data_dir=dataset_dir, train_flag=False, augmented=False, generate_prompt=True)

    train_collate_fn = train_dataset.collate_fn
    val_collate_fn = val_dataset.collate_fn

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

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=actual_train_bs, shuffle=sampler is None, num_workers=train_workers, sampler=sampler, drop_last=False, collate_fn=train_collate_fn, worker_init_fn=partial(worker_init_fn, base_seed=3407))
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=val_bs, shuffle=False, num_workers=val_workers, drop_last=False, collate_fn=val_collate_fn, worker_init_fn=partial(worker_init_fn, base_seed=3407))

    # SET UP MODEL
    base_model = ClimateSAM(model_type=worker_args.sam_type, mlp_ratio=worker_args.image_encoder_mlp_ratio, enable_wandb_logging=getattr(worker_args, 'debugging', False))

    # Wrap with LoRA
    lora_r = getattr(worker_args, 'lora_r', 8)
    lora_alpha = getattr(worker_args, 'lora_alpha', 32)
    freeze_base = getattr(worker_args, 'lora_freeze_base', True)

    # Use LoRAClimateSAMVanilla which applies LoRA to image_encoder and optionally freezes base
    model = LoRAClimateSAMVanilla(model_type=worker_args.sam_type, r=lora_r, lora_layers=None, input_weights=None, use_prompt_generator=False, mlp_ratio=worker_args.image_encoder_mlp_ratio, freeze_base=freeze_base, enable_wandb_logging=getattr(worker_args, 'debugging', False))
    model = model.to(device=device)

    if torch.distributed.is_initialized():
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        try:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        except Exception as e:
            print(f"DDP init error: {e}")
            model = model.to(device=device)

    optimizer, scheduler = setup_optimizer_and_scheduler(model, worker_args)

    best_miou_tc = 0
    best_miou_ar = 0
    ar_metrics = StreamSegMetrics(class_names=['Background', 'Foreground'])
    tc_metrics = StreamSegMetrics(class_names=['Background', 'Foreground'])

    scaler = torch.amp.GradScaler()
    model.train(mode=True, phase=worker_args.phase, verbose=True)
    # ensure LoRA params are trainable after train() call
    ensure_lora_trainable(model)

    for epoch in range(1, max_epoch_num + 1):
        if epoch % worker_args.valid_per_epochs == 1 or epoch == max_epoch_num:
            miou_tc, miou_ar = validate_one_epoch(epoch, val_dataloader, ar_metrics, tc_metrics, model, device, max_epoch_num, worker_args)
            print(f"Epoch {epoch} - mIoU TC: {miou_tc:.4f}, mIoU AR: {miou_ar:.4f}")

        train_one_epoch(epoch, train_dataloader, model, optimizer, scheduler, device, local_rank, worker_args, max_epoch_num, scaler)


if __name__ == '__main__':
    print("Starting LoRA training process...")
    args = parse()
    if hasattr(args, 'wandb') and args.wandb:
        wandb.init(project=args.project_name if hasattr(args, 'project_name') else 'climate-sam', name=getattr(args, 'run_name', None), config=vars(args))

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
