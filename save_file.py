import random
import numpy as np
import torch
import os
import torch.nn.functional as F
from functools import partial
from torch.utils.data import DataLoader
from train_util import batch_to_cuda, get_idle_gpu, get_idle_port, set_randomness,  plot_with_projection, plot_mask_with_points_and_bbox, prompt_debug, worker_init_fn, setup_optimizer_and_scheduler, setup_device, setup_device_and_distributed
from loss_function import ClimateLoss, compute_climate_loss
from tqdm import tqdm
from contextlib import nullcontext
from train_parser import parse
from climatesam import ClimateSAM
from dataset.climatenet import ClimateDataset
import copy
import wandb




@torch.no_grad()
def save_embeddings(model, dataloader, device, save_path):
    model.eval()
    for idx, batch in enumerate(dataloader):
        batch = batch_to_cuda(batch, device)
        imgs, img_features, interm_features = model.set_infer_img(batch['input'])
        if not os.path.exists('embeddings'):
            os.makedirs('embeddings')
        save_path = os.path.join('embeddings', f"image_embeddings_batch_{idx}.pth")
        torch.save({
            'imgs': imgs.cpu(),
            'img_features': img_features.cpu(),
            'interm_features': [feat.cpu() for feat in interm_features],
            'gt_mask': batch['gt_mask'],
            'index_name': batch['index_name']
        }, save_path)

@torch.no_grad()
def validate_one_epoch(epoch, val_dataloader, ar_metrics, tc_metrics, model, device, max_epoch_num, worker_args):

    # Example usage inside validation loop:
    # plot_mask_with_points(batch['gt_mask'][0], batch['tc_point_prompts'])
    model.eval()
    valid_pbar = tqdm(total=len(val_dataloader), desc='valid', leave=False)
    
    for val_step, batch in enumerate(val_dataloader):
        batch = batch_to_cuda(batch, device)
        
        # Set inference images once
        images = model.set_infer_img(batch['input'])

        ar_point_prompts_copy = copy.deepcopy(batch['ar_point_prompts'])
        tc_point_prompts_copy = copy.deepcopy(batch['tc_point_prompts'])
        ar_bbox_prompts_copy = copy.deepcopy(batch['ar_bbox_prompts'])
        tc_bbox_prompts_copy = copy.deepcopy(batch['tc_bbox_prompts'])

        # prompt_debug(batch, text=f"Validation Step {val_step}")
        
        # Perform inference with prompts
        tc_masks, ar_masks = model.infer(
            ar_point_prompts=batch['ar_point_prompts'],
            tc_point_prompts=batch['tc_point_prompts'],
            ar_bbox_prompts=batch['ar_bbox_prompts'],
            tc_bbox_prompts=batch['tc_bbox_prompts']
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
        augmented=True, generate_prompt=True
    )
    val_dataset = ClimateDataset(data_dir=dataset_dir, train_flag=False, augmented=False, generate_prompt=True)

    train_collate_fn = train_dataset.collate_fn
    val_collate_fn = val_dataset.collate_fn
    
    # DataLoader
    train_bs = worker_args.train_bs if worker_args.train_bs else (1 if worker_args.shot_num == 1 else 4)
    gradient_accumulation_steps = getattr(worker_args, 'gradient_accumulation_steps', 1)
    
    # Adjust batch size for gradient accumulation
    actual_train_bs = train_bs // gradient_accumulation_steps
    if actual_train_bs < 1:
        actual_train_bs = 1
        print(f"Warning: gradient_accumulation_steps ({gradient_accumulation_steps}) is larger than train_bs ({train_bs}). Setting actual batch size to 1.")
    
    effective_batch_size = actual_train_bs * gradient_accumulation_steps
    
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
    
    # SET UP MODEL - enable W&B logging only if debugging is True
    model = ClimateSAM(
        model_type=worker_args.sam_type, 
        mlp_ratio=worker_args.image_encoder_mlp_ratio,
        enable_wandb_logging=getattr(worker_args, 'debugging', False)  # Only log if debugging=True
    ).to(device=device)
    
    
    # Load pretrained weights
    if worker_args.load_pretrained:
        if worker_args.phase == 1:
            image_encoder_path = os.path.join(worker_args.exp_dir,'best_weights', f"phase_2_weights_best.pth")
            phase_1_checkpoint = torch.load(image_encoder_path, map_location=device)
            print(f"Pretrained weights from phase 1 loaded from {image_encoder_path}")
            model.image_encoder.load_state_dict(phase_1_checkpoint['image_encoder'])
            print(f"Image encoder weights loaded from {image_encoder_path}")
            model.mask_decoder.load_state_dict(phase_1_checkpoint['mask_decoder'])
            print(f"Mask decoder weights loaded from {image_encoder_path}")
    
            
    save_embeddings(model, train_dataloader, device, save_path='embeddings/train')
    save_embeddings(model, val_dataloader, device, save_path='embeddings/valid')

        
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

