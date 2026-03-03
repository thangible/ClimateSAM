from train import worker_init_fn, main_worker, setup_device_and_distributed, ClimateDataset, ClimateSAM, setup_optimizer_and_scheduler
import random
import numpy as np
import torch
import os
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from functools import partial
from torch.utils.data import DataLoader
from train_util import batch_to_cuda, get_idle_gpu, get_idle_port, set_randomness, calculate_dice_loss, calculate_focal_loss, plot_with_projection
from tqdm import tqdm
from contextlib import nullcontext

from climatesam import ClimateSAM
from dataset.climatenet import ClimateDataset
from evaluator import StreamSegMetrics
import wandb
import configargparse

import wandb


def parse():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, default='input_config_test', help='config file path')
    parser.add_argument(
        '--exp_dir', default='./exp', type=str,
        help="The directory to save the best checkpoint file. Default to be ./exp"
    )
    parser.add_argument(
        '--data_dir', default='./data', type=str,
        help="The directory that the datasets are placed. Default to be ./data"
    )
    parser.add_argument(
        '--num_workers', default=None, type=int,
        help="The num_workers argument used for the training and validation dataloaders. "
             "Default to be 1 for one-shot and 4 for 16- and full-shot."
    )
    parser.add_argument(
        '--train_bs', default= 16, type=int,
        help="The batch size for the training dataloader. Default to be 1 for one-shot and 4 for 16- and full-shot."
    )
    parser.add_argument(
        '--val_bs', default=None, type=int,
        help="The batch size for the validation dataloader. Default to be 1 for one-shot and 4 for 16- and full-shot."
    )

    parser.add_argument(
        '--shot_num', default=None, type=int, choices=[1, 16],
        help="The number of your target setting. For one-shot please give --shot_num 1. "
             "For 16-shot please give --shot_num 16. For full-shot please leave it blank. "
             "Default to be full-shot."
    )
    parser.add_argument(
        '--sam_type', default='vit_l', type=str, choices=['vit_b', 'vit_l', 'vit_h'],
        help='The type of the backbone SAM model. Default to be vit_l.'
    )
    
    parser.add_argument(
        '--max_epoch_num', default=50, type=int,
        help="The maximum number of epochs for training. Default is 50."
    )
    
    parser.add_argument(
        '--lr', default=1e-3, type=float,
        help="Learning rate for the optimizer. Default is 1e-3."
    )
    parser.add_argument(
        '--weight_decay', default=1e-4, type=float,
        help="Weight decay for the optimizer. Default is 1e-4."
    )
    parser.add_argument(
        '--save_model', action='store_true',
        help="Flag to save the best model. Default is False."
    )
    parser.add_argument(
        '--valid_per_epochs', default=1, type=int,
        help="Validation frequency in terms of epochs. Default is 1."
    )
    
    parser.add_argument(
        '--wandb', action='store_true',
        help="Flag to enable Weights & Biases logging. Default is False."
    )
    parser.add_argument(
        '--project_name', type=str, default="climate-sam",
        help="Project name for Weights & Biases logging."
    )
    parser.add_argument(
        '--run_name', type=str,
        help="Run name for Weights & Biases logging."
    )
    parser.add_argument(
        '--debugging', action='store_true',
        help="Flag to enable debugging mode. Default is False."
    )
    parser.add_argument(
        '--load_pretrained', action='store_true',
        help="Flag to load a pretrained model. Default is False."
    )

    parser.add_argument(
        '--pretrained_name', type=str,
        help="Name of the pretrained model to load."
    )
    
    parser.add_argument(
        '--bce_weight_ar', default=10, type=float,
        help="Weight for the BCE loss for AR. Default is 1.0."
    )
    
    parser.add_argument(
        '--bce_weight_tc', default=200, type=float,
        help="Weight for the BCE loss for TC. Default is 1.0."
    )
    
    parser.add_argument(
        '--gamma_ar', default=2, type=float,
        help="Gamma parameter for the Focal loss for AR. Default is 2.0."
    )
    
    parser.add_argument(
        '--gamma_tc', default=2, type=float,
        help="Gamma parameter for the Focal loss for TC. Default is 2.0."
    )
    
    parser.add_argument(
        '--alpha_ar', default=0.9, type=float,
        help="Alpha parameter for the Focal loss for AR. Default is 0.25."
    )
    
    parser.add_argument(
        '--alpha_tc', default=0.98, type=float,
        help="Alpha parameter for the Focal loss for TC. Default is 0.25."
    )
    
    parser.add_argument(
        '--phase', default='1', type=int, choices=[1, 2, 3, 4],
        help="Phase 1 for image_encoder, Phase 2 for prompt generator, phase 3 for input adapter, phase 4 for all"
    )
    
    parser.add_argument(
        '--image_encoder_mlp_ratio', default=0.25, type=float,
        help="MLP ratio for the image encoder. Default is 0.25."
    )
    
    args = parser.parse_args()

    return args

def main_worker(worker_id, worker_args):
    set_randomness()
    max_epoch_num = worker_args.max_epoch_num 
    if isinstance(worker_id, str):
        worker_id = int(worker_id)
    device, local_rank = setup_device_and_distributed(worker_id, worker_args)
    print(f"Worker {worker_id} initialized on device {device} with local rank {local_rank}.")
    
    # PREPARE DATASET
    dataset_dir = worker_args.data_dir
    train_dataset = ClimateDataset(
        data_dir=dataset_dir, train_flag=True, shot_num=worker_args.shot_num,
        transforms=None
    )
    val_dataset = ClimateDataset(data_dir=dataset_dir, train_flag=False)
    
    # DataLoader
    train_bs = worker_args.train_bs if worker_args.train_bs else (1 if worker_args.shot_num == 1 else 4)
    val_bs = worker_args.val_bs if worker_args.val_bs else 2
    train_workers, val_workers = 1 if worker_args.shot_num == 1 else 4, 2
    if worker_args.num_workers is not None:
        train_workers, val_workers = worker_args.num_workers, worker_args.num_workers
        
    sampler = None
    if torch.distributed.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_bs = int(train_bs / torch.distributed.get_world_size())
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=train_bs, shuffle=sampler is None, num_workers=train_workers,
        sampler=sampler, drop_last=False, collate_fn=train_dataset.collate_fn,
        worker_init_fn=partial(worker_init_fn, base_seed=3407)
    )
    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=val_bs, shuffle=False, num_workers=val_workers,
        drop_last=False, collate_fn=val_dataset.collate_fn
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
    
    scaler = torch.cuda.amp.GradScaler() 
    print(f"Validation will be performed every {worker_args.valid_per_epochs} epochs.")
    model.train(mode = True, phase = worker_args.phase, verbose=True)
    
    model.eval()
    epoch = 1
    valid_pbar = tqdm(total=len(val_dataloader), desc='valid', leave=False)
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
            # LOG
            if val_step == 0:
                imges = [images[i].cpu().numpy() for i in range(len(images))]
                masks_ar = [ar_masks[i].cpu().numpy() for i in range(len(ar_masks))]
                masks_tc = [tc_masks[i].cpu().numpy() for i in range(len(tc_masks))]
                masks_ar_gt = [masks_ar_gts[i].cpu().numpy() for i in range(len(masks_ar_gts))]
                masks_tc_gt = [masks_tc_gts[i].cpu().numpy() for i in range(len(masks_tc_gts))]
                for i in range(len(imges)):
                    save_path=os.path.join(worker_args.exp_dir, worker_args.run_name, 'eval_imgs', f"epoch_{epoch}_step_{val_step}_image_{i}.png")
                    
                    plot, titel = plot_with_projection(imges[i], masks_ar[i], masks_tc[i], masks_ar_gt[i], masks_tc_gt[i], save_path = save_path, epoch=epoch)
                
                    print(f"Epoch {epoch}- Image {i} saved.")

                del imges, masks_ar, masks_tc, masks_ar_gt, masks_tc_gt
                torch.cuda.empty_cache()
            

            
            # CAL
            merge_tc_masks = []
            merge_ar_masks = []
            for mask in tc_masks:
                combined_mask = torch.any(mask.bool(), dim=0).float()
                merge_tc_masks.append(combined_mask)
            for mask in ar_masks:
                combined_mask = torch.any(mask.bool(), dim=0).float()
                merge_ar_masks.append(combined_mask)
                
            ar_metrics.update(merge_tc_masks, masks_ar_gts,  batch['index_name'])
            tc_metrics.update(merge_ar_masks, masks_tc_gts,  batch['index_name'])
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
            
            
            
if __name__ == '__main__':
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
    else:
        # initialize multiprocessing start method
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            try:
                mp.set_start_method('forkserver')
                print("Fail to initialize multiprocessing module by spawn method. "
                      "Use forkserver method instead. Please be careful about it.")
            except RuntimeError as e:
                raise RuntimeError(
                    "Your server supports neither spawn or forkserver method as multiprocessing start methods. "
                    f"The error details are: {e}"
                )

        # dist_url is fixed to localhost here, so only single-node DDP is supported now.
        args.dist_url = "tcp://127.0.0.1" + f':{get_idle_port()}'
        # spawn one subprocess for each GPU
        mp.spawn(main_worker, nprocs=args.gpu_num, args=(args,))
