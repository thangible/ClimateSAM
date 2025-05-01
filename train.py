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
    model.train(mode = True, phase = worker_args.phase)
    train_pbar = tqdm(total=len(train_dataloader), desc='train', leave=False) if local_rank == 0 else None
    for train_step, batch in enumerate(train_dataloader):
        batch = batch_to_cuda(batch, device)
        # print(f"batch['input'] shape: {batch['input'].shape}") 
        
        tc_mask, ar_mask, images = model(batch['input'],
                                ar_point_prompts = batch['ar_point_prompts'],
                                tc_point_prompts = batch['tc_point_prompts'], 
                                ar_bbox_prompts = batch['ar_bbox_prompts'], 
                                tc_bbox_prompts= batch['tc_bbox_prompts'],
                                ar_mask_prompts = batch['ar_mask_prompts'],
                                tc_mask_prompts = batch['tc_mask_prompts']
                                )
        
        masks_ar_gt = batch['ar_object_masks']
        masks_tc_gt = batch['tc_object_masks']
        
        
        # some processing to make sure the masks are in the right shape
        # for masks in [masks_ar_gt, masks_tc_gt, ar_mask, tc_mask]:
        #         for i in range(len(masks)):
        #             if len(masks[i].shape) == 2:
        #                 masks[i] = masks[i][None, None, :]
        #             if len(masks[i].shape) == 3:
        #                 masks[i] = masks[i][:, None, :]
        #             if len(masks[i].shape) != 4:
        #                 raise RuntimeError
                    
        bce_loss_list_tc, bce_loss_list_ar = [], []
        dice_loss_list_tc, dice_loss_list_ar = [], []
        
        for i in range(len(masks_ar_gt)):
            if masks_ar_gt[i] is not None:
                # ar
                pred_ar, label_ar = ar_mask[i], masks_ar_gt[i]
                label_ar = torch.where(torch.gt(label_ar, 0.), 1., 0.)
                pos_weight_ar = torch.tensor([worker_args.bce_weight_ar]).to(device)
                b_loss_ar = F.binary_cross_entropy_with_logits(pred_ar, label_ar.float(), 
                                                            pos_weight=pos_weight_ar)
                d_loss_ar = calculate_focal_loss(pred_ar, label_ar, gamma=worker_args.gamma_ar, alpha=worker_args.alpha_ar)
                bce_loss_list_ar.append(b_loss_ar)
                dice_loss_list_ar.append(d_loss_ar)
            
            if masks_tc_gt[i] is not None:
            # tc
                pred_tc, label_tc = tc_mask[i], masks_tc_gt[i]
                label_tc = torch.where(torch.gt(label_tc, 0.), 1., 0.)
                pos_weight_tc = torch.tensor([worker_args.bce_weight_tc]).to(device)
                b_loss_tc = F.binary_cross_entropy_with_logits(pred_tc, label_tc.float(), pos_weight=pos_weight_tc)
                d_loss_tc = calculate_focal_loss(pred_tc, label_tc, gamma=worker_args.gamma_tc, alpha=worker_args.alpha_tc)
                bce_loss_list_tc.append(b_loss_tc)
                dice_loss_list_tc.append(d_loss_tc)
    
        theta_tc = 5
        # bce loss
        bce_loss_ar = sum(bce_loss_list_ar) / len(bce_loss_list_ar) if len(bce_loss_list_ar) > 0 else torch.tensor(0).to(device)
        bce_loss_tc = sum(bce_loss_list_tc) / len(bce_loss_list_tc) if len(bce_loss_list_tc) > 0 else torch.tensor(0).to(device)
        bce_loss_tc = bce_loss_tc * theta_tc
        bce_loss = bce_loss_ar + bce_loss_tc
        
        # focal loss
        dice_loss_ar = sum(dice_loss_list_ar) / len(dice_loss_list_ar) if len(dice_loss_list_ar) > 0 else torch.tensor(0).to(device)
        dice_loss_tc = sum(dice_loss_list_tc) / len(dice_loss_list_tc) if len(dice_loss_list_tc) > 0 else torch.tensor(0).to(device)
        dice_loss_tc = dice_loss_tc * theta_tc
        dice_loss = dice_loss_ar + dice_loss_tc
        
        # total loss
        theta_total = 10
        dice_loss_ar = dice_loss_ar * theta_total
        dice_loss_tc = dice_loss_tc * theta_total
        total_loss_ar = bce_loss_ar + dice_loss_ar 
        total_loss_tc = bce_loss_tc + dice_loss_tc 
        total_loss = bce_loss + dice_loss 
        loss_dict = dict(
            total_loss=total_loss.clone().detach(),
            total_loss_ar=total_loss_ar.clone().detach(),
            total_loss_tc=total_loss_tc.clone().detach(),
            bce_loss_ar=bce_loss_ar.clone().detach(),
            bce_loss_tc=bce_loss_tc.clone().detach(),
            dice_loss_ar=dice_loss_ar.clone().detach(),
            dice_loss_tc=dice_loss_tc.clone().detach(),
            bce_loss=bce_loss.clone().detach(),
            dice_loss=dice_loss.clone().detach()
        )
        
        if worker_args.wandb:
            wandb.log({f"train/{key}": value.item() for key, value in loss_dict.items()}, step=epoch)
        
        backward_context = nullcontext
        if torch.distributed.is_initialized():
            backward_context = model.no_sync
        # with backward_context():
        #     total_loss.backward()
        # optimizer.step()
        # optimizer.zero_grad()
        
        with backward_context():
            scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
    
        # Optionally force garbage collection and empty CUDA cache
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        if torch.distributed.is_initialized():
                for key in loss_dict.keys():
                    if hasattr(loss_dict[key], 'detach'):
                        loss_dict[key] = loss_dict[key].detach()
                    torch.distributed.reduce(loss_dict[key], dst=0, op=torch.distributed.ReduceOp.SUM)
                    loss_dict[key] /= torch.distributed.get_world_size()

        if train_pbar:
            train_pbar.update(1)
            str_step_info = "Epoch: {epoch}/{epochs:4}. " \
                            "Loss: {total_loss:.4f}(total), {bce_loss:.4f}(bce), {dice_loss:.4f}(dice)".format(
                epoch=epoch, epochs=max_epoch_num,
                total_loss=loss_dict['total_loss'], bce_loss=loss_dict['bce_loss'], dice_loss=loss_dict['dice_loss']
            )
            train_pbar.set_postfix_str(str_step_info)
            
    scheduler.step()
    if train_pbar:
        train_pbar.clear()

def validate_one_epoch(epoch, val_dataloader, ar_metrics, tc_metrics, model, device, max_epoch_num, worker_args):
    model.eval()
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
            if val_step == 2:
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
                        wandb.log({f"valid/image_{i}": wandb.Image(plot, caption=titel), "epoch": epoch}, step = epoch)
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
    model.train(mode = True, phase = worker_args.phase)
    for epoch in range(1, max_epoch_num + 1):
        train_one_epoch(epoch, train_dataloader, model, optimizer, scheduler, device, local_rank, worker_args, max_epoch_num, scaler)
        if epoch % worker_args.valid_per_epochs == 1 and local_rank == 0:
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
