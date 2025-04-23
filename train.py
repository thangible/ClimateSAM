import random
import numpy as np
import torch
import os
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from functools import partial
from torch.utils.data import DataLoader
from train_util import batch_to_cuda, get_idle_gpu, get_idle_port, set_randomness, calculate_dice_loss
from tqdm import tqdm
from contextlib import nullcontext
from train_parser import parse
from climatesam import ClimateSAM
from dataset.climatenet import ClimateDataset
from evaluator import StreamSegMetrics

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

    
def train_one_epoch(epoch, train_dataloader, model, optimizer, scheduler, device, local_rank, worker_args, max_epoch_num):
    train_pbar = tqdm(total=len(train_dataloader), desc='train', leave=False) if local_rank == 0 else None
    for train_step, batch in enumerate(train_dataloader):
        batch = batch_to_cuda(batch, device)
        print(f"batch['input'] shape: {batch['input'].shape}") 
        
        ar_mask, tc_mask = model(batch['input'])
        masks_gt = batch['gt_masks']
        masks_ar_gt = [ (mask == 2).to(torch.uint8) for mask in masks_gt ]
        masks_tc_gt = [ (mask == 1).to(torch.uint8) for mask in masks_gt ]
        
        # some processing to make sure the masks are in the right shape
        for masks in [masks_ar_gt, masks_tc_gt, ar_mask, tc_mask]:
                for i in range(len(masks)):
                    if len(masks[i].shape) == 2:
                        masks[i] = masks[i][None, None, :]
                    if len(masks[i].shape) == 3:
                        masks[i] = masks[i][:, None, :]
                    if len(masks[i].shape) != 4:
                        raise RuntimeError
                    
        bce_loss_list_tc, bce_loss_list_ar = [], []
        dice_loss_list_tc, dice_loss_list_ar = [], []
        focal_loss_list_tc, focal_loss_list_ar = [], []
        for i in range(len(masks_ar_gt)):
            # ar
            pred_ar, label_ar = ar_mask[i], masks_ar_gt[i]
            label_ar = torch.where(torch.gt(label_ar, 0.), 1., 0.)
            b_loss_ar = F.binary_cross_entropy_with_logits(pred_ar, label_ar.float())
            d_loss_ar = calculate_dice_loss(pred_ar, label_ar)
            # tc
            pred_tc, label_tc = tc_mask[i], masks_tc_gt[i]
            label_tc = torch.where(torch.gt(label_tc, 0.), 1., 0.)
            b_loss_tc = F.binary_cross_entropy_with_logits(pred_tc, label_tc.float())
            d_loss_tc = calculate_dice_loss(pred_tc, label_tc)
            # add the loss to the list
            bce_loss_list_ar.append(b_loss_ar)
            dice_loss_list_ar.append(d_loss_ar)
            bce_loss_list_tc.append(b_loss_tc)
            dice_loss_list_tc.append(d_loss_tc)
            
        bce_loss_ar = sum(bce_loss_list_ar) / len(bce_loss_list_ar)
        bce_loss_tc = sum(bce_loss_list_tc) / len(bce_loss_list_tc)
        bce_loss = bce_loss_ar + bce_loss_tc
        dice_loss_ar = sum(dice_loss_list_ar) / len(dice_loss_list_ar)
        dice_loss_tc = sum(dice_loss_list_tc) / len(dice_loss_list_tc)
        dice_loss = dice_loss_ar + dice_loss_tc
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
        
        backward_context = nullcontext
        if torch.distributed.is_initialized():
            backward_context = model.no_sync
        with backward_context():
            total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
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

        
def main_worker(worker_id, worker_args):
    set_randomness()
    max_epoch_num = worker_args.max_epoch_num 
    if isinstance(worker_id, str):
        worker_id = int(worker_id)
    device, local_rank = setup_device_and_distributed(worker_id, worker_args)
    
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
    model = ClimateSAM(model_type=worker_args.sam_type).to(device=device)
    if torch.distributed.is_initialized():
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        try:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
            )
        except Exception as e:
            print(f"Error initializing DistributedDataParallel: {e}")
            model = model.to(device=device)
            
    # Optimizer and scheduler
    optimizer, scheduler = setup_optimizer_and_scheduler(model, worker_args)
    best_miou = 0
    iou_eval  = StreamSegMetrics(class_names=['Background', 'Foreground'])
    
    for epoch in range(1, max_epoch_num + 1):
        train_one_epoch(epoch, train_dataloader, model, optimizer, scheduler, device, local_rank, worker_args, max_epoch_num)
        
    
    
    
    
if __name__ == '__main__':
    args = parse()
    
    if hasattr(args, 'wandb') and args.wandb:
        project_name = args.project_name if hasattr(args, 'project_name') else "climate-sam"
        run_name = args.run_name if hasattr(args, 'run_name') else None
        # wandb.init(project=project_name, name=run_name, config=vars(args))


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
