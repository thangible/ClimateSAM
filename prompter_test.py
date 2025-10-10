import torch
from dataset.climatenet import ClimateDataset 
from torch.utils.data import DataLoader
from functools import partial
import random
import numpy as np
import os
from train_util import prompt_debug, plot_mask_with_points_and_bbox, batch_to_cuda
from train_parser import parse

def setup_device_and_distributed(worker_id):
    gpu_num = 1
    world_size = os.environ['WORLD_SIZE'] if 'WORLD_SIZE' in os.environ.keys() else gpu_num
    base_rank = os.environ['RANK'] if 'RANK' in os.environ.keys() else 0
    local_rank = (base_rank * gpu_num) + worker_id

    device = torch.device(f"cuda:{worker_id}")
    torch.cuda.set_device(device)
    return device, local_rank

device, local_rank = setup_device_and_distributed(0)

data_dir = "../data/climatenet"
val_dataset = ClimateDataset(
        data_dir=data_dir, train_flag=False, transforms=None
    )
val_collate_fn = val_dataset.collate_fn
debug_size = 10
indices = list(range(min(debug_size, len(val_dataset))))
train_dataset = torch.utils.data.Subset(val_dataset, indices)

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
    

val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=2, shuffle=False, num_workers=0,
        drop_last=False, collate_fn=val_collate_fn, worker_init_fn=partial(worker_init_fn, base_seed=3407)
    )

import matplotlib.pyplot as plt
from train_util import batch_to_cuda


from evaluator import StreamSegMetrics
from climatesam import ClimateSAM

ar_metrics = StreamSegMetrics(class_names=['Background', 'Foreground'])
tc_metrics = StreamSegMetrics(class_names=['Background', 'Foreground'])
worker_args = parse()
model = ClimateSAM(
        model_type=worker_args.sam_type, 
        mlp_ratio=worker_args.image_encoder_mlp_ratio,
        enable_wandb_logging=getattr(worker_args, 'debugging', False)  # Only log if debugging=True
    ).to(device=device)

epoch = 1


for val_step, batch in enumerate(val_dataloader):
    batch = batch_to_cuda(batch, device)
    images = model.set_infer_img(batch['input'])
