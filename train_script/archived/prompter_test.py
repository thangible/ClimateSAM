import torch
from dataset.climatenet import ClimateDataset 
from torch.utils.data import DataLoader
from functools import partial
import random
import numpy as np
import os
from utility import prompt_debug, plot_mask_with_points_and_bbox, batch_to_cuda
from ClimateSAM.parser_config import parse
import matplotlib.pyplot as plt

from evaluator import StreamSegMetrics
from climatesam import ClimateSAM


def setup_device_and_distributed(worker_id):
    gpu_num = 1
    world_size = os.environ['WORLD_SIZE'] if 'WORLD_SIZE' in os.environ.keys() else gpu_num
    base_rank = os.environ['RANK'] if 'RANK' in os.environ.keys() else 0
    local_rank = (base_rank * gpu_num) + worker_id

    device = torch.device(f"cuda:{worker_id}")
    torch.cuda.set_device(device)
    return device, local_rank

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
    


device, local_rank = setup_device_and_distributed(0)
worker_args = parse()

data_dir = "../data/climatenet"
val_dataset = ClimateDataset(
        data_dir=data_dir, train_flag=False, transforms=None
    )
train_dataset = ClimateDataset(
        data_dir=data_dir, train_flag=True, shot_num=worker_args.shot_num,
        augmented=False
    )



val_collate_fn = val_dataset.collate_fn
train_collate_fn = train_dataset.collate_fn

train_debug_size = 30
indices = list(range(min(train_debug_size, len(train_dataset))))
train_dataset = torch.utils.data.Subset(train_dataset, indices)

val_debug_size = 10
indices = list(range(min(val_debug_size, len(val_dataset))))
val_dataset = torch.utils.data.Subset(val_dataset, indices)

sampler = None
    

train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=4, shuffle=sampler is None, num_workers=0,
        sampler=sampler, drop_last=False, collate_fn=train_collate_fn,
        worker_init_fn=partial(worker_init_fn, base_seed=3407)
    )

val_dataloader = DataLoader(
    dataset=val_dataset, batch_size=2, shuffle=False, num_workers=0,
    drop_last=False, collate_fn=val_collate_fn, worker_init_fn=partial(worker_init_fn, base_seed=3407)
)


ar_metrics = StreamSegMetrics(class_names=['Background', 'Foreground'])
tc_metrics = StreamSegMetrics(class_names=['Background', 'Foreground'])

model = ClimateSAM(
        model_type=worker_args.sam_type, 
        mlp_ratio=worker_args.image_encoder_mlp_ratio,
        enable_wandb_logging=getattr(worker_args, 'debugging', False)  # Only log if debugging=True
    ).to(device=device)

epoch = 1

model.train(mode = True, phase = 3, verbose = False)
for train_step, batch in enumerate(train_dataloader):

    batch = batch_to_cuda(batch, device)
    
    with torch.amp.autocast('cuda'):
        tc_mask, ar_mask, _ = model(batch['input'])