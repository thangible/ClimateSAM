import torch
from dataset.climatenet import ClimateDataset 
from torch.utils.data import DataLoader
from functools import partial
import random
import numpy as np
import os
from utility import prompt_debug, plot_mask_with_points_and_bbox, batch_to_cuda
from ClimateSAM.parser_config import parse

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
from utility import batch_to_cuda


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
    ar_points = batch['ar_point_prompts'].copy() #tuple of (list of (batch, num_points, 2) tensor or None, list of 1)
    tc_points = batch['tc_point_prompts'].copy() #tuple of (list of (batch, num_points, 2) tensor or None, lsit of 1)
    ar_bbox = batch['ar_bbox_prompts'].copy()  # list of (batch, num_boxes, 1, 4) tensor or None
    tc_bbox = batch['tc_bbox_prompts'].copy()  # list of (batch, num_boxes, 1, 4) tensor or None
    mask = batch['gt_mask'].copy()

    # prompt_debug(batch, text=f"Validation Step {val_step}")
    # if ar_points is not None:
    #     print(len(ar_bbox))
    #     print(ar_bbox)
    #     print(ar_bbox[0].shape)
    
    # tc_masks, ar_masks = model.infer(
    #     ar_point_prompts=ar_points,
    #     tc_point_prompts=tc_points,
    #     ar_bbox_prompts=ar_bbox,
    #     tc_bbox_prompts=tc_bbox
    # )
    mask = mask[0]
    tc_pred_mask = torch.zeros_like(mask)
    center_y, center_x = tc_pred_mask.shape[0] // 2, tc_pred_mask.shape[1] // 2
    radius = 30
    Y, X = torch.meshgrid(torch.arange(tc_pred_mask.shape[0]), torch.arange(tc_pred_mask.shape[1]), indexing='ij')
    dist = (X - center_x) ** 2 + (Y - center_y) ** 2
    tc_pred_mask[dist <= radius ** 2] = 1
    ar_pred_mask = torch.zeros_like(mask)  # Dummy prediction mask for AR
    
    ar_points_0 = batch['ar_point_prompts'][0]
    tc_points_0 = batch['tc_point_prompts'][0]
    ar_bbox_0 = batch['ar_bbox_prompts'][0]
    tc_bbox_0 = batch['tc_bbox_prompts'][0]
    
    # tc_pred_mask = tc_masks[0].squeeze()
    # ar_pred_mask = ar_masks[0].squeeze()
    
    # print(tc_pred_mask.shape)
    # print(ar_pred_mask.shape)
    # print(mask.shape)

    # # Additional debugging information
    # print("TC Pred Mask:", tc_pred_mask)
    # print("AR Pred Mask:", ar_pred_mask)
    # print("Ground Truth Mask:", mask)

    fig = plot_mask_with_points_and_bbox(mask, ar_points_0, tc_points_0, ar_bbox_0, tc_bbox_0, tc_pred_mask, ar_pred_mask, radius=8, save_path=f'exp/debug/val_step_{val_step}.png', axis =True)



  

        


  
# for val_step, batch in enumerate(val_dataloader):
#     # prompt_debug(batch, text=f"Before batch_to_cuda Step {val_step}")
#     batch = batch_to_cuda(batch, device)
    
#     # Set inference images once
#     images = model.set_infer_img(batch['input'])
    
#     ar_point_prompts=batch['ar_point_prompts']
#     tc_point_prompts=batch['tc_point_prompts']
#     ar_bbox_prompts=batch['ar_bbox_prompts']
#     tc_bbox_prompts=batch['tc_bbox_prompts']
    
#     prompt_debug(batch, text=f"Validation Step {val_step}")
    
#     # Perform inference with prompts
#     tc_masks, ar_masks = model.infer(
#         ar_point_prompts=ar_point_prompts,
#         tc_point_prompts=tc_point_prompts,
#         ar_bbox_prompts=ar_bbox_prompts,
#         tc_bbox_prompts=tc_bbox_prompts
#     )
    
#     masks_gt = batch['gt_mask']
#     masks_ar_gts = [(mask == 2).to(torch.uint8) for mask in masks_gt]
#     masks_tc_gts = [(mask == 1).to(torch.uint8) for mask in masks_gt]
    
#     # some processing to make sure the masks are in the right shape
#     for masks in [masks_ar_gts, masks_tc_gts, ar_masks, tc_masks]:
#             for i in range(len(masks)):
#                 if len(masks[i].shape) == 2:
#                     masks[i] = masks[i][None, None, :]
#                 if len(masks[i].shape) == 3:
#                     masks[i] = masks[i][:, None, :]
#                 if len(masks[i].shape) != 4:
#                     raise RuntimeError
#     # LOG
#     if val_step == 0:
#         for i in range(len(ar_point_prompts)):
#             mask = masks_gt[i]
#             ar_points = ar_point_prompts[i]
#             tc_points = tc_point_prompts[i]
#             ar_bbox = ar_bbox_prompts[i]
#             tc_bbox = tc_bbox_prompts[i]
#             tc_pred_mask = tc_masks[i]
#             ar_pred_mask = ar_masks[i]
#             save_path=os.path.join(worker_args.exp_dir, worker_args.run_name, 'images', f"epoch_{epoch}_step_{val_step}_image_{i}.png")
#             fig = plot_mask_with_points_and_bbox(mask, ar_points, tc_points, ar_bbox, tc_bbox, tc_pred_mask, ar_pred_mask, radius=8, save_path = save_path, axis=True)

#         torch.cuda.empty_cache()
        
#     ar_metrics.update(tc_masks, masks_ar_gts,  batch['index_name'])
#     tc_metrics.update(ar_masks, masks_tc_gts,  batch['index_name'])

    
# ar_metrict_dict, _ = ar_metrics.compute()
# tc_metric_dict, _ = tc_metrics.compute()

# miou_ar = ar_metrict_dict['Mean Foreground IoU']
# mean_acc_ar = ar_metrict_dict['Mean Acc']
# overall_acc_ar = ar_metrict_dict['Overall Acc']
# freqw_acc_ar = ar_metrict_dict['FreqW Acc']
# miout_including_bg_ar = ar_metrict_dict['Mean IoU']
# miou_tc = tc_metric_dict['Mean Foreground IoU']
# mean_acc_tc = tc_metric_dict['Mean Acc']
# overall_acc_tc = tc_metric_dict['Overall Acc']
# freqw_acc_tc = tc_metric_dict['FreqW Acc']
# miout_including_bg_tc = tc_metric_dict['Mean IoU']
# ar_metrics.reset()
# tc_metrics.reset()