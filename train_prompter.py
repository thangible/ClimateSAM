import random
import numpy as np
import torch
import os
import torch.nn.functional as F
from functools import partial
from torch.utils.data import DataLoader
from train_util import batch_to_cuda, get_idle_gpu, get_idle_port, set_randomness, plot_with_projection, plot_mask_with_points_and_bbox, prompt_debug
from loss_function import ClimateLoss, compute_climate_loss, GeneratorLoss
from tqdm import tqdm
from contextlib import nullcontext
from train_parser import parse
from climatesam import ClimateSAM
from model.prompt_generator import PromptGenerator
from dataset.climatenet import ClimateDataset
from evaluator import StreamSegMetrics
import copy
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
    Sets up optimizer and scheduler for the prompt generator.
    """
    lr = getattr(worker_args, 'lr', 1e-4)
    weight_decay = getattr(worker_args, 'weight_decay', 1e-4)

    all_trainable_params = list(p for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(
        params=all_trainable_params, lr=lr, weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=worker_args.max_epoch_num, eta_min=1e-5
    )
    return optimizer, scheduler

def setup_device():
    """
    Setup device for training (single GPU only).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def train_one_epoch(epoch, embeddings_file_path, model, optimizer, scheduler, device, worker_args, max_epoch_num, scaler):
    """
    Train the prompt generator for one epoch using precomputed embeddings.
    """
    model.train()
    
    epoch_loss_dict = {}
    epoch_loss_count = 0
    
    if hasattr(worker_args, 'verbose') and worker_args.verbose:
        pbar = tqdm(embeddings_file_path, desc=f'Epoch {epoch}/{max_epoch_num}')
    else:
        pbar = embeddings_file_path
    
    for embedding_file in pbar:
        embeddings = torch.load(embedding_file, map_location=device)
        imgs, img_features, interm_embeddings, gt_masks, index = (
            embeddings['imgs'], 
            embeddings['img_features'], 
            embeddings['interm_embeddings'], 
            embeddings['gt_masks'], 
            embeddings['index']
        )
        
        # Forward pass through prompt generator
        tc_masks, ar_masks = model(interm_embeddings)

        # Compute losses
        gen_loss = GeneratorLoss(device)
        losses = gen_loss.compute_loss(ar_masks, tc_masks, gt_masks)
        total_loss = losses['total_loss']
        
        # Accumulate losses for logging
        for key, value in losses.items():
            if key not in epoch_loss_dict:
                epoch_loss_dict[key] = 0
            epoch_loss_dict[key] += value.item() if torch.is_tensor(value) else value

        # Backward pass
        backward_context = nullcontext
        with backward_context():
            scaler.scale(total_loss).backward()
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        epoch_loss_count += 1
        
        # Update progress bar
        if hasattr(pbar, 'set_postfix'):
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
    
    # Calculate average losses for the epoch
    if epoch_loss_count > 0:
        avg_epoch_losses = {key: epoch_loss_dict[key] / epoch_loss_count for key in epoch_loss_dict.keys()}
        
        # Log to wandb
        if hasattr(worker_args, 'wandb') and worker_args.wandb:
            log_dict = {f"train/{key}": avg_epoch_losses[key] for key in avg_epoch_losses.keys()}
            log_dict["epoch"] = epoch
            log_dict["learning_rate"] = scheduler.get_last_lr()[0]
            wandb.log(log_dict, step=epoch)
    
    scheduler.step()

def main_worker(worker_args):
    """
    Main worker function for training the prompt generator.
    """
    set_randomness()
    max_epoch_num = worker_args.max_epoch_num
    device = setup_device()
    print(f"Training initialized on device {device}.")
    
    # Setup embeddings paths
    embedding_dir_path = os.path.join("embeddings")
    embeddings_file_path = [
        os.path.join(embedding_dir_path, f) 
        for f in os.listdir(embedding_dir_path) 
        if f.endswith('.pt')
    ]
    
    if not embeddings_file_path:
        raise ValueError(f"No embedding files found in {embedding_dir_path}")
    
    print(f"Found {len(embeddings_file_path)} embedding files.")
    
    # Model configuration
    num_features_map = {
        'vit_b': 12,
        'vit_l': 24,
        'vit_h': 32
    }
    feature_per_block = {
        'vit_b': 3,
        'vit_l': 6,
        'vit_h': 9
    }
    
    # Get model type from args or default to vit_l
    model_type = getattr(worker_args, 'sam_type', 'vit_l')
    
    # Initialize prompt generator
    prompt_generator = PromptGenerator(
        num_features=num_features_map[model_type],
        feature_per_block=feature_per_block[model_type]
    ).to(device)
    
    # Setup optimizer and scheduler
    optimizer, scheduler = setup_optimizer_and_scheduler(prompt_generator, worker_args)
    
    # Initialize scaler
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"Starting training for {max_epoch_num} epochs...")
    
    for epoch in range(1, max_epoch_num + 1):
        train_one_epoch(
            epoch, embeddings_file_path, prompt_generator, 
            optimizer, scheduler, device, worker_args, max_epoch_num, scaler
        )
        
        # Save model checkpoint
        if hasattr(worker_args, 'save_model') and worker_args.save_model and epoch % 10 == 0:
            save_path = os.path.join(worker_args.exp_dir, f"prompt_generator_epoch_{epoch}.pth")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': prompt_generator.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, save_path)
            print(f"Model checkpoint saved to {save_path}")
            
            if hasattr(worker_args, 'wandb') and worker_args.wandb:
                wandb.save(save_path)

if __name__ == '__main__':
    print("Starting prompt generator training process...")
    args = parse()
    
    # Set default values for prompt generator training
    if not hasattr(args, 'max_epoch_num'):
        args.max_epoch_num = 100
    if not hasattr(args, 'lr'):
        args.lr = 1e-4
    if not hasattr(args, 'weight_decay'):
        args.weight_decay = 1e-4
    if not hasattr(args, 'verbose'):
        args.verbose = True
    
    # Initialize wandb if enabled
    if hasattr(args, 'wandb') and args.wandb:
        project_name = getattr(args, 'project_name', "climate-sam-prompt-generator")
        run_name = getattr(args, 'run_name', None)
        wandb.init(project=project_name, name=run_name, config=vars(args))

    # Setup GPU
    if torch.cuda.is_available():
        if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
            used_gpu = os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0]
        else:
            used_gpu = get_idle_gpu(gpu_num=1)[0]
            os.environ['CUDA_VISIBLE_DEVICES'] = str(used_gpu)
        print(f"Using GPU: {used_gpu}")
    else:
        print("Using CPU")

    # Launch training
    main_worker(worker_args=args)
