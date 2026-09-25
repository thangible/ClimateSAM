import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(TRAIN_SCRIPT_DIR)

for p in (PROJECT_ROOT, TRAIN_SCRIPT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)


import time
import cv2
import numpy as np
import torch
import wandb
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm
from evaluator import StreamSegMetrics

from utility import get_idle_gpu, set_randomness, plot_mask_with_points_and_bbox, setup_optimizer_and_scheduler_for_generator
from loss_function import calculate_bce_loss, calculate_tversky_loss
from parser_config import parse
from model.climatesam import ClimateSAM
from dataset.climatenet import ClimateDataset
from model.mask_prompt_generator import MaskPromptGenerator

"""
Mask-prompt generator for ClimateSAM.

The generator predicts one logit map per class (TC, AR) at SAM's mask-prompt resolution (256x256),
and the frozen prompt encoder + mask decoder turn it into the final masks:
    AR: the AR logit map is the (only) prompt -> fully differentiable.
    TC: one box per connected component of the TC map, each paired with the TC logit map as mask prompt.
        Mask prompts alone lose small cyclones in SAM's 4x mask downscaling, so TCs need the box; the
        gradient still reaches the generator through the mask prompt.
The generator is trained on SAM's output (plus an auxiliary loss on its own maps) instead of on a proxy
segmentation loss followed by non-differentiable prompt extraction.

The mask prompts are logits, like SAM's own low-res masks: with ground-truth masks as prompts, {0, 1}
maps give ~0% IoU while +-10 logits give ~80% AR IoU on the validation set.

The image encoder is frozen and there is no augmentation, so all features are encoded once and cached
on the GPU; an epoch then only runs the small generator and the mask decoder.
"""

VIT_DIM = {'vit_b': 768, 'vit_l': 1024, 'vit_h': 1280}
CLASSES = (('TC', 1, 0), ('AR', 2, 1))  # (decoder type, gt label, generator channel)
EMPTY_LOGIT = -20.0  # SAM logit used where no TC box was proposed
TC_MIN_AREA = 2  # min TC component size in 256x256 cells (~27 px at 768x1152)
TC_MAX_BOXES = 16  # largest components kept per image (ground truth has at most 13 TCs per image)


# ------------------------------------------------------------
# FEATURE CACHE
# ------------------------------------------------------------
@torch.no_grad()
def cache_features(climatesam, dataset, layers, device, batch_size=4):
    """Encode every image once. Returns a dict of bf16 GPU tensors indexed by sample."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=ClimateDataset.collate_fn)
    emb, gt, names = [], [], []
    vit = {l: [] for l in layers}
    for batch in tqdm(loader, desc='caching features', leave=False):
        image_embeddings, interm_features, _, _ = climatesam.encode_images(batch['input'].to(device))
        emb.append(image_embeddings.to(torch.bfloat16))
        for l in layers:
            vit[l].append(interm_features[l].to(torch.bfloat16))
        gt.append(torch.stack(batch['gt_mask']).to(device, torch.uint8))
        names.extend(batch['index_name'])
    return {
        'emb': torch.cat(emb),
        'vit': {l: torch.cat(v) for l, v in vit.items()},
        'gt': torch.cat(gt),
        'names': names,
    }


def get_batch(cache, idx, gen_layers):
    emb = cache['emb'][idx].float()
    vit_feats = [cache['vit'][l][idx].float() for l in gen_layers]
    interm0 = cache['vit'][0][idx].float()
    gt = cache['gt'][idx].long()
    return emb, vit_feats, interm0, gt


# ------------------------------------------------------------
# SAM DECODING WITH GENERATED PROMPTS
# ------------------------------------------------------------
def sam_decode(climatesam, image_embeddings, interm0, mask_type, mask_prompts, boxes=None):
    """
    Args:
        mask_prompts: list (len B) of (N_i, 1, 256, 256) mask logits; one decoder pass per prompt
        boxes: optional list (len B) of (N_i, 4) boxes in the 1024x1024 SAM frame
    Returns:
        list (len B) of (N_i, 1, 256, 256) HQ mask logits
    """
    batch_size = len(image_embeddings)
    sparse, dense = [], []
    for b in range(batch_size):
        s, d = climatesam.prompt_encoder(points=None, boxes=boxes[b] if boxes is not None else None, masks=mask_prompts[b])
        sparse.append(s)
        dense.append(d)
    image_pe = climatesam.prompt_encoder.get_dense_pe()
    _, masks_hq = climatesam.mask_decoder(
        type=mask_type,
        image_embeddings=image_embeddings,
        image_pe=[image_pe] * batch_size,
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
        interm_embeddings=[interm0],
    )
    return masks_hq


def boxes_from_mask(mask):
    """(256, 256) bool mask -> (N, 4) float boxes of its largest connected components, in the 1024x1024 SAM frame."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.cpu().numpy().astype(np.uint8), connectivity=8)
    stats = sorted(stats[1:], key=lambda st: -st[4])[:TC_MAX_BOXES]
    boxes = [[x, y, x + w, y + h] for x, y, w, h, area in stats if area >= TC_MIN_AREA]
    return torch.tensor(boxes, dtype=torch.float32, device=mask.device).reshape(-1, 4) * 4


def seg_loss(logits, target, cls, worker_args):
    """Tversky + weighted BCE for one class. logits/target: (B, H, W)."""
    if cls == 'TC':
        alpha, beta, pos_weight = worker_args.alpha_tc_tversky, worker_args.beta_tc_tversky, worker_args.bce_weight_tc
    else:
        alpha, beta, pos_weight = worker_args.alpha_ar_tversky, worker_args.beta_ar_tversky, worker_args.bce_weight_ar
    tversky = calculate_tversky_loss(logits, target, alpha=alpha, beta=beta)
    bce = calculate_bce_loss(logits, target, weight=pos_weight)
    return worker_args.tversky_weight * tversky + worker_args.bce_weight * bce


def run_step(climatesam, generator, emb, vit_feats, interm0):
    """Returns generator logits (B, 2, 256, 256) and SAM HQ logits {cls: (B, 256, 256)}."""
    gen_logits = generator(emb, vit_feats)
    tc_prompt, ar_prompt = gen_logits[:, 0:1], gen_logits[:, 1:2]

    ar_masks = sam_decode(climatesam, emb, interm0, 'AR', [p[None] for p in ar_prompt])
    sam_logits = {'AR': torch.cat(ar_masks)[:, 0]}

    tc_boxes = [boxes_from_mask(p[0] > 0) for p in tc_prompt.detach()]
    keep = [b for b in range(len(emb)) if len(tc_boxes[b]) > 0]
    tc_union = {}
    if keep:
        tc_masks = sam_decode(climatesam, emb[keep], interm0[keep], 'TC',
                              [tc_prompt[b][None].expand(len(tc_boxes[b]), -1, -1, -1) for b in keep],
                              boxes=[tc_boxes[b] for b in keep])
        tc_union = {b: m[:, 0].max(dim=0).values for b, m in zip(keep, tc_masks)}  # union over boxes
    empty = torch.full(tc_prompt.shape[-2:], EMPTY_LOGIT, device=emb.device)
    sam_logits['TC'] = torch.stack([tc_union.get(b, empty).float() for b in range(len(emb))])
    return gen_logits, sam_logits


# ------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------
def train_one_epoch(epoch, cache, climatesam, generator, optimizer, scheduler, device, worker_args):
    generator.train()
    num_samples = len(cache['names'])
    perm = torch.randperm(num_samples, device=device)
    epoch_losses = {}
    num_steps = 0

    for start in range(0, num_samples, worker_args.train_bs):
        idx = perm[start:start + worker_args.train_bs]
        emb, vit_feats, interm0, gt = get_batch(cache, idx, worker_args.gen_feature_layers)

        with torch.autocast('cuda', dtype=torch.bfloat16):
            gen_logits, sam_logits = run_step(climatesam, generator, emb, vit_feats, interm0)

        losses = {}
        for cls, label, ch in CLASSES:
            target = (gt == label).float()
            gen_up = F.interpolate(gen_logits[:, ch:ch + 1].float(), size=gt.shape[-2:], mode='bilinear', align_corners=False)[:, 0]
            sam_up = F.interpolate(sam_logits[cls][:, None].float(), size=gt.shape[-2:], mode='bilinear', align_corners=False)[:, 0]
            losses[f'gen_{cls}'] = seg_loss(gen_up, target, cls, worker_args)
            losses[f'sam_{cls}'] = seg_loss(sam_up, target, cls, worker_args)

        total = worker_args.aux_weight * (losses['gen_TC'] + losses['gen_AR']) \
            + worker_args.sam_loss_weight * (losses['sam_TC'] + losses['sam_AR'])
        losses['total'] = total

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()

        for k, v in losses.items():
            epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()
        num_steps += 1

    scheduler.step()
    avg = {k: v / num_steps for k, v in epoch_losses.items()}
    if worker_args.wandb:
        log = {f'train/{k}': v for k, v in avg.items()}
        log['learning_rate'] = scheduler.get_last_lr()[0]
        wandb.log(log, step=epoch)
    return avg


# ------------------------------------------------------------
# EVAL
# ------------------------------------------------------------
@torch.no_grad()
def validate_one_epoch(epoch, cache, climatesam, generator, device, worker_args, save_images=False):
    generator.eval()
    metrics = {f'{src}_{cls}': StreamSegMetrics(class_names=['Background', 'Foreground'])
               for src in ('gen', 'sam') for cls, _, _ in CLASSES}
    num_samples = len(cache['names'])

    for start in range(0, num_samples, worker_args.val_bs):
        idx = torch.arange(start, min(start + worker_args.val_bs, num_samples), device=device)
        emb, vit_feats, interm0, gt = get_batch(cache, idx, worker_args.gen_feature_layers)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            gen_logits, sam_logits = run_step(climatesam, generator, emb, vit_feats, interm0)

        # same post-processing as ClimateSAM.postprocess: nearest upsampling, threshold at 0
        preds = {}
        for cls, label, ch in CLASSES:
            preds[f'gen_{cls}'] = F.interpolate(gen_logits[:, ch:ch + 1].float(), size=gt.shape[-2:], mode='nearest') > 0
            preds[f'sam_{cls}'] = F.interpolate(sam_logits[cls][:, None].float(), size=gt.shape[-2:], mode='nearest') > 0
            gts = [(g == label).to(torch.uint8)[None, None] for g in gt]
            for src in ('gen', 'sam'):
                key = f'{src}_{cls}'
                metrics[key].update(gts, [p[None].to(torch.uint8) for p in preds[key]], [cache['names'][i] for i in idx.tolist()])

        if save_images and start == 0:
            image_dir = os.path.join(worker_args.exp_dir, worker_args.run_name, 'images')
            os.makedirs(image_dir, exist_ok=True)
            for i in range(len(idx)):
                for src in ('gen', 'sam'):
                    fig = plot_mask_with_points_and_bbox(
                        gt[i], tc_pred_mask=preds[f'{src}_TC'][i:i + 1].float(), ar_pred_mask=preds[f'{src}_AR'][i:i + 1].float(),
                        save_path=os.path.join(image_dir, f'epoch_{epoch}_{src}_{i}.png'), axis=True,
                        title=f'Epoch {epoch} - {"Generator prompt" if src == "gen" else "SAM output"} {i}')
                    if worker_args.wandb:
                        wandb.log({f'valid/{src}_image_{i}': wandb.Image(fig)}, step=epoch)

    results = {key: m.compute()[0]['Mean Foreground IoU'] for key, m in metrics.items()}
    if worker_args.wandb:
        wandb.log({f'valid/iou_{k}': v for k, v in results.items()}, step=epoch)
    return results


# ------------------------------------------------------------
# MODELS
# ------------------------------------------------------------
def build_climatesam(worker_args, device):
    climatesam = ClimateSAM(model_type=worker_args.sam_type, mlp_ratio=worker_args.image_encoder_mlp_ratio).to(device)
    ckpt_path = os.path.join(worker_args.exp_dir, f"{worker_args.encoder_weights_name}.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Pretrained weights not found at {ckpt_path}.")
    ckpt = torch.load(ckpt_path, map_location=device)
    for name in ('image_encoder', 'mask_decoder', 'input_adapter'):
        if name not in ckpt:
            raise ValueError(f"{name} weights not found in {ckpt_path}.")
        getattr(climatesam, name).load_state_dict(ckpt[name])
    print(f"ClimateSAM weights loaded from {ckpt_path}")

    climatesam.eval()
    for p in climatesam.parameters():
        p.requires_grad = False
    return climatesam


def build_generator(worker_args, device):
    generator = MaskPromptGenerator(
        vit_dim=VIT_DIM[worker_args.sam_type],
        num_vit_feats=len(worker_args.gen_feature_layers),
        channels=worker_args.fuse_channels,
        num_blocks=worker_args.gen_num_blocks,
    ).to(device)
    if worker_args.load_pretrained:
        ckpt = torch.load(os.path.join(worker_args.exp_dir, worker_args.pretrained_name), map_location=device)
        generator.load_state_dict(ckpt['prompt_generator'])
    return generator


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main(worker_args):
    set_randomness()
    device = torch.device('cuda')

    num_layers = {'vit_b': 12, 'vit_l': 24, 'vit_h': 32}[worker_args.sam_type]
    worker_args.gen_feature_layers = [l % num_layers for l in worker_args.gen_feature_layers]
    cached_layers = sorted(set(worker_args.gen_feature_layers) | {0})  # block 0 feeds the HQ decoder

    climatesam = build_climatesam(worker_args, device)
    generator = build_generator(worker_args, device)
    print(f"Generator trainable params: {sum(p.numel() for p in generator.parameters()):,}")

    train_dataset = ClimateDataset(data_dir=worker_args.data_dir, train_flag=True, augmented=False, generate_prompt=False)
    val_dataset = ClimateDataset(data_dir=worker_args.data_dir, train_flag=False, augmented=False, generate_prompt=False)
    if worker_args.debugging:
        train_dataset = torch.utils.data.Subset(train_dataset, list(range(16)))
        val_dataset = torch.utils.data.Subset(val_dataset, list(range(8)))
        worker_args.max_epoch_num, worker_args.valid_per_epochs = 2, 1

    t0 = time.time()
    train_cache = cache_features(climatesam, train_dataset, cached_layers, device)
    val_cache = cache_features(climatesam, val_dataset, cached_layers, device)
    print(f"Cached {len(train_cache['names'])} train / {len(val_cache['names'])} val samples in {time.time() - t0:.0f}s "
          f"({torch.cuda.memory_allocated() / 1024 ** 3:.1f} GB on GPU)")

    optimizer, scheduler = setup_optimizer_and_scheduler_for_generator(None, generator, worker_args)

    best_score, best = -1, None
    for epoch in range(1, worker_args.max_epoch_num + 1):
        t0 = time.time()
        losses = train_one_epoch(epoch, train_cache, climatesam, generator, optimizer, scheduler, device, worker_args)
        print(f"Epoch {epoch} ({time.time() - t0:.1f}s) - " + ", ".join(f"{k}: {v:.4f}" for k, v in losses.items()))

        if epoch % worker_args.valid_per_epochs == 0 or epoch == worker_args.max_epoch_num:
            results = validate_one_epoch(epoch, val_cache, climatesam, generator, device, worker_args,
                                         save_images=epoch % (worker_args.valid_per_epochs * 5) == 0)
            # without the SAM loss the generator is a plain segmenter, so select it on its own masks
            src = 'sam' if worker_args.sam_loss_weight > 0 else 'gen'
            score = (results[f'{src}_TC'] + results[f'{src}_AR']) / 2
            print(f"  valid - SAM IoU TC: {results['sam_TC']:.2%}, AR: {results['sam_AR']:.2%} | "
                  f"generator IoU TC: {results['gen_TC']:.2%}, AR: {results['gen_AR']:.2%}")
            if score > best_score:
                best_score, best = score, dict(results, epoch=epoch)
                if worker_args.save_model:
                    save_dir = os.path.join(worker_args.exp_dir, 'best_weights')
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"best_mask_prompt_generator_{worker_args.sam_type}_{worker_args.run_name}.pth")
                    torch.save({
                        'prompt_generator': generator.state_dict(),
                        'gen_feature_layers': worker_args.gen_feature_layers,
                        'fuse_channels': worker_args.fuse_channels,
                        'gen_num_blocks': worker_args.gen_num_blocks,
                        'encoder_weights_name': worker_args.encoder_weights_name,
                    }, save_path)

    print(f"Best (epoch {best['epoch']}): SAM IoU TC {best['sam_TC']:.2%}, AR {best['sam_AR']:.2%} | "
          f"generator IoU TC {best['gen_TC']:.2%}, AR {best['gen_AR']:.2%}")
    return best


if __name__ == '__main__':
    args = parse()
    if 'CUDA_VISIBLE_DEVICES' not in os.environ and torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(get_idle_gpu(gpu_num=1)[0])
    if args.wandb:
        wandb.init(project=args.project_name, name=args.run_name, config=vars(args))
    main(args)
