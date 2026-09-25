"""
Shared pieces of the prompter benchmark (Section 4.3).

- one frozen ClimateSAM checkpoint for every prompter
- a feature cache (encoder run once, stored as float16 memmaps)
- a fixed split: 358 train / 40 validation (held out from the ClimateNet train set) / 61 test
- SAM decoding from points, boxes and dense mask prompts on the cached features
- conversion of a predicted mask into point / box / mask prompts
- dataset-level IoU and object-level detection metrics
"""
import os
import sys
import json
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, 'train_script')):
    if p not in sys.path:
        sys.path.append(p)  # appended so prompter_bench/train.py is not shadowed by train_script/train.py
os.chdir(ROOT)  # ClimateSAM loads ./pretrained/sam_*.pth relative to the repo root

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.climatesam import ClimateSAM
from dataset.climatenet import ClimateDataset
from model.prompt.prompt_maker import make_bbox_prompts, make_point_prompts

IMG_H, IMG_W = 768, 1152
SAM_SIZE = 1024
LOWRES = 256
NUM_LAYERS = {'vit_b': 12, 'vit_l': 24, 'vit_h': 32}
VIT_DIM = {'vit_b': 768, 'vit_l': 1024, 'vit_h': 1280}
CLASSES = (('TC', 1), ('AR', 2))  # (decoder type, label in the ground truth)
EMPTY_LOGIT = -20.0
DATA_DIR = os.path.join(os.path.dirname(ROOT), 'data', 'climatenet')
CACHE_ROOT = os.path.join(ROOT, 'exp', 'feature_cache')
RESULTS = os.path.join(ROOT, 'results')
NUM_VAL = 40
MIN_OBJECT_PX = 20  # same component threshold as PromptMaker


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------
def load_climatesam(ckpt_name, mlp_ratio, device, sam_type='vit_b'):
    """ckpt_name is relative to exp/ without .pth, e.g. 'best_weights/infused_token_vitb_mlp1_best'."""
    climatesam = ClimateSAM(model_type=sam_type, mlp_ratio=mlp_ratio).to(device)
    ckpt = torch.load(os.path.join(ROOT, 'exp', f'{ckpt_name}.pth'), map_location=device)
    for name in ('image_encoder', 'mask_decoder', 'input_adapter'):
        getattr(climatesam, name).load_state_dict(ckpt[name])
    climatesam.eval()
    for p in climatesam.parameters():
        p.requires_grad = False
    return climatesam


# ------------------------------------------------------------
# SPLIT + FEATURE CACHE
# ------------------------------------------------------------
def split_indices():
    """Fixed split of the 398 ClimateNet training samples into train / validation."""
    perm = np.random.RandomState(0).permutation(398)
    return {'train': np.sort(perm[NUM_VAL:]), 'val': np.sort(perm[:NUM_VAL])}


def fold_of(num_train, num_folds=5):
    """Fold id of every position in the train split (for out-of-fold prompts)."""
    folds = np.empty(num_train, dtype=int)
    for k, part in enumerate(np.array_split(np.random.RandomState(1).permutation(num_train), num_folds)):
        folds[part] = k
    return folds


def cache_dir(tag):
    return os.path.join(CACHE_ROOT, tag)


@torch.no_grad()
def build_feature_cache(climatesam, tag, device, sam_type='vit_b', batch_size=4):
    """Encode the train and test sets once; features are stored as float16 memmaps."""
    out = cache_dir(tag)
    if os.path.exists(os.path.join(out, 'done.json')):
        return
    os.makedirs(out, exist_ok=True)
    L, D = NUM_LAYERS[sam_type], VIT_DIM[sam_type]
    for split, train_flag in (('train', True), ('test', False)):
        ds = ClimateDataset(data_dir=DATA_DIR, train_flag=train_flag, augmented=False, generate_prompt=False)
        n = len(ds)
        emb = np.lib.format.open_memmap(os.path.join(out, f'{split}_emb.npy'), 'w+', np.float16, (n, 256, 64, 64))
        vit = np.lib.format.open_memmap(os.path.join(out, f'{split}_vit.npy'), 'w+', np.float16, (n, L, 64, 64, D))
        gt = np.lib.format.open_memmap(os.path.join(out, f'{split}_gt.npy'), 'w+', np.uint8, (n, IMG_H, IMG_W))
        cg = np.lib.format.open_memmap(os.path.join(out, f'{split}_cgnet.npy'), 'w+', np.float32, (n, 4, IMG_H, IMG_W))
        names, i = [], 0
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=ClimateDataset.collate_fn)
        for batch in tqdm(loader, desc=f'encoding {split}'):
            e, feats, _, _ = climatesam.encode_images(batch['input'].to(device))
            b = len(e)
            emb[i:i + b] = e.half().cpu().numpy()
            vit[i:i + b] = torch.stack(feats, dim=1).half().cpu().numpy()
            gt[i:i + b] = torch.stack(batch['gt_mask']).numpy().astype(np.uint8)
            cg[i:i + b] = batch['cgnet_input'].numpy()
            names.extend(batch['index_name'])
            i += b
        for arr in (emb, vit, gt, cg):
            arr.flush()
        with open(os.path.join(out, f'{split}_names.json'), 'w') as f:
            json.dump(names, f)
    with open(os.path.join(out, 'done.json'), 'w') as f:
        json.dump({'tag': tag}, f)


class FeatureCache:
    """Read-only view of a cached split. 'train'/'val' index into the ClimateNet train set."""

    def __init__(self, tag, split):
        src = 'test' if split == 'test' else 'train'
        d = cache_dir(tag)
        self.emb = np.load(os.path.join(d, f'{src}_emb.npy'), mmap_mode='r')
        self.vit = np.load(os.path.join(d, f'{src}_vit.npy'), mmap_mode='r')
        self.gt = np.load(os.path.join(d, f'{src}_gt.npy'), mmap_mode='r')
        self.cgnet = np.load(os.path.join(d, f'{src}_cgnet.npy'), mmap_mode='r')
        with open(os.path.join(d, f'{src}_names.json')) as f:
            all_names = json.load(f)
        self.index = np.arange(len(all_names)) if split == 'test' else split_indices()[split]
        self.names = [all_names[i] for i in self.index]

    def __len__(self):
        return len(self.index)

    def load_to(self, device, layers, with_cgnet=False):
        """Load the whole split into memory on `device` (float16 features, uint8 labels)."""
        idx = self.index
        data = {
            'emb': torch.from_numpy(np.ascontiguousarray(self.emb[idx])).to(device),
            'vit': {l: torch.from_numpy(np.ascontiguousarray(self.vit[idx, l])).to(device) for l in layers},
            'gt': torch.from_numpy(np.ascontiguousarray(self.gt[idx])).to(device),
            'names': self.names,
        }
        if with_cgnet:
            data['cgnet'] = torch.from_numpy(np.ascontiguousarray(self.cgnet[idx])).to(device)
        return data


# ------------------------------------------------------------
# SAM DECODING
# ------------------------------------------------------------
def to_sam_frame(xy):
    """(..., 2) points or (..., 4) boxes in the 1152x768 frame -> the 1024x1024 SAM frame (stretched, not padded)."""
    scale = torch.tensor([SAM_SIZE / IMG_W, SAM_SIZE / IMG_H], device=xy.device, dtype=xy.dtype)
    shape = xy.shape
    return torch.round(xy.reshape(*shape[:-1], -1, 2) * scale).reshape(shape)


def sam_decode(climatesam, emb, interm0, mask_type, points=None, boxes=None, masks=None):
    """
    Decode prompts for ONE image. emb: (1, 256, 64, 64), interm0: (1, 64, 64, D).
    points: (coords (N, P, 2), labels (N, P)) and boxes (N, 4) in the 1024 frame; masks (N, 1, 256, 256) logits.
    Returns (N, 256, 256) HQ logits.
    """
    sparse, dense = climatesam.prompt_encoder(points=points, boxes=boxes, masks=masks)
    _, masks_hq = climatesam.mask_decoder(
        type=mask_type, image_embeddings=emb, image_pe=[climatesam.prompt_encoder.get_dense_pe()],
        sparse_prompt_embeddings=[sparse], dense_prompt_embeddings=[dense],
        multimask_output=False, interm_embeddings=[interm0])
    return masks_hq[0][:, 0]


def union_logits(logits):
    """(N, h, w) per-prompt logits -> (h, w): a pixel is foreground if any prompt says so."""
    if logits is None or len(logits) == 0:
        return torch.full((LOWRES, LOWRES), EMPTY_LOGIT)
    return logits.max(dim=0).values


def upsample(logits, mode='nearest'):
    """(h, w) logits -> (768, 1152), same post-processing as ClimateSAM.postprocess."""
    return F.interpolate(logits[None, None].float(), (IMG_H, IMG_W), mode=mode)[0, 0]


# ------------------------------------------------------------
# PROMPTS FROM A PREDICTED (OR GROUND-TRUTH) MASK
# ------------------------------------------------------------
def score_to_prompt(score, scale=1.0):
    """(768, 1152) or (256, 256) logit map -> (1, 1, 256, 256) dense mask prompt (logits, clamped)."""
    if score.shape[-2:] != (LOWRES, LOWRES):
        score = F.interpolate(score[None, None].float(), (LOWRES, LOWRES), mode='bilinear', align_corners=False)[0, 0]
    return (score.float() * scale).clamp(-20, 20)[None, None]


def binary_to_score(mask, value=10.0):
    """Binary mask -> +-value logit map (for prompters that only give hard masks)."""
    return mask.float() * 2 * value - value


def make_prompts(mask_np, kind, score=None, enlarge_ratio=0.0, n_pos=5, n_neg=5, device='cuda'):
    """
    mask_np: (768, 1152) binary numpy mask of one class.
    kind: 'bbox' | 'point' | 'bbox+point' | 'mask' | 'bbox+mask'
    score: (H, W) logit map used for the dense mask prompt (defaults to +-10 of the binary mask).
    Returns kwargs for sam_decode, or None if there is no object.
    """
    if score is None:
        score = binary_to_score(torch.from_numpy(mask_np).to(device))
    if kind == 'mask':
        return {'masks': score_to_prompt(score)} if mask_np.sum() >= MIN_OBJECT_PX else None

    boxes, _ = make_bbox_prompts(mask_np, 8, MIN_OBJECT_PX, enlarge_ratio=enlarge_ratio)
    if boxes is None:
        return None
    out = {}
    if 'bbox' in kind:
        out['boxes'] = to_sam_frame(boxes[:, 0].to(device))
    if 'point' in kind:
        pts, _ = make_point_prompts(mask_np, 8, MIN_OBJECT_PX, num_positive_points=n_pos, num_negative_points=n_neg)
        out['points'] = (to_sam_frame(pts[0].to(device)), pts[1].to(device))
    if 'mask' in kind:
        out['masks'] = score_to_prompt(score).expand(len(boxes), -1, -1, -1)
    return out


# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------
class SegMetrics:
    """Dataset-level IoU (TP/FP/FN summed over all test images), as in the thesis tables."""

    def __init__(self):
        self.tp, self.fp, self.fn = {}, {}, {}
        for c in ('TC', 'AR', 'BG'):
            self.tp[c] = self.fp[c] = self.fn[c] = 0
        self.per_image = []  # [tp, fp, fn] x (TC, AR, BG) per image, for bootstrap confidence intervals

    def update(self, pred_tc, pred_ar, gt):
        """pred_*: bool (H, W) tensors, gt: (H, W) labels. AR wins where both classes are predicted."""
        pred_tc = pred_tc & ~pred_ar
        preds = {'TC': pred_tc, 'AR': pred_ar, 'BG': ~(pred_tc | pred_ar)}
        gts = {'TC': gt == 1, 'AR': gt == 2, 'BG': gt == 0}
        counts = []
        for c in preds:
            p, g = preds[c], gts[c]
            tp, fp, fn = (p & g).sum().item(), (p & ~g).sum().item(), (~p & g).sum().item()
            self.tp[c] += tp
            self.fp[c] += fp
            self.fn[c] += fn
            counts += [tp, fp, fn]
        self.per_image.append(counts)

    def compute(self):
        iou = {c: self.tp[c] / max(self.tp[c] + self.fp[c] + self.fn[c], 1) for c in self.tp}
        return {'TC IoU': iou['TC'], 'AR IoU': iou['AR'], 'BG IoU': iou['BG'],
                'Mean IoU': (iou['TC'] + iou['AR'] + iou['BG']) / 3, 'Mean FG IoU': (iou['TC'] + iou['AR']) / 2}


class ObjectMetrics:
    """Object-level detection: a GT object is found if any predicted pixel overlaps it, and vice versa."""

    def __init__(self):
        self.c = {cls: dict(gt=0, found=0, pred=0, false=0) for cls, _ in CLASSES}

    def update(self, pred, gt_mask, cls):
        pred, gt_mask = pred.cpu().numpy().astype(np.uint8), gt_mask.cpu().numpy().astype(np.uint8)
        n_p, pl, ps, _ = cv2.connectedComponentsWithStats(pred, connectivity=8)
        n_g, gl, gs, _ = cv2.connectedComponentsWithStats(gt_mask, connectivity=8)
        c = self.c[cls]
        gt_ids = [k for k in range(1, n_g) if gs[k, 4] >= MIN_OBJECT_PX]
        pred_ids = [k for k in range(1, n_p) if ps[k, 4] >= MIN_OBJECT_PX]
        hit_gt = np.unique(gl[(pl > 0) & (gl > 0)])
        hit_pred = np.unique(pl[(gl > 0) & (pl > 0)])
        c['gt'] += len(gt_ids)
        c['found'] += sum(k in hit_gt for k in gt_ids)
        c['pred'] += len(pred_ids)
        c['false'] += sum(k not in hit_pred for k in pred_ids)

    def compute(self):
        out = {}
        for cls, c in self.c.items():
            out[f'{cls} recall'] = c['found'] / max(c['gt'], 1)
            out[f'{cls} precision'] = 1 - c['false'] / max(c['pred'], 1)
            out[f'{cls} objects'] = c['pred']
        return out
