import copy
import math
from typing import Any, Iterable, List, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from model.segment_anything_ext.build_sam import sam_model_registry
from model.prompt_encoder import PromptEncoderWrapper
# from model.prompt_generator import PromptGenerator
from model.input_adapter import ClimateInputAdapter, LinearClimateInputAdapter

sam_ckpt_path_dict = dict(
    vit_b='./pretrained/sam_vit_b_01ec64.pth',
    vit_l='./pretrained/sam_vit_l_0b3195.pth',
    vit_h='./pretrained/sam_vit_h_4b8939.pth'
)


class LoRALinear(nn.Module):
    """Simple LoRA wrapper for nn.Linear (keeps original Linear and adds A,B adapters)."""

    def __init__(self, linear: nn.Linear, r: int = 4, alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError("LoRALinear must wrap nn.Linear")
        self.linear = linear
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 1.0
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        if r > 0:
            self.A = nn.Parameter(torch.zeros(r, self.in_features))
            self.B = nn.Parameter(torch.zeros(self.out_features, r))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)
            # marker
            setattr(self.A, 'is_lora', True)
            setattr(self.B, 'is_lora', True)
        else:
            self.A = None
            self.B = None
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        if self.r > 0 and not self.merged:
            x2 = self.dropout(x)
            lora_inter = torch.matmul(x2, self.A.t())
            lora_out = torch.matmul(lora_inter, self.B.t())
            out = out + self.scaling * lora_out
        return out

    def merge(self):
        if self.r <= 0 or self.merged:
            return
        delta = (self.B @ self.A) * self.scaling
        with torch.no_grad():
            self.linear.weight += delta
        self.merged = True

    def unmerge(self):
        if self.r <= 0 or not self.merged:
            return
        delta = (self.B @ self.A) * self.scaling
        with torch.no_grad():
            self.linear.weight -= delta
        self.merged = False


class _LoRA_qkv(nn.Module):
    """Replacement for qkv linear that adds low-rank adapters for q and v slices."""

    def __init__(self, qkv: nn.Linear, a_q: nn.Linear, b_q: nn.Linear, a_v: nn.Linear, b_v: nn.Linear):
        super().__init__()
        if not isinstance(qkv, nn.Linear):
            raise TypeError("qkv must be nn.Linear")
        self.qkv = qkv
        self.a_q = a_q
        self.b_q = b_q
        self.a_v = a_v
        self.b_v = b_v
        self.dim = qkv.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        new_q = self.b_q(self.a_q(x))
        new_v = self.b_v(self.a_v(x))
        qkv[..., : self.dim] = qkv[..., : self.dim] + new_q
        qkv[..., -self.dim :] = qkv[..., -self.dim :] + new_v
        return qkv


class LoRA_Sam(nn.Module):
    """Apply q/v LoRA adapters to SAM image_encoder blocks in-place."""

    def __init__(self, sam_model: nn.Module, r: int = 4, lora_layers: Optional[List[int]] = None):
        super().__init__()
        assert r > 0
        self.sam = sam_model
        blocks = getattr(self.sam.image_encoder, 'blocks', None)
        if blocks is None:
            raise RuntimeError('sam_model.image_encoder.blocks not found')
        if lora_layers is None:
            lora_layers = list(range(len(blocks)))
        self.adapters = []
        # freeze base
        for p in self.sam.image_encoder.parameters():
            p.requires_grad = False
        # inject adapters
        for i, blk in enumerate(blocks):
            if i not in lora_layers:
                continue
            if not hasattr(blk.attn, 'qkv') or not isinstance(blk.attn.qkv, nn.Linear):
                continue
            w_qkv = blk.attn.qkv
            dim = w_qkv.in_features
            a_q = nn.Linear(dim, r, bias=False)
            b_q = nn.Linear(r, dim, bias=False)
            a_v = nn.Linear(dim, r, bias=False)
            b_v = nn.Linear(r, dim, bias=False)
            # initialize
            nn.init.kaiming_uniform_(a_q.weight, a=math.sqrt(5))
            nn.init.zeros_(b_q.weight)
            nn.init.kaiming_uniform_(a_v.weight, a=math.sqrt(5))
            nn.init.zeros_(b_v.weight)
            # mark
            setattr(a_q.weight, 'is_lora', True)
            setattr(b_q.weight, 'is_lora', True)
            setattr(a_v.weight, 'is_lora', True)
            setattr(b_v.weight, 'is_lora', True)
            blk.attn.qkv = _LoRA_qkv(w_qkv, a_q, b_q, a_v, b_v)
            self.adapters.extend([a_q, b_q, a_v, b_v])

    def num_adapters(self) -> int:
        return len(self.adapters)


class LoRAClimateSAMVanilla(nn.Module):
    """LoRA SAM built from vanilla sam_model_registry, adapted for two-label (TC/AR) outputs.

    - loads a vanilla SAM from sam_model_registry
    - adapts input channels to accept 16-channel climate input (input_adapter)
    - duplicates mask_decoder into two heads: tc and ar
    - applies LoRA adapters to the image encoder qkv projections
    - exposes same high-level API as ClimateSAM for training/inference
    """

    def __init__(
        self,
        model_type: str = 'vit_b',
        r: int = 8,
        lora_layers: Optional[List[int]] = None,
        input_weights: Optional[List[int]] = None,
        use_prompt_generator: bool = False,
        mlp_ratio: float = 0.25,
        freeze_base: bool = True,
        enable_wandb_logging: bool = False,
    ):
        super().__init__()
        assert model_type in ['vit_b', 'vit_l', 'vit_h']
        self.enable_wandb_logging = enable_wandb_logging
        self.use_prompt_generator = use_prompt_generator

        # load vanilla sam
        self.ori_sam = sam_model_registry[model_type](sam_ckpt_path_dict[model_type])
        self.sam_img_size = (self.ori_sam.image_encoder.img_size, self.ori_sam.image_encoder.img_size)

        # input adapter 16->3 like ClimateSAM
        self.input_adapter = LinearClimateInputAdapter(in_channels=16, out_channels=3)
        # set initial weights focusing on selected input channels
        # with torch.no_grad():
        #     nn.init.normal_(self.input_adapter[0].weight, mean=0.0, std=0.02)
        #     if input_weights is None:
        #         input_weights = [0, 1, 2]
        #     for out_ch in range(self.input_adapter[0].weight.shape[0]):
        #         for in_ch in input_weights:
        #             self.input_adapter[0].weight[out_ch, in_ch, 0, 0] = 1.0

        # duplicate mask_decoder into two independent heads so they can adapt separately
        self.mask_decoder_tc = copy.deepcopy(self.ori_sam.mask_decoder)
        self.mask_decoder_ar = copy.deepcopy(self.ori_sam.mask_decoder)

        # optionally use PromptGenerator
        # if self.use_prompt_generator:
        #     num_features_map = {'vit_b': 12, 'vit_l': 24, 'vit_h': 32}
        #     features_per_block = {'vit_b': 3, 'vit_l': 6, 'vit_h': 9}
        #     self.prompt_generator = PromptGenerator(num_features=num_features_map[model_type],
        #                                             features_per_block=features_per_block[model_type])

        # prompt encoder wrapper from original SAM
        self.prompt_encoder = PromptEncoderWrapper(ori_sam=self.ori_sam, fix=True)

        # apply LoRA adapters to image encoder
        # we inject adapters in-place into self.ori_sam.image_encoder.blocks
        self.lora = LoRA_Sam(self.ori_sam, r=r, lora_layers=lora_layers)
        if freeze_base:
            # By default LoRA_Sam already froze base encoder params; ensure mask decoders are frozen
            for p in self.mask_decoder_tc.parameters():
                p.requires_grad = False
            for p in self.mask_decoder_ar.parameters():
                p.requires_grad = False
        # we keep reference to image_encoder for API compatibility
        self.image_encoder = self.ori_sam.image_encoder

    def train(self, mode: bool = True, phase: int = 1, verbose: bool = False):
        """Mirror ClimateSAM.train() so that the phase/verbose API is respected.

        Phase 1: train image_encoder (LoRA adapters inside it) + mask decoders.
        Phase 3: train prompt_generator only.
        LoRA adapter parameters are always kept trainable.
        """
        super().train(mode)

        # Freeze everything first
        for param in self.parameters():
            param.requires_grad = False

        if phase == 1:
            # Unfreeze image encoder LoRA adapters (already injected in-place)
            for n, c in self.named_children():
                if n in ['image_encoder', 'mask_decoder_tc', 'mask_decoder_ar']:
                    c.train(mode=mode)
                else:
                    c.eval()
            # Only LoRA adapter params should require grad
            for blk in self.image_encoder.blocks:
                if hasattr(blk, 'attn') and hasattr(blk.attn, 'qkv'):
                    qkv = blk.attn.qkv
                    if isinstance(qkv, _LoRA_qkv):
                        for p in qkv.parameters():
                            if hasattr(p, 'is_lora') and p.is_lora:
                                p.requires_grad = True
            # also unfreeze input_adapter and mask decoders
            for n, c in self.named_children():
                if n in ['mask_decoder_tc', 'mask_decoder_ar']:
                    for p in c.parameters():
                        p.requires_grad = True
            if verbose:
                print("Phase 1: training LoRA adapters + mask decoders")
                
        if phase == 2:
            for n, c in self.named_children():
                if n in ['image_encoder', 'mask_decoder_tc', 'mask_decoder_ar', 'input_adapter']:
                    c.eval()
                else:
                    c.train(mode=mode)
            # Only LoRA adapter params should require grad
            for blk in self.image_encoder.blocks:
                if hasattr(blk, 'attn') and hasattr(blk.attn, 'qkv'):
                    qkv = blk.attn.qkv
                    if isinstance(qkv, _LoRA_qkv):
                        for p in qkv.parameters():
                            if hasattr(p, 'is_lora') and p.is_lora:
                                p.requires_grad = True
            # also unfreeze input_adapter and mask decoders
            for n, c in self.named_children():
                if n in ['mask_decoder_tc', 'mask_decoder_ar', 'input_adapter']:
                    for p in c.parameters():
                        p.requires_grad = True
            if verbose:
                print("Phase 1: training LoRA adapters + mask decoders")
                
            if verbose:
                print("Phase 2: training LoRA adapters + mask decoders + input_adapter")

        elif phase == 3:
            if self.use_prompt_generator:
                for n, c in self.named_children():
                    if n == 'prompt_generator':
                        c.train(mode=mode)
                        for p in c.parameters():
                            p.requires_grad = True
                    else:
                        c.eval()
            if verbose:
                print("Phase 3: training prompt_generator")

        if verbose:
            for n, c in self.named_children():
                total = sum(p.numel() for p in c.parameters())
                trainable = sum(p.numel() for p in c.parameters() if p.requires_grad)
                if total > 0:
                    print(f"{n.upper():<25} | train={str(c.training):<5} | {trainable:>9,}/{total:>12,} ({100*trainable/total:>5.2f}%)")
            total = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"Phase {phase}: trainable = {trainable:,} / {total:,}")

        return self

    def encode_images(self, input: torch.Tensor):
        ori_img_size = [(input[i].shape[-2], input[i].shape[-1]) for i in range(len(input))]
        input = self.interpolate_input(input)
        imgs = self.input_adapter(input)
        imgs = self.preprocess_images(imgs)
        image_input = imgs.clone().detach()
        image_embeddings, interm_embeddings = self.image_encoder(imgs)
        return image_embeddings, interm_embeddings, image_input, ori_img_size

    def forward(self,
                image_input,
                image_embeddings: torch.Tensor,
                interm_embeddings: List[torch.Tensor],
                ori_img_size: List[Tuple],
                ar_point_prompts: List[Union[torch.Tensor, None]] = None,
                tc_point_prompts: List[Union[torch.Tensor, None]] = None,
                ar_bbox_prompts: List[Union[torch.Tensor, None]] = None,
                tc_bbox_prompts: List[Union[torch.Tensor, None]] = None,
                ar_mask_prompts: List[Union[torch.Tensor, None]] = None,
                tc_mask_prompts: List[Union[torch.Tensor, None]] = None,
                return_all_hq_masks: bool = False,
                hq_token_weight_ar: Optional[torch.Tensor] = None,
                hq_token_weight_tc: Optional[torch.Tensor] = None,
                ):
        batch_size = len(image_embeddings)

        ar_point_prompts, tc_point_prompts, ar_bbox_prompts, tc_bbox_prompts = self.preprocess_prompts(
            ar_point_prompts=ar_point_prompts,
            tc_point_prompts=tc_point_prompts,
            ar_bbox_prompts=ar_bbox_prompts,
            tc_bbox_prompts=tc_bbox_prompts,
            ori_img_size=ori_img_size
        )

        # encode prompts per image
        tc_sparse_embeddings, tc_dense_embeddings = [], []
        ar_sparse_embeddings, ar_dense_embeddings = [], []
        for batch_idx in range(batch_size):
            current_tc_sparse, current_tc_dense = self.prompt_encoder(
                points=tc_point_prompts[batch_idx] if tc_point_prompts is not None else None,
                boxes=tc_bbox_prompts[batch_idx] if tc_bbox_prompts is not None else None,
                masks=tc_mask_prompts[batch_idx] if tc_mask_prompts is not None else None,
            )
            current_ar_sparse, current_ar_dense = self.prompt_encoder(
                points=ar_point_prompts[batch_idx] if ar_point_prompts is not None else None,
                boxes=ar_bbox_prompts[batch_idx] if ar_bbox_prompts is not None else None,
                masks=ar_mask_prompts[batch_idx] if ar_mask_prompts is not None else None,
            )
            tc_sparse_embeddings.append(current_tc_sparse)
            tc_dense_embeddings.append(current_tc_dense)
            ar_sparse_embeddings.append(current_ar_sparse)
            ar_dense_embeddings.append(current_ar_dense)

        # decode masks per image with two separate decoders (vanilla SAM API)
        tc_pred_masks = []
        ar_pred_masks = []
        dense_pe = self.prompt_encoder.get_dense_pe()
        for i in range(batch_size):
            tc_masks_i, _ = self.mask_decoder_tc(
                image_embeddings=image_embeddings[i:i+1],
                image_pe=dense_pe,
                sparse_prompt_embeddings=tc_sparse_embeddings[i],
                dense_prompt_embeddings=tc_dense_embeddings[i],
                multimask_output=False,
            )
            ar_masks_i, _ = self.mask_decoder_ar(
                image_embeddings=image_embeddings[i:i+1],
                image_pe=dense_pe,
                sparse_prompt_embeddings=ar_sparse_embeddings[i],
                dense_prompt_embeddings=ar_dense_embeddings[i],
                multimask_output=False,
            )
            tc_pred_masks.append(tc_masks_i)
            ar_pred_masks.append(ar_masks_i)

        # postprocess masks to original sizes
        tc_post = [self.postprocess(m.clone(), ori_img_size[i]) for i, m in enumerate(tc_pred_masks)]
        ar_post = [self.postprocess(m.clone(), ori_img_size[i]) for i, m in enumerate(ar_pred_masks)]

        if not self.training:
            tc_post = self.assemble_raw_masks(tc_post)
            ar_post = self.assemble_raw_masks(ar_post)

        # free memory
        del image_embeddings, interm_embeddings
        torch.cuda.empty_cache()
        return tc_post, ar_post, image_input

    @torch.no_grad()
    def infer(self,
              ar_point_prompts: List[Union[torch.Tensor, None]] = None,
              tc_point_prompts: List[Union[torch.Tensor, None]] = None,
              ar_bbox_prompts: List[Union[torch.Tensor, None]] = None,
              tc_bbox_prompts: List[Union[torch.Tensor, None]] = None,
              ar_mask_prompts: List[Union[torch.Tensor, None]] = None,
              tc_mask_prompts: List[Union[torch.Tensor, None]] = None,
              return_all_hq_masks: bool = False):
        if not hasattr(self, 'img_features') or not hasattr(self, 'interm_features'):
            raise RuntimeError('Call set_infer_img() before infer()')
        batch_size = len(self.img_features)

        ar_point_prompts, tc_point_prompts, ar_bbox_prompts, tc_bbox_prompts = self.preprocess_prompts(
            ar_point_prompts=ar_point_prompts,
            tc_point_prompts=tc_point_prompts,
            ar_bbox_prompts=ar_bbox_prompts,
            tc_bbox_prompts=tc_bbox_prompts,
            ori_img_size=self.ori_infer_img_size
        )

        if self.use_prompt_generator:
            tc_masks, ar_masks = self.prompt_generator(self.interm_features)
            ar_point_prompts, tc_point_prompts, ar_bbox_prompts, tc_bbox_prompts = None, None, None, None

        tc_sparse_embeddings, tc_dense_embeddings = [], []
        ar_sparse_embeddings, ar_dense_embeddings = [], []
        for batch_idx in range(batch_size):
            cur_tc_sparse, cur_tc_dense = self.prompt_encoder(
                points=tc_point_prompts[batch_idx] if tc_point_prompts is not None else None,
                boxes=tc_bbox_prompts[batch_idx] if tc_bbox_prompts is not None else None,
                masks=tc_mask_prompts[batch_idx] if tc_mask_prompts is not None else None,
            )
            cur_ar_sparse, cur_ar_dense = self.prompt_encoder(
                points=ar_point_prompts[batch_idx] if ar_point_prompts is not None else None,
                boxes=ar_bbox_prompts[batch_idx] if ar_bbox_prompts is not None else None,
                masks=ar_mask_prompts[batch_idx] if ar_mask_prompts is not None else None,
            )
            tc_sparse_embeddings.append(cur_tc_sparse)
            tc_dense_embeddings.append(cur_tc_dense)
            ar_sparse_embeddings.append(cur_ar_sparse)
            ar_dense_embeddings.append(cur_ar_dense)

        tc_pred_masks = []
        ar_pred_masks = []
        dense_pe = self.prompt_encoder.get_dense_pe()
        for i in range(batch_size):
            tc_masks_i, _ = self.mask_decoder_tc(
                image_embeddings=self.img_features[i:i+1],
                image_pe=dense_pe,
                sparse_prompt_embeddings=tc_sparse_embeddings[i],
                dense_prompt_embeddings=tc_dense_embeddings[i],
                multimask_output=False,
            )
            ar_masks_i, _ = self.mask_decoder_ar(
                image_embeddings=self.img_features[i:i+1],
                image_pe=dense_pe,
                sparse_prompt_embeddings=ar_sparse_embeddings[i],
                dense_prompt_embeddings=ar_dense_embeddings[i],
                multimask_output=False,
            )
            tc_pred_masks.append(tc_masks_i)
            ar_pred_masks.append(ar_masks_i)

        tc_post = [self.postprocess(m.clone(), self.ori_infer_img_size[i]) for i, m in enumerate(tc_pred_masks)]
        ar_post = [self.postprocess(m.clone(), self.ori_infer_img_size[i]) for i, m in enumerate(ar_pred_masks)]

        tc_post = self.assemble_raw_masks(tc_post)
        ar_post = self.assemble_raw_masks(ar_post)

        return tc_post, ar_post

    @torch.no_grad()
    def set_infer_img(self, input: Union[List[torch.Tensor], torch.Tensor]):
        if isinstance(input, torch.Tensor):
            if len(input.shape) == 3:
                input = [input]
            elif len(input.shape) == 4:
                input = [input[i] for i in range(input.shape[0])]
            else:
                raise RuntimeError(f"Unsupported input shape: {input.shape}")
        elif not isinstance(input, list):
            raise RuntimeError('Input must be tensor or list of tensors')
        self.ori_infer_img_size = [(img.shape[-2], img.shape[-1]) for img in input]
        self.ori_infer_img = input
        input = self.interpolate_input(torch.stack(input))
        imgs = self.input_adapter(input)
        imgs = self.preprocess_images(imgs)
        self.img_features, self.interm_features = self.image_encoder(imgs)
        return imgs, self.img_features, self.interm_features

    @staticmethod
    def postprocess(output_masks: torch.Tensor, ori_img_size: Tuple):
        output_mask_size = (output_masks.size(-2), output_masks.size(-1))
        if output_mask_size != ori_img_size:
            if len(output_masks.shape) == 3:
                output_masks = output_masks.unsqueeze(1)
            output_masks = F.interpolate(output_masks, ori_img_size, mode='nearest')
        return output_masks

    def interpolate_input(self, input: torch.Tensor):
        if input.shape[-2:] != self.sam_img_size:
            input = F.interpolate(input, size=self.sam_img_size, mode='nearest')
        return input

    def preprocess_images(self, input: torch.Tensor):
        pixel_mean = self.ori_sam.pixel_mean.clone().detach().to(input.device).view(1, 3, 1, 1)
        pixel_std = self.ori_sam.pixel_std.clone().detach().to(input.device).view(1, 3, 1, 1)
        return (input - pixel_mean) / pixel_std

    def preprocess_prompts(self, ar_point_prompts=None, tc_point_prompts=None, ar_bbox_prompts=None, tc_bbox_prompts=None, ori_img_size=None):
        batch_num = len(ori_img_size)
        for i in range(batch_num):
            h_scale = self.sam_img_size[0] / ori_img_size[i][0]
            w_scale = self.sam_img_size[1] / ori_img_size[i][1]
            if tc_point_prompts is not None and tc_point_prompts[i] is not None:
                tc_point, tc_label = tc_point_prompts[i]
                tc_point[:, :, 0] *= w_scale
                tc_point[:, :, 1] *= h_scale
                tc_point = torch.round(tc_point)
                tc_point_prompts[i] = (tc_point, tc_label)
            if ar_point_prompts is not None and ar_point_prompts[i] is not None:
                ar_point, ar_label = ar_point_prompts[i]
                ar_point[:, :, 0] *= w_scale
                ar_point[:, :, 1] *= h_scale
                ar_point = torch.round(ar_point)
                ar_point_prompts[i] = (ar_point, ar_label)
            if tc_bbox_prompts is not None and tc_bbox_prompts[i] is not None:
                tc_bbox_prompts[i][..., 0] *= w_scale
                tc_bbox_prompts[i][..., 1] *= h_scale
                tc_bbox_prompts[i][..., 2] *= w_scale
                tc_bbox_prompts[i][..., 3] *= h_scale
                tc_bbox_prompts[i] = torch.round(tc_bbox_prompts[i])
            if ar_bbox_prompts is not None and ar_bbox_prompts[i] is not None:
                ar_bbox_prompts[i][..., 0] *= w_scale
                ar_bbox_prompts[i][..., 1] *= h_scale
                ar_bbox_prompts[i][..., 2] *= w_scale
                ar_bbox_prompts[i][..., 3] *= h_scale
                ar_bbox_prompts[i] = torch.round(ar_bbox_prompts[i])
        return ar_point_prompts, tc_point_prompts, ar_bbox_prompts, tc_bbox_prompts

    def discretize_mask(self, masks_logits):
        return torch.gt(masks_logits, self.ori_sam.mask_threshold).float()

    def assemble_raw_masks(self, raw_masks: List):
        masks = []
        for r_m in raw_masks:
            r_m = self.discretize_mask(r_m)
            r_m = torch.sum(r_m, dim=0, keepdim=True)
            masks.append(torch.clamp(r_m, max=1.0))
        return masks


__all__ = [
    'LoRAClimateSAMVanilla',
    'LoRA_Sam',
    'LoRALinear',
]
