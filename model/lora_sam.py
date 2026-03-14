import torch
from torch import nn
from typing import Iterable, List, Any, Optional, Tuple, Union
import math


class LoRALinear(nn.Module):
    """
    Wrap an existing nn.Linear and add a low-rank adaptation W + scale * (B @ A)
    where A: (r, in_features), B: (out_features, r). The adapter is applied to input x
    as (B @ (A @ x)).

    Attributes:
        merged: when True, adapter is merged into the base linear weight for fast inference.
    """

    def __init__(self, linear: nn.Linear, r: int = 4, alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError("LoRALinear must wrap nn.Linear")

        self.in_features = linear.in_features
        self.out_features = linear.out_features

        # Keep a copy of the original linear (weight and bias)
        self.linear = linear

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 1.0
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        if r > 0:
            # A maps from in_features -> r (we store as (r, in_features) so A @ x works)
            self.A = nn.Parameter(torch.zeros(r, self.in_features))
            # B maps from r -> out_features (store as (out_features, r))
            self.B = nn.Parameter(torch.zeros(self.out_features, r))
            # initialize
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)
            # mark params for easy detection
            self.A.is_lora = True  # type: ignore[attr-defined]
            self.B.is_lora = True  # type: ignore[attr-defined]
        else:
            self.A = None
            self.B = None

        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base linear output
        result = self.linear(x)
        if self.r > 0 and not self.merged:
            # LoRA path: x -> A @ x  -> B @ (A @ x)
            # x shape: (..., in_features)
            orig_shape = x.shape
            x2 = self.dropout(x)
            # compute A @ x: need shape (..., r)
            # A: (r, in_f), so use matmul(x, A.t())
            lora_inter = torch.matmul(x2, self.A.t())  # (..., r)
            lora_out = torch.matmul(lora_inter, self.B.t())  # (..., out_features)
            result = result + self.scaling * lora_out
        return result

    def merge(self):
        """Merge LoRA adapter into base linear weight (in-place)."""
        if self.r <= 0 or self.merged:
            return
        # weight: (out_features, in_features)
        delta = (self.B @ self.A) * self.scaling
        with torch.no_grad():
            self.linear.weight += delta
        self.merged = True

    def unmerge(self):
        """Remove merged adapter from base linear weight (in-place)."""
        if self.r <= 0 or not self.merged:
            return
        delta = (self.B @ self.A) * self.scaling
        with torch.no_grad():
            self.linear.weight -= delta
        self.merged = False


def _replace_linear(module: nn.Module, name: str, r: int, alpha: float, target_substrings: Iterable[str], dropout: float = 0.0) -> int:
    """Recursively replace matching nn.Linear children with LoRALinear.

    Returns the number of replacements performed in this module.
    """
    replacements = 0
    for child_name, child in list(module.named_children()):
        full_name = f"{name}.{child_name}" if name else child_name
        # If child itself has submodules, recurse first
        replacements += _replace_linear(child, full_name, r, alpha, target_substrings, dropout)

        # If the child's class is Linear and its name contains any target substring, replace it
        if isinstance(child, nn.Linear) and any(sub in child_name for sub in target_substrings):
            wrapped = LoRALinear(child, r=r, alpha=alpha, dropout=dropout)
            setattr(module, child_name, wrapped)
            replacements += 1

    return replacements


def apply_lora_to_sam(sam_model: nn.Module, r: int = 4, alpha: float = 1.0, target_substrings: List[str] = None, dropout: float = 0.0) -> int:
    """
    Apply LoRA adapters to a SAM model by replacing selected nn.Linear modules.

    Args:
        sam_model: model instance (e.g. ori_sam returned by sam_model_registry)
        r: LoRA rank
        alpha: LoRA scaling
        target_substrings: list of substrings of module names to target (defaults to common attention/mlp proj names)
        dropout: optional dropout on LoRA input

    Returns:
        int: number of linear layers replaced
    """
    if target_substrings is None:
        target_substrings = ["q", "k", "v", "proj", "mlp", "fc1", "fc2"]

    # Apply replacement on the whole sam_model (safe generic approach)
    replacements = _replace_linear(sam_model, name="", r=r, alpha=alpha, target_substrings=target_substrings, dropout=dropout)
    return replacements


def set_lora_trainable_only(model: nn.Module):
    """
    Freeze all parameters except LoRA adapter parameters (marked by attribute is_lora).
    After calling this, only the LoRA A/B params will require gradients.
    """
    for p in model.parameters():
        p.requires_grad = False

    for module in model.modules():
        for name, param in getattr(module, "named_parameters", lambda **kwargs: [])(recurse=False):
            # safety: skip if no attribute
            pass

    # Unfreeze parameters that have the marker is_lora
    for p in model.parameters():
        if hasattr(p, 'is_lora') and getattr(p, 'is_lora'):
            p.requires_grad = True


def merge_lora(model: nn.Module):
    """Merge all LoRALinear adapters into their base linear weights (use before exporting/eval if desired)."""
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.merge()


def unmerge_lora(model: nn.Module):
    """Unmerge all LoRALinear adapters from base linear weights."""
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.unmerge()


class _LoRA_qkv(nn.Module):
    """LoRA-aware replacement for SAM's qkv linear.

    It keeps the original qkv linear and adds separate low-rank adapters
    for q and v (like Sheng Wang's implementation). During forward the
    adapters are computed and added to the q and v regions of the qkv output.
    """

    def __init__(
        self,
        qkv: nn.Linear,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
    ):
        super().__init__()
        if not isinstance(qkv, nn.Linear):
            raise TypeError("qkv must be nn.Linear")
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # original qkv output (..., 3*dim)
        qkv = self.qkv(x)
        # adapter outputs shape (..., dim)
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        # add into q and v slices
        qkv[..., : self.dim] = qkv[..., : self.dim] + new_q
        qkv[..., -self.dim :] = qkv[..., -self.dim :] + new_v
        return qkv


class LoRA_Sam(nn.Module):
    """Apply LoRA adapters to a SAM-like image encoder by surgery on the qkv linears.

    This wrapper keeps a reference to the original sam_model and replaces
    the qkv Linear in selected transformer blocks with _LoRA_qkv which
    contains small adapter matrices.
    """

    def __init__(self, sam_model: nn.Module, r: int = 4, lora_layer: Optional[List[int]] = None):
        super().__init__()
        assert r > 0, "r must be > 0"
        self.sam = sam_model

        # choose layers (by default all blocks)
        blocks = getattr(self.sam.image_encoder, "blocks", None)
        if blocks is None:
            raise RuntimeError("sam_model.image_encoder.blocks not found")

        if lora_layer is None:
            self.lora_layer = list(range(len(blocks)))
        else:
            self.lora_layer = lora_layer

        # storage for adapters
        self.w_As: List[nn.Module] = []
        self.w_Bs: List[nn.Module] = []

        # freeze base encoder weights
        for p in self.sam.image_encoder.parameters():
            p.requires_grad = False

        # perform surgery per-block
        for i, blk in enumerate(blocks):
            if i not in self.lora_layer:
                continue
            # expect block.attn.qkv to be an nn.Linear
            if not hasattr(blk.attn, "qkv") or not isinstance(blk.attn.qkv, nn.Linear):
                continue
            w_qkv_linear = blk.attn.qkv
            dim = w_qkv_linear.in_features
            # create adapter linear layers (A then B) for q and v
            w_a_q = nn.Linear(dim, r, bias=False)
            w_b_q = nn.Linear(r, dim, bias=False)
            w_a_v = nn.Linear(dim, r, bias=False)
            w_b_v = nn.Linear(r, dim, bias=False)

            # store
            self.w_As.append(w_a_q)
            self.w_Bs.append(w_b_q)
            self.w_As.append(w_a_v)
            self.w_Bs.append(w_b_v)

            # replace qkv with LoRA-enabled module
            blk.attn.qkv = _LoRA_qkv(w_qkv_linear, w_a_q, w_b_q, w_a_v, w_b_v)

        # initialize adapter parameters
        self.reset_parameters()

    def reset_parameters(self):
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def num_adapters(self) -> int:
        return len(self.w_As)

    def state_dict(self, *args, **kwargs):
        # return only lora params plus base if needed; default to sam state_dict
        return self.sam.state_dict(*args, **kwargs)

    def forward(self, *args, **kwargs):
        # transparent wrapper (users normally call sam directly)
        return self.sam(*args, **kwargs)


class LoRAClimateSAM(nn.Module):
    """A thin LoRA wrapper around an existing ClimateSAM instance.

    This wrapper keeps the same input/output API as ClimateSAM but applies
    the LoRA adapters to the image encoder internals. It delegates all
    behavior to the wrapped ClimateSAM except for applying and managing the
    LoRA adapters.

    Usage:
      base = ClimateSAM(...)
      lora_wrapper = LoRAClimateSAM(base, r=8, alpha=32, freeze_base=True)
      # use lora_wrapper.encode_images / forward / infer etc.
    """

    def __init__(
        self,
        base_model: nn.Module,
        r: int = 4,
        alpha: float = 1.0,
        target_substrings: Optional[List[str]] = None,
        dropout: float = 0.0,
        freeze_base: bool = True,
    ):
        super().__init__()
        self.base = base_model

        if target_substrings is None:
            target_substrings = ["q", "k", "v", "proj", "mlp", "fc1", "fc2"]

        # Apply LoRA adapters to the image encoder (only)
        self.replacements = apply_lora_to_sam(self.base.image_encoder, r=r, alpha=alpha, target_substrings=target_substrings, dropout=dropout)

        # Optionally freeze all non-LoRA params so only LoRA adapters are trained
        if freeze_base:
            set_lora_trainable_only(self.base)
        else:
            # ensure LoRA params are trainable even if the caller didn't freeze the rest
            for p in self.base.parameters():
                if hasattr(p, 'is_lora') and getattr(p, 'is_lora'):
                    p.requires_grad = True

    # Delegate common ClimateSAM interfaces so this wrapper can be used interchangeably
    def encode_images(self, *args, **kwargs):
        return self.base.encode_images(*args, **kwargs)

    def save_image_embeddings(self, *args, **kwargs):
        return self.base.save_image_embeddings(*args, **kwargs)

    def set_infer_img(self, *args, **kwargs):
        return self.base.set_infer_img(*args, **kwargs)

    def infer(self, *args, **kwargs):
        return self.base.infer(*args, **kwargs)

    def preprocess_images(self, *args, **kwargs):
        return self.base.preprocess_images(*args, **kwargs)

    def preprocess_prompts(self, *args, **kwargs):
        return self.base.preprocess_prompts(*args, **kwargs)

    def postprocess(self, *args, **kwargs):
        return self.base.postprocess(*args, **kwargs)

    def assemble_raw_masks(self, *args, **kwargs):
        return self.base.assemble_raw_masks(*args, **kwargs)

    def discretize_mask(self, *args, **kwargs):
        return self.base.discretize_mask(*args, **kwargs)

    def log_masks(self, *args, **kwargs):
        return self.base.log_masks(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.base.forward(*args, **kwargs)

    def train(self, mode: bool = True, *args, **kwargs):
        # delegate to base.train which may accept extra params like phase, verbose
        return self.base.train(mode, *args, **kwargs)

    # LoRA utilities that operate on the wrapped model
    def merge_lora(self):
        """Merge LoRA adapters into base weights for faster inference."""
        merge_lora(self.base.image_encoder)

    def unmerge_lora(self):
        """Unmerge LoRA adapters from base weights to resume training."""
        unmerge_lora(self.base.image_encoder)

    def num_lora_replacements(self) -> int:
        return int(self.replacements)

    # Fallback attribute access to the wrapped model for transparency
    def __getattr__(self, name: str) -> Any:
        # This ensures attributes not defined on the wrapper are fetched from base
        if name in ("base", "replacements"):
            return super().__getattribute__(name)
        return getattr(self.base, name)


__all__ = [
    "LoRALinear",
    "apply_lora_to_sam",
    "set_lora_trainable_only",
    "merge_lora",
    "unmerge_lora",
    "LoRAClimateSAM",
]
