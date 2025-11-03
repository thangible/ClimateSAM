import math
import random
import torch.nn.functional as F
import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import torch

class BaseTransform(ABC):
    def __init__(self, p: float = 1.0, **kwargs):
        assert 0.0 < p <= 1.0
        self.p = p
        self.end_init_hook(**kwargs)

    def end_init_hook(self, **kwargs):
        pass

    def __call__(self, img: np.ndarray, mask: np.ndarray = None, extra: np.ndarray = None):
        if random.random() < self.p:
            return self.apply(img, mask, extra)
        else:
            return img, mask, extra

    @abstractmethod
    def apply(self, input: np.ndarray, mask: np.ndarray = None, extra: np.ndarray = None):
        raise NotImplementedError


class Compose:
    def __init__(self, transforms: List[BaseTransform]):
        if isinstance(transforms, BaseTransform):
            transforms = [transforms]
        self.transforms = transforms

    def __call__(self, input: np.ndarray, mask: np.ndarray = None, extra: np.ndarray = None):
        for t in self.transforms:
            res = t(input, mask, extra)
            if not isinstance(res, tuple):
                input = res
            elif len(res) == 3:
                input, mask, extra = res
            elif len(res) == 2:
                input, mask = res
            else:
                raise RuntimeError("Transform returned tuple with unsupported size")
        return dict(input=input, mask=mask, extra=extra)


## Spatial Transforms Reworked for (F, H, W) input

class VerticalFlip(BaseTransform):
    def apply(self, input: np.ndarray, mask: np.ndarray = None, extra: np.ndarray = None):
        # input (F, H, W): flip along H-axis (axis 1)
        input = np.ascontiguousarray(input[:, ::-1, ...]) 
        
        # Mask (H, W): flip along H-axis (axis 0)
        if mask is not None:
            mask = np.ascontiguousarray(mask[::-1, ...])
        
        # Extra: apply same as input if provided
        if extra is not None:
            extra = np.ascontiguousarray(extra[:, ::-1, ...])
            
        return input, mask, extra


class HorizontalFlip(BaseTransform):
    def apply(self, input: np.ndarray, mask: np.ndarray = None, extra: np.ndarray = None):
        # input (F, H, W): flip along W-axis (axis 2)
        input = np.ascontiguousarray(input[:, :, ::-1, ...])
        
        # Mask (H, W): flip along W-axis (axis 1)
        if mask is not None:
            mask = np.ascontiguousarray(mask[:, ::-1, ...])
        
        # Extra: apply same as input if provided
        if extra is not None:
            extra = np.ascontiguousarray(extra[:, :, ::-1, ...])
            
        return input, mask, extra


class RandomHorizontalRoll(BaseTransform):
    """
    Cyclically shifts the input content horizontally (Roll). 
    The content that shifts off one side wraps around to the other side.
    """
    def end_init_hook(self, shift_limit: Union[Tuple[float, float], float] = (-0.2, 0.2)):
        """
        Initializes the random shift limits.
        
        Args:
            shift_limit: A float or a tuple (min_ratio, max_ratio) for the horizontal shift.
        """
        if isinstance(shift_limit, float):
            self.shift_limit = (-shift_limit, shift_limit)
        else:
            self.shift_limit = shift_limit
            
        assert self.shift_limit[0] < self.shift_limit[1]
        assert -1.0 <= self.shift_limit[0] and self.shift_limit[1] <= 1.0

    def apply(self, input: np.ndarray, mask: np.ndarray = None, extra: np.ndarray = None):
        """
        Performs the cyclic shift (roll) on the input and extra.
        """
        # Get H, W from input. input shape is (F, H, W)
        _, height, width = input.shape[:3] 

        # 1. Determine random shift ratio within limits
        shift_ratio = random.uniform(*self.shift_limit)

        # 2. Convert ratio to integer pixel shift 'k'
        # k > 0 shifts content to the right
        k = int(shift_ratio * width)

        # 3. Apply numpy.roll for cyclic shift along axis=2 (width) for input (F, H, W)
        input = np.roll(input, k, axis=2) 

        # 4. Apply numpy.roll for cyclic shift along axis=1 (width) for mask (H, W)
        if mask is not None:
            mask = np.roll(mask, k, axis=1)
            
        # 5. Extra: applied same as input if provided (roll along axis=2)
        if extra is not None:
            extra = np.roll(extra, k, axis=2)

        return input, mask, extra