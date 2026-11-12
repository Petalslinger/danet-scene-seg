"""训练/验证的输入变换。

所有几何变换同时作用于 image 与 mask：同一个随机数、同一个裁剪框，
只是 mask 用 NEAREST（不能插值，否则会出现 255 与类别之间的伪值）。

训练：随机缩放 → 随机裁剪 → 水平翻转 → 颜色抖动 → 归一化
验证：**不做任何几何缩放**，只把右下角 pad 到 8 的倍数（模型要求 stride=8 整除），
      pad 区 mask 填 255，指标里自然被忽略。这样预测和原图 GT 是同一坐标系，
      画图时不需要把预测再 resize 回去。
"""

from __future__ import annotations

import random
from typing import Tuple, Dict

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ColorJitter
import torchvision.transforms.functional as TF

__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "TrainTransform",
    "EvalTransform",
    "mask_to_tensor",
    "to_normalized_tensor",
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def to_normalized_tensor(
    image: Image.Image,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> torch.Tensor:
    """PIL RGB -> 归一化后的 CHW float32 张量。"""
    return TF.normalize(TF.to_tensor(image), mean, std)


def mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    """PIL(P 或 L) -> HW int64 张量，值域 0..20 与 255。"""
    return torch.from_numpy(np.array(mask, dtype=np.int64))


class TrainTransform:
    """训练增强。

    Args:
        crop_size:   裁剪尺寸，必须是 8 的倍数。
        scale_range: 随机缩放范围（在裁剪之前）。
        jitter:      颜色抖动强度，0 表示关闭。
        hflip_p:     水平翻转概率。
    """

    def __init__(
        self,
        crop_size: int = 512,
        scale_range: Tuple[float, float] = (0.5, 2.0),
        jitter: float = 0.4,
        hflip_p: float = 0.5,
        mean: Tuple[float, float, float] = IMAGENET_MEAN,
        std: Tuple[float, float, float] = IMAGENET_STD,
    ):
        if crop_size % 8 != 0:
            raise ValueError("crop_size 必须是 8 的倍数")
        if not 0 < scale_range[0] <= scale_range[1]:
            raise ValueError("scale_range 非法")
        self.crop_size = crop_size
        self.scale_range = scale_range
        self.hflip_p = hflip_p
        self.mean = mean
        self.std = std
        self.color_jitter = (
            ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter)
            if jitter > 0
            else None
        )

    def __call__(
        self, image: Image.Image, mask: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
        w, h = image.size
        crop = self.crop_size

        # 1) 随机缩放；若缩放后短边不足 crop，则按比例放大到刚好够裁
        scale = random.uniform(*self.scale_range)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        if min(nw, nh) < crop:
            ratio = crop / min(nw, nh)
            nw = max(crop, int(round(nw * ratio)))
            nh = max(crop, int(round(nh * ratio)))

        image = image.resize((nw, nh), Image.BILINEAR)
        mask = mask.resize((nw, nh), Image.NEAREST)

        # 2) 随机裁剪（同一个框）
        left = random.randint(0, nw - crop)
        top = random.randint(0, nh - crop)
        box = (left, top, left + crop, top + crop)
        image = image.crop(box)
        mask = mask.crop(box)

        # 3) 水平翻转
        if random.random() < self.hflip_p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # 4) 颜色抖动（只作用于图像）
        if self.color_jitter is not None:
            image = self.color_jitter(image)

        return to_normalized_tensor(image, self.mean, self.std), mask_to_tensor(mask), {}


class EvalTransform:
    """验证/推理变换：pad 到 pad_to 的倍数，不缩放、不增强。

    返回的 meta["pad"] = (pad_h, pad_w)，可视化时可用于裁掉填充区。
    """

    def __init__(
        self,
        pad_to: int = 8,
        ignore_index: int = 255,
        mean: Tuple[float, float, float] = IMAGENET_MEAN,
        std: Tuple[float, float, float] = IMAGENET_STD,
    ):
        if pad_to <= 0:
            raise ValueError("pad_to 必须为正")
        self.pad_to = pad_to
        self.ignore_index = ignore_index
        self.mean = mean
        self.std = std

    def __call__(
        self, image: Image.Image, mask: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
        w, h = image.size
        pad_w = (self.pad_to - w % self.pad_to) % self.pad_to
        pad_h = (self.pad_to - h % self.pad_to) % self.pad_to

        if pad_w or pad_h:
            new_w, new_h = w + pad_w, h + pad_h
            image_canvas = Image.new("RGB", (new_w, new_h), (0, 0, 0))
            image_canvas.paste(image, (0, 0))

            # mask 在 numpy 里补：直接取调色板索引，避免 P/L 模式互转时经调色板改写标签值
            indices = np.array(mask)
            padded = np.full((new_h, new_w), self.ignore_index, dtype=indices.dtype)
            padded[:h, :w] = indices
            image, mask = image_canvas, Image.fromarray(padded)

        return (
            to_normalized_tensor(image, self.mean, self.std),
            mask_to_tensor(mask),
            {"pad": (pad_h, pad_w)},
        )
