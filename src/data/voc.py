"""Pascal VOC 2012 分割数据集。

目录假设（两种都能自动识别）：
    data/VOCdevkit/VOC2012/...
    data/VOC2012/...

每张图返回 dict：
    image:    C x H x W，已归一化
    mask:     H x W int64，0..20 与 255（忽略边界）
    image_id: 字符串
    meta:     变换产生的附加信息（pad 等）
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import EvalTransform

__all__ = [
    "VOC_CLASSES",
    "VOC_PALETTE",
    "VOCSegmentation",
    "resolve_voc_root",
    "decode_mask",
    "collate_seg",
]

VOC_CLASSES: Tuple[str, ...] = (
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)

# VOC 官方调色板，用于把标签图变成彩色图
VOC_PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128),
    (128, 0, 128), (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0),
    (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128),
    (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
    (0, 64, 128),
)


def resolve_voc_root(root: str | Path, year: str = "2012") -> Path:
    """在常见布局中定位 VOC 根目录。"""
    root = Path(root)
    candidates = [
        root / "VOCdevkit" / f"VOC{year}",
        root / f"VOC{year}",
        root,
    ]
    for candidate in candidates:
        if (candidate / "JPEGImages").is_dir() and (candidate / "ImageSets").is_dir():
            return candidate
    raise FileNotFoundError(
        f"找不到 VOC{year} 数据集，已尝试：{', '.join(str(c) for c in candidates)}\n"
        f"请确认存在 JPEGImages/ 与 ImageSets/Segmentation/ 两个目录。"
    )


def decode_mask(mask: np.ndarray, ignore_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """把 0..20/255 的标签图转成 RGB 彩色图。"""
    mask = np.asarray(mask)
    rgb = np.zeros(mask.shape + (3,), dtype=np.uint8)
    palette = np.array(VOC_PALETTE, dtype=np.uint8)
    valid = mask < len(VOC_PALETTE)
    rgb[valid] = palette[mask[valid]]
    rgb[~valid] = np.array(ignore_color, dtype=np.uint8)
    return rgb


class VOCSegmentation(Dataset):
    """VOC2012 分割数据集。

    Args:
        root:     数据集根目录（data/ 或 data/VOC2012 均可）。
        split:    train / val / trainval。
        transform: 接受 (PIL image, PIL mask) 返回 (tensor, tensor, meta) 的可调用对象；
                   为 None 时使用验证变换。
        year:     VOC 年份。
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Optional[Any] = None,
        year: str = "2012",
        ignore_index: int = 255,
    ):
        self.root = resolve_voc_root(root, year)
        self.split = split
        self.ignore_index = ignore_index

        self.images_dir = self.root / "JPEGImages"
        self.masks_dir = self.root / "SegmentationClass"

        split_file = self.root / "ImageSets" / "Segmentation" / f"{split}.txt"
        if not split_file.is_file():
            raise FileNotFoundError(f"找不到划分文件：{split_file}")
        self.ids: List[str] = [
            line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        if not self.ids:
            raise RuntimeError(f"{split_file} 为空")

        self.transform = transform if transform is not None else EvalTransform(ignore_index=ignore_index)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image_id = self.ids[index]
        image = Image.open(self.images_dir / f"{image_id}.jpg").convert("RGB")
        mask = Image.open(self.masks_dir / f"{image_id}.png")
        # VOC 的 GT 是调色板 PNG：P 模式下 resize/裁剪可能经调色板转换而改写标签值，
        # 先转成"值即类别索引"的 L 图，后续几何变换只做 NEAREST，索引原样保留。
        mask = Image.fromarray(np.array(mask))

        image_t, mask_t, meta = self.transform(image, mask)
        return {
            "image": image_t,
            "mask": mask_t,
            "image_id": image_id,
            "meta": meta,
        }


def collate_seg(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """默认 collate 处理不了字符串，这里保留 image_id / meta 为 list。

    注意：验证集各图尺寸不同，只能 batch_size=1；训练集因固定裁剪可以凑 batch。
    """
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
        "image_id": [item["image_id"] for item in batch],
        "meta": [item["meta"] for item in batch],
    }
