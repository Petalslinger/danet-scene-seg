"""数据相关入口。"""

from .transforms import EvalTransform, TrainTransform, mask_to_tensor, to_normalized_tensor
from .voc import (
    VOC_CLASSES,
    VOC_PALETTE,
    VOCSegmentation,
    collate_seg,
    decode_mask,
    resolve_voc_root,
)

__all__ = [
    "VOCSegmentation",
    "VOC_CLASSES",
    "VOC_PALETTE",
    "collate_seg",
    "decode_mask",
    "resolve_voc_root",
    "TrainTransform",
    "EvalTransform",
    "mask_to_tensor",
    "to_normalized_tensor",
]
