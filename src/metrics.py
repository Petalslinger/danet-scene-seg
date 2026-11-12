"""分割指标：用混淆矩阵一次性算出 mIoU / PA / MA / 逐类 IoU。

约定：
  * 只统计 target != ignore_index 的像素；
  * mIoU 只对「在 GT 或预测中出现过」的类取平均（与 VOC 官方评测一致），
    全零类会拉低分数且没有意义；
  * PA = 对像素取平均；MA = 对类别召回率取平均（只在 GT 中出现过的类上平均）。
"""

from __future__ import annotations

from typing import Any, Dict

import torch

__all__ = ["SegmentationMetrics"]


class SegmentationMetrics:
    """累加混淆矩阵，compute() 在最后一次性求指标。"""

    def __init__(self, num_classes: int = 21, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.hist = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    def reset(self) -> None:
        self.hist.zero_()

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """pred / target: B x H x W 的整数张量。"""
        if pred.shape != target.shape:
            raise ValueError(f"pred 与 target 形状不一致: {tuple(pred.shape)} vs {tuple(target.shape)}")

        pred = pred.detach().reshape(-1).to(torch.int64).cpu()
        target = target.detach().reshape(-1).to(torch.int64).cpu()
        valid = target != self.ignore_index
        pred, target = pred[valid], target[valid]
        if pred.numel() == 0:
            return

        k = self.num_classes
        index = target * k + pred
        hist = torch.bincount(index, minlength=k * k).reshape(k, k)
        self.hist += hist

    def compute(self) -> Dict[str, Any]:
        hist = self.hist.to(torch.float64)
        tp = torch.diag(hist)
        fp = hist.sum(dim=0) - tp
        fn = hist.sum(dim=1) - tp
        union = tp + fp + fn

        iou = torch.zeros_like(tp)
        present = union > 0
        iou[present] = tp[present] / union[present]

        gt_freq = hist.sum(dim=1)
        recall = torch.zeros_like(tp)
        in_gt = gt_freq > 0
        recall[in_gt] = tp[in_gt] / gt_freq[in_gt]

        total = hist.sum()
        pixel_acc = float(tp.sum() / total) if total > 0 else 0.0

        return {
            "miou": float(iou[present].mean()) if present.any() else 0.0,
            "pixel_acc": pixel_acc,
            "mean_acc": float(recall[in_gt].mean()) if in_gt.any() else 0.0,
            "per_class_iou": iou.tolist(),
            "per_class_recall": recall.tolist(),
            "num_present_classes": int(present.sum()),
        }


def format_per_class(iou: list, class_names: list) -> str:
    """把逐类 IoU 排成可以直接 print 的表格。"""
    rows = sorted(zip(class_names, iou), key=lambda x: x[1], reverse=True)
    lines = [f"{'class':<14}{'IoU':>8}"]
    for name, value in rows:
        lines.append(f"{name:<14}{value * 100:7.2f}%")
    return "\n".join(lines)
