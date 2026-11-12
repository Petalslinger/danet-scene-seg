"""分割任务的损失：主输出 + DANet 的两路辅助输出。

DANet 原文训练时对 sa_output / sc_output 也施加交叉熵（aux_weight = 0.4），
本文件沿用该设置。`ignore_index=255` 对应 VOC 的边界忽略像素。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models.danet import DANetOutput

__all__ = ["SegmentationLoss"]


class SegmentationLoss(nn.Module):
    """交叉熵 + DANet 辅助损失。

    Args:
        ignore_index:  忽略的标签值（VOC 边界为 255）。
        aux_weight:    辅助损失权重，官方 0.4；设为 0 即只用主输出。
        class_weights: 可选的每类权重（缓解类别不平衡），形状 (num_classes,)。
    """

    def __init__(
        self,
        ignore_index: int = 255,
        aux_weight: float = 0.4,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.aux_weight = aux_weight
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)

    def forward(
        self,
        output: DANetOutput,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """返回 (总损失, 各分项损失的 detach 版本，用于日志)。"""
        main_loss = self._ce(output.logits, target)
        total = main_loss
        parts: Dict[str, torch.Tensor] = {"main": main_loss.detach()}

        if self.aux_weight > 0:
            for name, logits in output.aux.items():
                aux = self._ce(logits, target)
                total = total + self.aux_weight * aux
                parts[name] = aux.detach()

        return total, parts

    def _ce(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 标签尺寸和 logits 不一致时（例如做了 resize），用最近邻对齐
        if target.shape[-2:] != logits.shape[-2:]:
            target = F.interpolate(
                target.unsqueeze(1).float(), size=logits.shape[-2:], mode="nearest"
            ).squeeze(1).long()
        return F.cross_entropy(
            logits,
            target,
            ignore_index=self.ignore_index,
            weight=self.class_weights,
        )
