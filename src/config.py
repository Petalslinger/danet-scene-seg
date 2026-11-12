"""训练超参与实验配置。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional

from .models.danet import DANetConfig

__all__ = ["TrainConfig"]


@dataclass
class TrainConfig:
    # ---------------- 数据 ----------------
    data_root: str = "data"
    crop_size: int = 512
    batch_size: int = 8
    num_workers: int = 4

    # ---------------- 模型 ----------------
    attention: str = "both"        # none / pam / cam / both（对应 A / B / C / D）
    backbone: str = "resnet50"
    pretrained: bool = True
    output_stride: int = 8
    use_layer4: bool = False
    num_classes: int = 21

    # ---------------- 优化 ----------------
    epochs: int = 50
    lr: float = 1e-2
    momentum: float = 0.9
    weight_decay: float = 1e-4
    poly_power: float = 0.9
    aux_weight: float = 0.4        # DANet 辅助损失权重
    grad_clip: float = 1.0
    amp: bool = True

    # ---------------- 其他 ----------------
    ignore_index: int = 255
    seed: int = 42
    output_dir: str = "outputs"
    val_interval: int = 1
    save_interval: int = 5
    log_interval: int = 20
    progress: bool = True          # 终端进度条（--no-progress 关闭）
    resume: Optional[str] = None   # 从 ckpt_last.pth 继续训练

    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        if self.crop_size % 8 != 0:
            raise ValueError("crop_size 必须是 8 的倍数")
        if self.attention not in ("none", "pam", "cam", "both"):
            raise ValueError(f"attention 非法：{self.attention}")
        if self.batch_size < 1 or self.epochs < 1:
            raise ValueError("batch_size / epochs 必须为正")

    @property
    def experiment_name(self) -> str:
        """用于 checkpoint 与 CSV 文件名，例如 danet_resnet50_both。"""
        prefix = "baseline" if self.attention == "none" else "danet"
        return f"{prefix}_{self.backbone}_{self.attention}"

    def model_config(self) -> DANetConfig:
        return DANetConfig(
            num_classes=self.num_classes,
            backbone=self.backbone,
            pretrained=self.pretrained,
            output_stride=self.output_stride,
            use_layer4=self.use_layer4,
            attention=self.attention,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainConfig":
        """忽略多余字段（例如旧 checkpoint 里已删除的键）。"""
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in allowed})
