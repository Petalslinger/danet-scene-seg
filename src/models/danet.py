"""DANet 场景分割模型。

结构（对应 README 架构图）：

    RGB ──> DilatedResNet(output_stride=8) ──> DANet 头 ──> 1x1 分类 ──> 双线性上采样 ──> logits

DANet 头对齐官方实现（encoding/models/sseg/danet.py 中的 DANetHead）：
  * 两个 3x3 conv 分别把 backbone 特征降到 inter_channels = in_channels // 4；
  * PAM 与 CAM 各走一支，各自再过一层 3x3 conv；
  * 两支结果**逐元素相加**，再 1x1 卷积出主输出；
  * 每支另有自己的 1x1 分类头（sa_output / sc_output），训练时作为辅助损失
    （官方 aux_weight = 0.4）。这是 DANet 原文的一部分，不是可选的 trick；
    只留 PAM 或只留 CAM 时，这两个辅助头天然就是消融实验。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import PAM, CAM
from .backbone import DilatedResNet

__all__ = ["DANetConfig", "DANetHead", "DANet", "DANetOutput"]


def conv_bn_relu(in_channels: int, out_channels: int, kernel_size: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


# none = baseline（无注意力，配置 A）；pam = 单模块消融（配置 B）；cam = 单模块消融（配置 C）；
# both = 完整 DANet（配置 D）
_ATTENTION_CHOICES = ("none", "pam", "cam", "both")


@dataclass
class DANetConfig:
    """模型超参。默认值对应 README 的配置 D（完整 DANet）。"""

    num_classes: int = 21
    backbone: str = "resnet50"
    pretrained: bool = True
    output_stride: int = 8
    use_layer4: bool = False
    # none = 无注意力 baseline（配置 A）；pam = 单模块消融（配置 B）；cam = 单模块消融（配置 C）；
    # both = 完整 DANet（配置 D）
    attention: str = "both"
    inter_reduction: int = 4       # inter_channels = in_channels // 4
    dropout: float = 0.1
    aux_loss: bool = True
    # 缓存注意力图用于画热力图。PAM 的是 BxNxN（512 输入时 N=4096，约 64MB/张，fp32），
    # 训练时务必保持 False。
    record_attn: bool = False
    align_corners: bool = False

    def validate(self) -> None:
        if self.attention not in _ATTENTION_CHOICES:
            raise ValueError(f"attention 只能是 {_ATTENTION_CHOICES}，收到 {self.attention!r}")
        if self.inter_reduction < 1:
            raise ValueError("inter_reduction 必须 >= 1")


@dataclass
class DANetOutput:
    """一次前向的输出，logits 均已上采样到输入分辨率。"""

    logits: torch.Tensor                               # B x num_classes x H x W
    aux: Dict[str, torch.Tensor] = field(default_factory=dict)   # pam / cam 分支
    attn: Dict[str, torch.Tensor] = field(default_factory=dict)  # 注意力图（按需）

    def argmax(self) -> torch.Tensor:
        return self.logits.argmax(dim=1)


class DANetHead(nn.Module):
    """DANet 双注意力头（输出 stride=8 的特征上运行）。"""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        attention: str = "both",
        inter_reduction: int = 4,
        dropout: float = 0.1,
        aux_loss: bool = True,
        record_attn: bool = False,
    ):
        super().__init__()
        if attention not in _ATTENTION_CHOICES:
            raise ValueError(f"attention 只能是 {_ATTENTION_CHOICES}，收到 {attention!r}")

        self.attention = attention
        self.use_attn = attention != "none"
        self.use_pam = attention in ("both", "pam")
        self.use_cam = attention in ("both", "cam")
        self.aux_loss = aux_loss

        inter_channels = max(in_channels // inter_reduction, 1)

        if not self.use_attn:
            # baseline：3x3 卷积直接接分类头（配置 A）
            self.conv_no_attn = conv_bn_relu(in_channels, inter_channels)

        if self.use_pam:
            self.conv_proj_pam = conv_bn_relu(in_channels, inter_channels)
            self.pam = PAM(inter_channels, record_attn=record_attn)
            self.conv_out_pam = conv_bn_relu(inter_channels, inter_channels)
            if aux_loss:
                self.cls_pam = nn.Sequential(
                    nn.Dropout2d(dropout), nn.Conv2d(inter_channels, num_classes, 1)
                )

        if self.use_cam:
            self.conv_proj_cam = conv_bn_relu(in_channels, inter_channels)
            self.cam = CAM(inter_channels, record_attn=record_attn)
            self.conv_out_cam = conv_bn_relu(inter_channels, inter_channels)
            if aux_loss:
                self.cls_cam = nn.Sequential(
                    nn.Dropout2d(dropout), nn.Conv2d(inter_channels, num_classes, 1)
                )

        # 两支特征逐元素相加，通道数仍是 inter_channels
        self.cls_main = nn.Sequential(
            nn.Dropout2d(dropout), nn.Conv2d(inter_channels, num_classes, 1)
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # 分类头（1x1）用很小的初始化：21 类 logits 在训练初期不至于过大
        for head_name in ("cls_main", "cls_pam", "cls_cam"):
            head = getattr(self, head_name, None)
            if head is None:
                continue
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if not self.use_attn:
            return self.cls_main(self.conv_no_attn(x)), {}, {}

        feats = []
        aux: Dict[str, torch.Tensor] = {}
        attn: Dict[str, torch.Tensor] = {}

        if self.use_pam:
            feat = self.conv_proj_pam(x)
            pam_feat = self.pam(feat)
            pam_out = self.conv_out_pam(pam_feat)
            feats.append(pam_out)
            if self.aux_loss:
                aux["pam"] = self.cls_pam(pam_out)
            if self.pam.record_attn and self.pam.last_attn is not None:
                attn["pam"] = self.pam.last_attn

        if self.use_cam:
            feat = self.conv_proj_cam(x)
            cam_feat = self.cam(feat)
            cam_out = self.conv_out_cam(cam_feat)
            feats.append(cam_out)
            if self.aux_loss:
                aux["cam"] = self.cls_cam(cam_out)
            if self.cam.record_attn and self.cam.last_attn is not None:
                attn["cam"] = self.cam.last_attn

        fused = feats[0] if len(feats) == 1 else feats[0] + feats[1]
        main = self.cls_main(fused)
        return main, aux, attn


class DANet(nn.Module):
    """完整的场景分割网络。"""

    def __init__(self, config: Optional[DANetConfig] = None, **overrides):
        super().__init__()
        if config is None:
            config = DANetConfig()
        if overrides:
            config = DANetConfig(**{**config.__dict__, **overrides})
        config.validate()
        self.config = config

        self.backbone = DilatedResNet(
            arch=config.backbone,
            pretrained=config.pretrained,
            output_stride=config.output_stride,
            use_layer4=config.use_layer4,
        )
        self.head = DANetHead(
            in_channels=self.backbone.out_channels,
            num_classes=config.num_classes,
            attention=config.attention,
            inter_reduction=config.inter_reduction,
            dropout=config.dropout,
            aux_loss=config.aux_loss,
            record_attn=config.record_attn,
        )

    # ------------------------------------------------------------------ #
    # 便捷构造
    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, **kwargs) -> "DANet":
        """DANet.from_config(num_classes=21, attention='cam', pretrained=False)"""
        return cls(DANetConfig(**kwargs))

    # ------------------------------------------------------------------ #
    # 前向
    # ------------------------------------------------------------------ #
    def forward(
        self,
        x: torch.Tensor,
        return_attn: Optional[bool] = None,
    ) -> DANetOutput:
        if return_attn is None:
            return_attn = self.config.record_attn

        hw = x.shape[-2:]
        feat = self.backbone(x)
        main, aux, attn = self.head(feat)

        logits = self._upsample(main, hw)
        aux = {name: self._upsample(t, hw) for name, t in aux.items()}

        if not return_attn:
            attn = {}
        return DANetOutput(logits=logits, aux=aux, attn=attn)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """推理：返回 B x H x W 的类别索引图（调用前请确保 model.eval()）。"""
        with torch.no_grad():
            return self.forward(x).argmax()

    def set_record_attn(self, flag: bool = True) -> None:
        """打开/关闭注意力图缓存（画热力图前打开，训练时保持关闭）。"""
        for name in ("pam", "cam"):
            module = getattr(self.head, name, None)
            if module is not None:
                module.record_attn = flag

    def _upsample(self, t: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        return F.interpolate(
            t, size=hw, mode="bilinear", align_corners=self.config.align_corners
        )
