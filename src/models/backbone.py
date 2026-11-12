"""Dilated ResNet 编码器（torchvision 权重 + 手动改 stride/dilation）。

分数下降的典型做法（DeepLab 系列）：不去继续下采样，而是把 stride 换成 dilation，
在保持 stride=8 分辨率的同时扩大感受野。

本文件的取舍：
  * output_stride=8, use_layer4=False  -> 1024 x 64 x 64（输入 512），与 README 架构图一致，
    也是课程作业更省显存的版本；
  * use_layer4=True                    -> 2048 x 64 x 64，对齐官方 DANet 的配置。
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torchvision.models as tvm

SUPPORTED_ARCHS = ("resnet50", "resnet101", "resnet152")


def _torchvision_weights(arch: str):
    """返回 torchvision 的 ImageNet 权重枚举，取不到就返回 None（离线环境）。"""
    name = {
        "resnet50": "ResNet50_Weights",
        "resnet101": "ResNet101_Weights",
        "resnet152": "ResNet152_Weights",
    }[arch]
    enum = getattr(tvm, name, None)
    if enum is None:
        return None
    try:
        return enum.IMAGENET1K_V1
    except AttributeError:  # 老版本 torchvision 用 V2
        return enum.IMAGENET1K_V2


def _make_dilated(layer: nn.Module, dilation: int) -> nn.Module:
    """把某个 stage 的 stride-2 下采样替换成 dilated 卷积，输出 stride 不变。

    torchvision 的 Bottleneck 中：conv1/conv2 是 3x3（stride 2 出现在 conv2），
    downsample 是 1x1 stride 2，两者都要改。
    """
    for m in layer.modules():
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
            if m.kernel_size == (3, 3):
                m.dilation = (dilation, dilation)
                m.padding = (dilation, dilation)
    return layer


class DilatedResNet(nn.Module):
    """ResNet 的 conv1~layer3(/layer4) 部分，输出 stride 可控的特征图。

    Args:
        arch:          resnet50 / resnet101 / resnet152。
        pretrained:    是否加载 ImageNet 预训练权重（默认是；离线时会告警并随机初始化）。
        output_stride: 8 或 16，最终特征图相对输入的下采样倍率。
        use_layer4:    是否使用 layer4。False 时输出 1024 通道、更省显存。
        freeze_stem:   冻结 conv1 与 layer1（显存/时间紧张时的降级选项）。
    """

    def __init__(
        self,
        arch: str = "resnet50",
        pretrained: bool = True,
        output_stride: int = 8,
        use_layer4: bool = False,
        freeze_stem: bool = False,
    ):
        super().__init__()
        if arch not in SUPPORTED_ARCHS:
            raise ValueError(f"不支持的 backbone: {arch}，可选 {SUPPORTED_ARCHS}")
        if output_stride not in (8, 16):
            raise ValueError("output_stride 只支持 8 或 16")

        self.arch = arch
        self.output_stride = output_stride
        self.use_layer4 = use_layer4

        weights = _torchvision_weights(arch) if pretrained else None
        try:
            model = getattr(tvm, arch)(weights=weights)
        except Exception as exc:  # 典型情况：离线 / 无缓存
            warnings.warn(f"加载 {arch} 预训练权重失败（{exc}），改为随机初始化。")
            model = getattr(tvm, arch)(weights=None)

        # 骨架：conv1 -> bn -> relu -> maxpool（共 stride 4）
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4 if use_layer4 else None

        # layer1/layer2 固定为 stride 8 的中间态：stem(4) * layer2(2) = 8
        # output_stride=8 时 layer3 用 dilation=2，保持 stride 8
        # output_stride=16 时 layer3 保持默认 stride 2，得到 stride 16
        if output_stride == 8:
            _make_dilated(self.layer3, dilation=2)
            if self.layer4 is not None:
                _make_dilated(self.layer4, dilation=4)
        else:
            if self.layer4 is not None:
                _make_dilated(self.layer4, dilation=2)

        self.out_channels = 2048 if use_layer4 else 1024

        self.frozen_stem = freeze_stem
        if freeze_stem:
            for p in self.stem.parameters():
                p.requires_grad_(False)
            for p in self.layer1.parameters():
                p.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        # 冻结的部分即使整体 .train() 也要保持在 eval 模式（BN 用滑动统计量）
        if self.frozen_stem:
            self.stem.eval()
            self.layer1.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.layer4 is not None:
            x = self.layer4(x)
        return x
