"""DANet 双注意力模块。

Implementation follows the reference code of
    Fu et al., "Dual Attention Network for Scene Segmentation", CVPR 2019.
    https://github.com/junfu1115/DANet  (encoding/nn/da_att.py)

两个模块都写成 "残差式"：out = gamma * attn_out + x，
其中 gamma 初始化为 0，即训练开始时模块是恒等映射，注意力分支逐渐"接管"。

与论文/官方代码一致的几个容易写错的点：
  1. PAM 的 query/key 用 1x1 conv 把通道降到 C//8，value 保持 C；
  2. PAM 的相似度**不做** sqrt(d) 缩放，直接 softmax；
  3. CAM 不接任何 conv，直接把 x 当成 query/key/value；
  4. CAM 在 softmax 前做了 max-shift：energy = max(energy, -1) - energy。
     这不是数值稳定性技巧，而是论文里的取相反数操作（越相似权重越大）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PAM(nn.Module):
    """位置注意力模块 Position Attention Module。

    对特征图做 N x N 的像素级自注意力（N = H*W），任意两个位置之间建立直接联系，
    用于缓解卷积感受野有限导致的大面积区域碎片化。

    Args:
        in_channels: 输入通道数 C。
        reduction:   query/key 的通道压缩比，官方取 8。
        record_attn: 是否缓存注意力图（BxNxN，很大，只在可视化时打开）。
    """

    def __init__(self, in_channels: int, reduction: int = 8, record_attn: bool = False):
        super().__init__()
        inter_channels = max(in_channels // reduction, 1)
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.record_attn = record_attn

        self.query_conv = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.last_attn: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        # q: B x N x C'   k: B x C' x N
        q = self.query_conv(x).flatten(2).transpose(1, 2)
        k = self.key_conv(x).flatten(2)
        energy = torch.bmm(q, k)                       # B x N x N
        attn = F.softmax(energy, dim=-1)               # 沿 key 位置归一化

        v = self.value_conv(x).flatten(2)              # B x C x N
        out = torch.bmm(v, attn.transpose(1, 2)).view(b, c, h, w)

        if self.record_attn:
            self.last_attn = attn.detach()

        return self.gamma * out + x


class CAM(nn.Module):
    """通道注意力模块 Channel Attention Module。

    在通道维计算 C x C 相似度，每个通道被重新加权为该图像所有通道的凸组合，
    相当于自适应的通道重标定（比 SE 的全局池化更细粒度的 "通道-通道" 依赖）。

    Args:
        in_channels: 输入通道数 C。
        record_attn: 是否缓存注意力图（BxCxC，很小，可视化几乎零成本）。
    """

    def __init__(self, in_channels: int, record_attn: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.record_attn = record_attn
        self.gamma = nn.Parameter(torch.zeros(1))
        self.last_attn: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        x_flat = x.flatten(2)                                  # B x C x N
        energy = torch.bmm(x_flat, x_flat.transpose(1, 2))     # B x C x C

        # 官方实现：max-shift 后取负数，相似度越大权重越大
        energy = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attn = F.softmax(energy, dim=-1)                       # B x C x C

        out = torch.bmm(attn, x_flat).view(b, c, h, w)

        if self.record_attn:
            self.last_attn = attn.detach()

        return self.gamma * out + x
