"""DANet 自检脚本：不依赖数据集，用随机张量验证形状、损失与反向传播。

用法（在项目根目录，需要装好 torch）：
    python scripts/smoke_test.py                  # 默认 256x256, batch 2, CPU
    python scripts/smoke_test.py --size 512 --batch 1 --device cuda

检查项：
  1. 四种 attention 配置（none/pam/cam/both）能否前向；
  2. logits / aux 输出形状是否等于输入分辨率；
  3. 注意力图形状是否为 PAM: BxNxN、CAM: BxCxC；
  4. 损失能否计算，反向传播能否跑到 gamma 参数上。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.losses import SegmentationLoss  # noqa: E402
from src.models import DANet  # noqa: E402


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def check_config(name: str, size: int, batch: int, device: torch.device) -> None:
    model = DANet.from_config(
        num_classes=21, attention=name, pretrained=False, record_attn=True
    ).to(device).eval()

    x = torch.randn(batch, 3, size, size, device=device)
    with torch.no_grad():
        out = model(x, return_attn=True)

    assert out.logits.shape == (batch, 21, size, size), out.logits.shape
    expected_aux = 0 if name == "none" else (1 if name in ("pam", "cam") else 2)
    assert len(out.aux) == expected_aux, (name, list(out.aux))

    n = (size // 8) ** 2
    c_inter = 1024 // 4  # inter_channels = 256（ResNet-50 layer3 输出 1024）
    if name in ("pam", "both"):
        assert out.attn["pam"].shape == (batch, n, n), out.attn["pam"].shape
    if name in ("cam", "both"):
        assert out.attn["cam"].shape == (batch, c_inter, c_inter), out.attn["cam"].shape

    print(
        f"[ok] attention={name:<5} logits={tuple(out.logits.shape)} "
        f"aux={list(out.aux)} attn={ {k: tuple(v.shape) for k, v in out.attn.items()} } "
        f"params={count_params(model) / 1e6:.1f}M"
    )


def check_backward(size: int, batch: int, device: torch.device) -> None:
    model = DANet.from_config(attention="both", pretrained=False).to(device).train()
    criterion = SegmentationLoss(ignore_index=255, aux_weight=0.4)

    x = torch.randn(batch, 3, size, size, device=device)
    target = torch.randint(0, 21, (batch, size, size), device=device)
    target[:, :4, :4] = 255  # 模拟边界忽略像素

    out = model(x)
    loss, parts = criterion(out, target)
    loss.backward()

    assert torch.isfinite(loss), loss
    gammas = {
        "pam": model.head.pam.gamma.grad,
        "cam": model.head.cam.gamma.grad,
    }
    for key, grad in gammas.items():
        assert grad is not None and torch.isfinite(grad).all(), f"{key}.gamma 无梯度"

    print(
        f"[ok] backward loss={loss.item():.4f} "
        f"parts={ {k: round(v.item(), 4) for k, v in parts.items()} } "
        f"grad(pam.gamma)={gammas['pam'].item():.3e} grad(cam.gamma)={gammas['cam'].item():.3e}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="DANet 形状自检")
    parser.add_argument("--size", type=int, default=256, help="输入分辨率（需能被 8 整除）")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    if args.size % 8 != 0:
        parser.error("size 必须是 8 的倍数")
    device = torch.device(args.device)

    for name in ("none", "pam", "cam", "both"):
        check_config(name, args.size, args.batch, device)

    check_backward(args.size, args.batch, device)
    print("全部通过。")


if __name__ == "__main__":
    main()
