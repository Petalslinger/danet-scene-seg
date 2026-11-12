"""可视化：三联图 / PAM 位置注意力热力图 / CAM 通道注意力图。

用法：
    python scripts/visualize.py --ckpt outputs/danet_resnet50_both/ckpt_best.pth --mode triple
    python scripts/visualize.py --ckpt ... --mode pam --ids 2007_000032 2007_000129
    python scripts/visualize.py --ckpt ... --mode all

产出：figures/<mode>_<image_id>.png（dpi 200，可直接贴进报告）

需要 matplotlib；没有的话脚本会给出明确提示而不是抛栈。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import TrainConfig  # noqa: E402
from src.data import VOCSegmentation, decode_mask  # noqa: E402
from src.data.transforms import EvalTransform  # noqa: E402
from src.models import DANet  # noqa: E402
from src.models.danet import DANetConfig  # noqa: E402
from src.utils import ensure_dir  # noqa: E402

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DANet 可视化")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--ids", nargs="*", default=None, help="不指定则取前 --num 张")
    parser.add_argument("--num", type=int, default=4)
    parser.add_argument("--mode", choices=["triple", "pam", "cam", "all"], default="triple")
    parser.add_argument("--out-dir", default="figures")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_model(ckpt_path: str | Path, device: torch.device) -> tuple[DANet, TrainConfig]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = TrainConfig.from_dict(ckpt.get("config", {}))
    model = DANet(
        DANetConfig(
            num_classes=cfg.num_classes,
            backbone=cfg.backbone,
            pretrained=False,
            output_stride=cfg.output_stride,
            use_layer4=cfg.use_layer4,
            attention=cfg.attention,
            record_attn=True,
        )
    )
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device).eval()
    model.set_record_attn(True)
    return model, cfg


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def crop_valid(arr: np.ndarray, pad: tuple[int, int]) -> np.ndarray:
    """裁掉右下角的 pad 区。"""
    pad_h, pad_w = pad
    h, w = arr.shape[:2]
    return arr[: h - pad_h, : w - pad_w]


def attention_to_map(attn_row: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    """把某个 query 的注意力向量 reshape 成特征图并上采样到原图尺寸。"""
    hf, wf = size
    heat = attn_row.reshape(1, 1, hf, wf)
    up = torch.nn.functional.interpolate(heat, size=None, scale_factor=8, mode="bilinear", align_corners=False)
    heat = up[0, 0].detach().cpu().numpy()
    heat = heat - heat.min()
    denom = heat.max() if heat.max() > 0 else 1.0
    return heat / denom


def pick_query(pred: np.ndarray, feature_hw: tuple[int, int]) -> tuple[int, int]:
    """挑一个「预测为该图面积最大的前景类」的质心作为 query。

    返回 (h, w) 的**特征图坐标**。
    """
    hf, wf = feature_hw
    values, counts = np.unique(pred, return_counts=True)
    order = np.argsort(counts)[::-1]
    target = next((values[i] for i in order if values[i] != 0), None)
    if target is None:  # 全是背景就退化成图中心
        return hf // 2, wf // 2

    ys, xs = np.mgrid[0 : pred.shape[0], 0 : pred.shape[1]]
    mask = pred == target
    cy = int(round(ys[mask].mean() / 8))
    cx = int(round(xs[mask].mean() / 8))
    return min(max(cy, 0), hf - 1), min(max(cx, 0), wf - 1)


def main() -> None:
    args = parse_args()
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("需要 matplotlib：pip install matplotlib")

    device = torch.device(args.device)
    model, cfg = load_model(args.ckpt, device)
    data_root = args.data_root or cfg.data_root
    dataset = VOCSegmentation(
        data_root, split=args.split, transform=EvalTransform(pad_to=8, ignore_index=cfg.ignore_index)
    )

    ids = args.ids if args.ids else dataset.ids[: args.num]
    unknown = [i for i in ids if i not in dataset.ids]
    if unknown:
        raise SystemExit(f"这些 id 不在 {args.split} 里：{unknown}")

    out_dir = ensure_dir(args.out_dir)
    has_pam = cfg.attention in ("both", "pam")
    has_cam = cfg.attention in ("both", "cam")

    for image_id in ids:
        index = dataset.ids.index(image_id)
        item = dataset[index]
        image_t, mask_t, meta = item["image"], item["mask"], item["meta"]

        with torch.no_grad():
            out = model(image_t.unsqueeze(0).to(device), return_attn=True)
        pred = out.logits.argmax(dim=1)[0].cpu().numpy()
        gt = mask_t.numpy()
        pad = tuple(meta.get("pad", (0, 0)))

        image_rgb = crop_valid(denormalize(image_t), pad)
        gt_rgb = crop_valid(decode_mask(gt), pad)
        pred_rgb = crop_valid(decode_mask(pred), pad)
        pred_valid = crop_valid(pred, pad)

        if args.mode in ("triple", "all"):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for ax, arr, title in zip(
                axes,
                [image_rgb, gt_rgb, pred_rgb],
                ["Image", "Ground truth", "Prediction"],
            ):
                ax.imshow(arr)
                ax.set_title(title)
                ax.axis("off")
            fig.tight_layout()
            path = out_dir / f"triple_{image_id}.png"
            fig.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"saved {path}")

        if args.mode in ("pam", "all") and has_pam:
            attn = out.attn.get("pam")
            if attn is not None:
                hp, wp = image_t.shape[-2:]
                feature_hw = (hp // 8, wp // 8)
                cy, cx = pick_query(pred_valid, feature_hw)
                row = attn[0, cy * feature_hw[1] + cx]
                heat = attention_to_map(row, feature_hw)[: pred_valid.shape[0], : pred_valid.shape[1]]

                fig, axes = plt.subplots(1, 2, figsize=(11, 5))
                axes[0].imshow(image_rgb)
                axes[0].plot(cx * 8, cy * 8, "r+", markersize=14, markeredgewidth=2)
                axes[0].set_title("Query position")
                axes[0].axis("off")
                axes[1].imshow(image_rgb)
                axes[1].imshow(heat, cmap="jet", alpha=0.55)
                axes[1].set_title("PAM attention")
                axes[1].axis("off")
                fig.tight_layout()
                path = out_dir / f"pam_{image_id}.png"
                fig.savefig(path, dpi=200, bbox_inches="tight")
                plt.close(fig)
                print(f"saved {path}")

        if args.mode in ("cam", "all") and has_cam:
            attn = out.attn.get("cam")
            if attn is not None:
                matrix = attn[0].cpu().numpy()
                fig, ax = plt.subplots(figsize=(6, 5))
                im = ax.imshow(matrix, cmap="viridis")
                ax.set_title("CAM channel affinity (C x C)")
                ax.set_xlabel("key channel")
                ax.set_ylabel("query channel")
                fig.colorbar(im, ax=ax, fraction=0.046)
                fig.tight_layout()
                path = out_dir / f"cam_{image_id}.png"
                fig.savefig(path, dpi=200, bbox_inches="tight")
                plt.close(fig)
                print(f"saved {path}")

    if args.mode in ("pam", "all") and not has_pam:
        print(f"注意：该 checkpoint 的 attention={cfg.attention}，没有 PAM，跳过。")
    if args.mode in ("cam", "all") and not has_cam:
        print(f"注意：该 checkpoint 的 attention={cfg.attention}，没有 CAM，跳过。")
    print(f"全部完成，输出目录：{out_dir}")


if __name__ == "__main__":
    main()
