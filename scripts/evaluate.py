"""评测一个 checkpoint：导出 mIoU/PA/MA 与逐类 IoU。

用法：
    python scripts/evaluate.py --ckpt outputs/danet_resnet50_both/ckpt_best.pth

产出：
    outputs/metrics_summary.csv          所有评测过的 checkpoint 汇总（追加）
    outputs/<name>/per_class_iou.csv      逐类 IoU（21 行）
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import TrainConfig  # noqa: E402
from src.data import EvalTransform, VOCSegmentation, VOC_CLASSES, collate_seg  # noqa: E402
from src.engine import evaluate  # noqa: E402
from src.models import DANet  # noqa: E402
from src.models.danet import DANetConfig  # noqa: E402
from src.utils import ensure_dir, save_json, seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评测 DANet checkpoint")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data-root", default=None, help="默认沿用 checkpoint 里的配置")
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--tag", default=None, help="汇总表里的实验名，默认用 ckpt 所在目录名")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def load_model(ckpt_path: str | Path, device: torch.device) -> tuple[DANet, TrainConfig]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = TrainConfig.from_dict(ckpt.get("config", {}))

    # 评测时不需要 ImageNet 权重，避免联网下载
    model = DANet(
        DANetConfig(
            num_classes=cfg.num_classes,
            backbone=cfg.backbone,
            pretrained=False,
            output_stride=cfg.output_stride,
            use_layer4=cfg.use_layer4,
            attention=cfg.attention,
        )
    )
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if unexpected:
        raise RuntimeError(f"checkpoint 里有未知参数：{unexpected[:5]}")
    if missing:
        print(f"警告：{len(missing)} 个参数缺失（{'、'.join(missing[:3])} …）")
    return model.to(device).eval(), cfg


def main() -> None:
    args = parse_args()
    seed_everything(0)
    device = torch.device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")

    model, cfg = load_model(args.ckpt, device)
    data_root = args.data_root or cfg.data_root

    dataset = VOCSegmentation(
        data_root,
        split=args.split,
        transform=EvalTransform(pad_to=8, ignore_index=cfg.ignore_index),
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_seg,
    )

    metrics = evaluate(model, loader, cfg.num_classes, cfg.ignore_index, device, amp=amp_enabled)

    tag = args.tag or Path(args.ckpt).parent.name
    out_dir = ensure_dir(Path(args.output_dir) / tag)
    save_json(metrics, out_dir / "metrics.json")

    with (out_dir / "per_class_iou.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "iou", "recall"])
        for name, iou, recall in zip(VOC_CLASSES, metrics["per_class_iou"], metrics["per_class_recall"]):
            writer.writerow([name, f"{iou:.6f}", f"{recall:.6f}"])

    summary = Path(args.output_dir) / "metrics_summary.csv"
    write_header = not summary.exists() or summary.stat().st_size == 0
    with summary.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["experiment", "attention", "backbone", "split", "miou", "pixel_acc", "mean_acc", "ckpt"])
        writer.writerow(
            [
                tag,
                cfg.attention,
                cfg.backbone,
                args.split,
                f"{metrics['miou']:.6f}",
                f"{metrics['pixel_acc']:.6f}",
                f"{metrics['mean_acc']:.6f}",
                str(args.ckpt),
            ]
        )

    print(f"[{tag}] split={args.split}  n={len(dataset)}")
    print(f"  mIoU = {metrics['miou'] * 100:.2f}")
    print(f"  PA   = {metrics['pixel_acc'] * 100:.2f}")
    print(f"  MA   = {metrics['mean_acc'] * 100:.2f}")
    print(f"  结果已写入 {out_dir} 与 {summary}")


if __name__ == "__main__":
    main()
