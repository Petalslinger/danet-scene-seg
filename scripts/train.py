"""训练入口。

示例：
    python scripts/train.py --attention both --epochs 50 --batch-size 8
    python scripts/train.py --attention none --epochs 30 --crop-size 320 --batch-size 4

产出（outputs/<experiment_name>/ 下）：
    train_log.csv      每个 epoch 的损失与验证指标
    metrics.json       最后一次验证的完整指标（含逐类 IoU）
    ckpt_best.pth      验证 mIoU 最高的权重
    ckpt_last.pth      最后一轮权重（含 optimizer / scaler，可 --resume）
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import TrainConfig  # noqa: E402
from src.data import (  # noqa: E402
    EvalTransform,
    TrainTransform,
    VOCSegmentation,
    VOC_CLASSES,
    collate_seg,
)
from src.engine import build_param_groups, evaluate, make_grad_scaler, train_one_epoch  # noqa: E402
from src.losses import SegmentationLoss  # noqa: E402
from src.metrics import format_per_class  # noqa: E402
from src.models import DANet  # noqa: E402
from src.utils import CSVLogger, ensure_dir, save_json, seed_everything  # noqa: E402


def parse_args() -> TrainConfig:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="DANet 场景分割训练")
    parser.add_argument("--data-root", default=defaults.data_root)
    parser.add_argument("--crop-size", type=int, default=defaults.crop_size)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--poly-power", type=float, default=defaults.poly_power)
    parser.add_argument("--aux-weight", type=float, default=defaults.aux_weight)
    parser.add_argument("--grad-clip", type=float, default=defaults.grad_clip)
    parser.add_argument("--attention", choices=["none", "pam", "cam", "both"], default=defaults.attention)
    parser.add_argument("--backbone", choices=["resnet50", "resnet101", "resnet152"], default=defaults.backbone)
    parser.add_argument("--output-stride", type=int, choices=[8, 16], default=defaults.output_stride)
    parser.add_argument("--use-layer4", action="store_true", help="用 layer4（2048 通道，对齐官方 DANet，更吃显存）")
    parser.add_argument("--no-pretrained", action="store_false", dest="pretrained", help="不加载 ImageNet 权重")
    parser.add_argument("--no-amp", action="store_false", dest="amp", help="关闭混合精度")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--val-interval", type=int, default=defaults.val_interval)
    parser.add_argument("--save-interval", type=int, default=defaults.save_interval)
    parser.add_argument("--log-interval", type=int, default=defaults.log_interval)
    parser.add_argument("--resume", default=None, help="从 ckpt_last.pth 继续训练")
    args = parser.parse_args()

    return TrainConfig(
        data_root=args.data_root,
        crop_size=args.crop_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        attention=args.attention,
        backbone=args.backbone,
        pretrained=args.pretrained,
        output_stride=args.output_stride,
        use_layer4=args.use_layer4,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        poly_power=args.poly_power,
        aux_weight=args.aux_weight,
        grad_clip=args.grad_clip,
        amp=args.amp,
        seed=args.seed,
        output_dir=args.output_dir,
        val_interval=args.val_interval,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        resume=args.resume,
    )


def build_loaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader]:
    train_set = VOCSegmentation(cfg.data_root, split="train", transform=TrainTransform(crop_size=cfg.crop_size))
    # 验证集各图尺寸不同（只 pad 到 8 的倍数），必须 batch_size=1
    val_set = VOCSegmentation(
        cfg.data_root, split="val", transform=EvalTransform(pad_to=8, ignore_index=cfg.ignore_index)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_seg,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_seg,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    return train_loader, val_loader


def save_checkpoint(path: Path, model, optimizer, scaler, cfg: TrainConfig, epoch: int, iteration: int, best_miou: float) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "epoch": epoch,
            "iteration": iteration,
            "best_miou": best_miou,
            "config": cfg.to_dict(),
        },
        path,
    )


def main() -> None:
    cfg = parse_args()
    cfg.validate()
    seed_everything(cfg.seed, deterministic=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and cfg.amp:
        print("提示：未检测到 CUDA，已自动关闭 AMP。")
        cfg.amp = False

    out_dir = ensure_dir(Path(cfg.output_dir) / cfg.experiment_name)
    save_json(cfg.to_dict(), out_dir / "config.json")
    print(f"实验：{cfg.experiment_name}  device={device}  输出：{out_dir}")

    train_loader, val_loader = build_loaders(cfg)
    print(f"train: {len(train_loader.dataset)} 张 / {len(train_loader)} iters    val: {len(val_loader.dataset)} 张")

    model = DANet(cfg.model_config()).to(device)
    criterion = SegmentationLoss(ignore_index=cfg.ignore_index, aux_weight=cfg.aux_weight)
    optimizer = torch.optim.SGD(
        build_param_groups(model, cfg.weight_decay),
        lr=cfg.lr,
        momentum=cfg.momentum,
        nesterov=True,
    )
    amp_enabled = bool(cfg.amp and device.type == "cuda")
    scaler = make_grad_scaler(amp_enabled)

    total_iters = cfg.epochs * len(train_loader)
    start_epoch, iteration, best_miou = 0, 0, -1.0
    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") is not None and scaler.is_enabled():
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        iteration = int(ckpt.get("iteration", 0))
        best_miou = float(ckpt.get("best_miou", -1.0))
        print(f"从 {cfg.resume} 恢复：epoch={start_epoch} it={iteration} best_miou={best_miou:.4f}")

    logger = CSVLogger(out_dir / "train_log.csv")
    start_time = time.perf_counter()
    last_metrics = None

    for epoch in range(start_epoch, cfg.epochs):
        stats, iteration = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            cfg=cfg,
            epoch=epoch + 1,
            start_iter=iteration,
            total_iters=total_iters,
        )
        print(
            f"epoch {epoch + 1}/{cfg.epochs}  loss={stats['loss']:.4f}  "
            f"{stats['iters_per_sec']:.2f} it/s  {stats['seconds']:.0f}s"
        )

        if (epoch + 1) % cfg.val_interval == 0 or (epoch + 1) == cfg.epochs:
            metrics = evaluate(
                model, val_loader, cfg.num_classes, cfg.ignore_index, device, amp=amp_enabled
            )
            last_metrics = metrics
            print(
                f"  val  mIoU={metrics['miou'] * 100:.2f}  "
                f"PA={metrics['pixel_acc'] * 100:.2f}  MA={metrics['mean_acc'] * 100:.2f}"
            )
            logger.log(
                {
                    "epoch": epoch + 1,
                    "iteration": iteration,
                    "lr": optimizer.param_groups[0]["lr"],
                    "train_loss": round(stats["loss"], 6),
                    "miou": round(metrics["miou"], 6),
                    "pixel_acc": round(metrics["pixel_acc"], 6),
                    "mean_acc": round(metrics["mean_acc"], 6),
                    "seconds": round(stats["seconds"], 1),
                }
            )
            save_json(metrics, out_dir / "metrics.json")

            if metrics["miou"] > best_miou:
                best_miou = metrics["miou"]
                save_checkpoint(out_dir / "ckpt_best.pth", model, optimizer, scaler, cfg, epoch + 1, iteration, best_miou)
                print(f"  * 新的 best mIoU={best_miou * 100:.2f}，已保存 ckpt_best.pth")
                save_json({"per_class_iou": metrics["per_class_iou"]}, out_dir / "per_class_iou.json")

        if (epoch + 1) % cfg.save_interval == 0 or (epoch + 1) == cfg.epochs:
            save_checkpoint(out_dir / "ckpt_last.pth", model, optimizer, scaler, cfg, epoch + 1, iteration, best_miou)

    elapsed = (time.perf_counter() - start_time) / 3600
    print(f"训练结束，用时 {elapsed:.2f} h，best mIoU = {best_miou * 100:.2f}")
    if last_metrics is not None:
        print("逐类 IoU：")
        print(format_per_class(last_metrics["per_class_iou"], list(VOC_CLASSES)))


if __name__ == "__main__":
    main()
