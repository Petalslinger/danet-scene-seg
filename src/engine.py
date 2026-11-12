"""训练/验证的一个 epoch。

只做「一个 epoch 干什么」，不碰命令行和文件组织，方便被 train.py 与 evaluate.py 复用。
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Tuple

import torch
import torch.nn as nn

from .metrics import SegmentationMetrics
from .progress import make_bar
from .utils import AverageMeter, poly_lr

__all__ = ["train_one_epoch", "evaluate", "build_param_groups", "make_grad_scaler"]


def make_grad_scaler(enabled: bool, device_type: str = "cuda") -> torch.amp.GradScaler:
    """AMP 关掉时返回一个 no-op 的 scaler，调用方不用写分支。"""
    return torch.amp.GradScaler(device_type, enabled=enabled)


def build_param_groups(model: nn.Module, weight_decay: float) -> list:
    """权重衰减只作用于卷积/线性层的权重，BN 参数与 bias 不加。"""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: Callable,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    cfg: Any,
    epoch: int,
    start_iter: int,
    total_iters: int,
    log_fn: Callable[[str], None] = print,
    progress: bool = True,
) -> Tuple[Dict[str, float], int]:
    """返回 (平均损失字典, 结束时的全局迭代数)。"""
    model.train()
    amp_enabled = bool(cfg.amp and device.type == "cuda")

    meters: Dict[str, AverageMeter] = {"loss": AverageMeter()}
    iteration = start_iter
    epoch_start = time.perf_counter()

    bar = make_bar(loader, total=len(loader), desc=f"epoch {epoch}/{cfg.epochs}", enabled=progress)
    using_bar = bar is not loader
    emit = bar.write if using_bar else log_fn

    for batch in bar:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True)

        lr = poly_lr(cfg.lr, iteration, total_iters, cfg.poly_power)
        for group in optimizer.param_groups:
            group["lr"] = lr

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)
            loss, parts = criterion(outputs, targets)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        meters["loss"].update(loss.item(), n=images.size(0))
        for name, value in parts.items():
            meters.setdefault(name, AverageMeter()).update(value.item(), n=images.size(0))

        if using_bar:
            bar.set_postfix(loss=f"{meters['loss'].avg:.4f}", lr=f"{lr:.1e}")

        if cfg.log_interval > 0 and (iteration - start_iter) % cfg.log_interval == 0:
            detail = " ".join(f"{k}={m.avg:.4f}" for k, m in meters.items())
            emit(f"  epoch {epoch} it {iteration} lr {lr:.2e} {detail}")

        iteration += 1

    if using_bar:
        bar.close()

    elapsed = time.perf_counter() - epoch_start
    stats = {k: v.avg for k, v in meters.items()}
    stats["seconds"] = elapsed
    stats["iters_per_sec"] = len(loader) / elapsed if elapsed > 0 else 0.0
    return stats, iteration


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    num_classes: int,
    ignore_index: int,
    device: torch.device,
    amp: bool = True,
    desc: str = "val",
    progress: bool = True,
) -> Dict[str, Any]:
    """跑完一遍验证集，返回 mIoU / PA / MA / 逐类 IoU。"""
    model.eval()
    metrics = SegmentationMetrics(num_classes=num_classes, ignore_index=ignore_index)
    amp_enabled = bool(amp and device.type == "cuda")

    bar = make_bar(loader, total=len(loader), desc=desc, enabled=progress)
    using_bar = bar is not loader

    for batch in bar:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images).logits
        metrics.update(logits.argmax(dim=1), targets)

    if using_bar:
        bar.close()

    return metrics.compute()
