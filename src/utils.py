"""零散工具：随机种子、poly 学习率、均值计、CSV 日志。"""

from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

__all__ = [
    "seed_everything",
    "poly_lr",
    "AverageMeter",
    "CSVLogger",
    "ensure_dir",
    "save_json",
    "load_json",
]


def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def poly_lr(base_lr: float, iteration: int, total_iterations: int, power: float = 0.9) -> float:
    """poly 衰减：lr = base * (1 - it/total)^power。

    注意 iteration 与 total 都是「迭代数」而不是 epoch 数，策略按 iteration 更新。
    """
    if total_iterations <= 0:
        return base_lr
    progress = min(max(iteration / total_iterations, 0.0), 1.0)
    return base_lr * (1.0 - progress) ** power


class AverageMeter:
    """滑动平均计。"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


class CSVLogger:
    """追加写 CSV，文件不存在时写表头；字段顺序由第一次写入决定。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        ensure_dir(self.path.parent)
        self.fieldnames: Optional[list] = None

    def log(self, row: Dict[str, Any]) -> None:
        if self.fieldnames is None:
            self.fieldnames = list(row.keys())
            write_header = not self.path.exists() or self.path.stat().st_size == 0
        else:
            write_header = False

        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
