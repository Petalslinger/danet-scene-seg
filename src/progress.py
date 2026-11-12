"""终端进度条。

优先用 tqdm（自适应宽度、自动限流刷新）；环境里没装 tqdm 时退化为纯 Python 的
``\\r`` 刷新条，所以进度条本身不引入硬依赖，缺包也能正常训练。
"""

from __future__ import annotations

import sys
import time
from typing import Any, Iterable, Iterator, Optional, TextIO

try:  # tqdm 是可选依赖
    from tqdm import tqdm as _tqdm
except Exception:  # pragma: no cover - 环境里没装 tqdm
    _tqdm = None

__all__ = ["TQDM_AVAILABLE", "SimpleBar", "make_bar"]

TQDM_AVAILABLE = _tqdm is not None


class SimpleBar:
    """不依赖第三方库的进度条，只实现训练/验证用到的接口。"""

    def __init__(
        self,
        iterable: Iterable[Any],
        total: Optional[int] = None,
        desc: str = "",
        width: int = 24,
        stream: Optional[TextIO] = None,
        min_interval: float = 0.1,
    ) -> None:
        self._iterable = iterable
        self.total = int(total) if total is not None else 0
        self.desc = desc
        self.width = width
        self.stream = stream or sys.stderr
        self.min_interval = min_interval
        self.n = 0
        self.postfix: dict = {}
        self._start = time.perf_counter()
        self._last = 0.0
        self._line_len = 0

    def __iter__(self) -> Iterator[Any]:
        self._start = time.perf_counter()
        for item in self._iterable:
            yield item
            self.n += 1
            now = time.perf_counter()
            if now - self._last >= self.min_interval or (self.total and self.n >= self.total):
                self._last = now
                self._draw()

    def set_postfix(self, refresh: bool = True, **kwargs: Any) -> None:
        self.postfix.update(kwargs)
        if refresh:
            self._draw()

    def write(self, msg: str) -> None:
        """在进度条上方输出一行，不打断后续刷新。"""
        self._clear()
        self.stream.write(msg + "\n")
        self._draw()

    def close(self) -> None:
        self._draw()
        self.stream.write("\n")
        self.stream.flush()

    # ------------------------------------------------------------------ #
    def _clear(self) -> None:
        if self._line_len:
            self.stream.write("\r" + " " * self._line_len + "\r")
            self._line_len = 0

    def _draw(self) -> None:
        frac = min(max(self.n / self.total, 0.0), 1.0) if self.total else 0.0
        filled = int(round(self.width * frac))
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.perf_counter() - self._start
        rate = self.n / elapsed if elapsed > 0 else 0.0
        eta = (self.total - self.n) / rate if rate > 0 and self.total else 0.0
        postfix = "  " + " ".join(f"{k}={v}" for k, v in self.postfix.items()) if self.postfix else ""
        line = (
            f"\r{self.desc} [{bar}] {self.n}/{self.total} {frac * 100:5.1f}% "
            f"{rate:5.2f} it/s ETA {eta:4.0f}s{postfix}"
        )
        pad = max(self._line_len - len(line), 0)
        self.stream.write(line + " " * pad)
        self.stream.flush()
        self._line_len = len(line)


def make_bar(
    iterable: Iterable[Any],
    *,
    total: Optional[int] = None,
    desc: str = "",
    enabled: bool = True,
    leave: bool = False,
):
    """返回包在 iterable 外面的进度条；enabled=False 时原样返回 iterable。

    装了 tqdm 就用 tqdm，否则用 SimpleBar；两者都支持 for 循环、
    ``set_postfix(...)``、``write(...)``、``close()``。
    """
    if not enabled:
        return iterable
    if _tqdm is not None:
        return _tqdm(iterable, total=total, desc=desc, leave=leave, dynamic_ncols=True, file=sys.stderr)
    return SimpleBar(iterable, total=total, desc=desc)
