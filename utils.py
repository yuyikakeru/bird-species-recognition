from __future__ import annotations

import csv
import json
import os
import random
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


class RunLock:
    def __init__(self, lock_path: Path | str) -> None:
        self.lock_path = Path(lock_path)
        self.fd: int | None = None

    def __enter__(self) -> "RunLock":
        ensure_dir(self.lock_path.parent)
        try:
            self.fd = os.open(
                self.lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError as exc:
            lock_text = self.lock_path.read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(
                f"Run lock already exists: {self.lock_path}\n"
                "Another process may be writing this run. Use a different "
                "--run-name or remove this single lock after confirming its "
                "recorded process is no longer running.\n"
                f"Lock content: {lock_text}"
            ) from exc

        payload = {
            "pid": os.getpid(),
            "cwd": str(Path.cwd()),
            "lock_path": str(self.lock_path),
        }
        os.write(
            self.fd,
            json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
        )
        os.close(self.fd)
        self.fd = None
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        if self.lock_path.exists():
            self.lock_path.unlink()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def accuracy(
    logits: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...] = (1, 5)
) -> dict[str, float]:
    max_k = min(max(topk), logits.size(1))
    _, pred = logits.topk(max_k, dim=1)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    results: dict[str, float] = {}
    batch_size = target.size(0)
    for k in topk:
        k = min(k, logits.size(1))
        correct_k = correct[:k].reshape(-1).float().sum(0)
        results[f"top{k}"] = float(correct_k.mul_(100.0 / batch_size).item())
    return results


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def ensure_dir(path: Path | str) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def save_json(data: dict[str, object], path: Path | str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def save_csv(rows: list[dict[str, object]], path: Path | str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    try:
        handle = path.open("w", encoding="utf-8", newline="")
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_path = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
        print(
            f"warning: cannot write CSV because it is locked or permission denied: {path}. "
            f"Writing to fallback file: {fallback_path}"
        )
        handle = fallback_path.open("w", encoding="utf-8", newline="")

    with handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: Iterable[float]) -> tuple[float, float]:
    values = list(values)
    if not values:
        return 0.0, 0.0
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return mean, std


def save_checkpoint(
    state: dict[str, object],
    path: Path | str,
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(state, path)
