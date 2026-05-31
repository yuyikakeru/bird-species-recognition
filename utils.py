from __future__ import annotations

import json
import random
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch


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
    logits: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1, 5)
) -> Dict[str, float]:
    max_k = min(max(topk), logits.size(1))
    _, pred = logits.topk(max_k, dim=1)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    results: Dict[str, float] = {}
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


def save_json(data: Dict[str, object], path: Path | str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def save_csv(rows: List[Dict[str, object]], path: Path | str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: Iterable[float]) -> Tuple[float, float]:
    values = list(values)
    if not values:
        return 0.0, 0.0
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return mean, std


def save_checkpoint(
    state: Dict[str, object],
    path: Path | str,
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(state, path)
