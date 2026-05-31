from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast

from utils import AverageMeter, accuracy, ensure_dir, save_checkpoint


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        cfg,
        train_loader=None,
        val_loader=None,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[object] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = GradScaler(enabled=cfg.train.amp and self.device.type == "cuda")
        self.best_top1 = 0.0

        self.model.to(self.device)
        ensure_dir(cfg.train.output_dir)
        ensure_dir(cfg.train.ckpt_dir)

    def fit(self) -> Dict[str, float]:
        if self.train_loader is None or self.val_loader is None:
            raise ValueError("train_loader and val_loader are required for fit().")
        if self.optimizer is None:
            raise ValueError("optimizer is required for fit().")

        last_metrics: Dict[str, float] = {}
        for epoch in range(self.cfg.train.epochs):
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.evaluate(epoch)
            last_metrics = {**train_metrics, **val_metrics}

            if self.scheduler is not None:
                self.scheduler.step()

            if val_metrics["val_top1"] >= self.best_top1:
                self.best_top1 = val_metrics["val_top1"]
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "best_top1": self.best_top1,
                        "config": str(self.cfg),
                    },
                    Path(self.cfg.train.ckpt_dir) / f"{self.cfg.model.name}_best.pt",
                )

        return last_metrics

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        loss_meter = AverageMeter()
        top1_meter = AverageMeter()

        for step, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.scaler.is_enabled()):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_size = labels.size(0)
            acc = accuracy(logits.detach(), labels, topk=(1,))
            loss_meter.update(float(loss.item()), batch_size)
            top1_meter.update(acc["top1"], batch_size)

            if step % self.cfg.train.log_interval == 0:
                print(
                    f"epoch={epoch + 1} step={step} "
                    f"loss={loss_meter.avg:.4f} top1={top1_meter.avg:.2f}"
                )

        return {"train_loss": loss_meter.avg, "train_top1": top1_meter.avg}

    @torch.no_grad()
    def evaluate(self, epoch: int = 0) -> Dict[str, float]:
        if self.val_loader is None:
            raise ValueError("val_loader is required for evaluate().")

        self.model.eval()
        loss_meter = AverageMeter()
        top1_meter = AverageMeter()
        top5_meter = AverageMeter()

        for batch in self.val_loader:
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            logits = self.model(images)
            loss = self.criterion(logits, labels)
            acc = accuracy(logits, labels, topk=(1, 5))

            batch_size = labels.size(0)
            loss_meter.update(float(loss.item()), batch_size)
            top1_meter.update(acc["top1"], batch_size)
            top5_meter.update(acc["top5"], batch_size)

        print(
            f"eval epoch={epoch + 1} "
            f"loss={loss_meter.avg:.4f} top1={top1_meter.avg:.2f} "
            f"top5={top5_meter.avg:.2f}"
        )
        return {
            "val_loss": loss_meter.avg,
            "val_top1": top1_meter.avg,
            "val_top5": top5_meter.avg,
        }
