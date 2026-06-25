from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from utils import AverageMeter, accuracy, save_checkpoint, save_csv, save_json


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        cfg,
        train_loader=None,
        val_loader=None,
        criterion: nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: object | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp_enabled = cfg.train.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.best_top1 = -float("inf")
        self.best_epoch = 0
        self.optimizer_steps = 0

        self.model.to(self.device)

    def _forward_batch(self, images: torch.Tensor, batch: dict) -> torch.Tensor:
        return self.model(images)

    def fit(self) -> dict[str, float]:
        if self.train_loader is None or self.val_loader is None:
            raise ValueError("train_loader and val_loader are required for fit().")
        if self.optimizer is None:
            raise ValueError("optimizer is required for fit().")

        last_metrics: dict[str, float] = {}
        history = []
        for epoch in range(self.cfg.train.epochs):
            optimizer_steps_before = self.optimizer_steps
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.evaluate(epoch)
            val_top1 = val_metrics["val_top1"]
            is_best = val_top1 > self.best_top1

            if is_best:
                self.best_top1 = val_top1
                self.best_epoch = epoch + 1
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "model": self.model.state_dict(),
                        "best_top1": self.best_top1,
                        "config": str(self.cfg),
                    },
                    Path(self.cfg.train.ckpt_dir) / f"{self.cfg.model.name}_best.pt",
                )

            last_metrics = {
                **train_metrics,
                **val_metrics,
                "best_top1": self.best_top1,
                "best_epoch": self.best_epoch,
            }
            history.append({"epoch": epoch + 1, **last_metrics})

            if (
                self.scheduler is not None
                and self.optimizer_steps > optimizer_steps_before
            ):
                self.scheduler.step()

        if self.best_top1 == -float("inf"):
            self.best_top1 = 0.0

        save_json(
            {
                "model": self.cfg.model.name,
                "best_top1": self.best_top1,
                "best_epoch": self.best_epoch,
                "epochs_ran": len(history),
                "history": history,
            },
            Path(self.cfg.train.output_dir) / f"{self.cfg.model.name}_history.json",
        )
        save_csv(history, Path(self.cfg.train.output_dir) / f"{self.cfg.model.name}_history.csv")
        return {
            **last_metrics,
            "best_top1": self.best_top1,
            "best_epoch": self.best_epoch,
        }

    def fit_fixed(self) -> dict[str, float]:
        if self.train_loader is None:
            raise ValueError("train_loader is required for fit_fixed().")
        if self.optimizer is None:
            raise ValueError("optimizer is required for fit_fixed().")

        history = []
        last_metrics: dict[str, float] = {}
        for epoch in range(self.cfg.train.epochs):
            optimizer_steps_before = self.optimizer_steps
            last_metrics = self.train_one_epoch(epoch)
            history.append({"epoch": epoch + 1, **last_metrics})
            if (
                self.scheduler is not None
                and self.optimizer_steps > optimizer_steps_before
            ):
                self.scheduler.step()

        checkpoint_path = (
            Path(self.cfg.train.ckpt_dir) / f"{self.cfg.model.name}_final.pt"
        )
        save_checkpoint(
            {
                "epoch": self.cfg.train.epochs,
                "model": self.model.state_dict(),
                "config": str(self.cfg),
            },
            checkpoint_path,
        )
        result = {
            "model": self.cfg.model.name,
            "epochs_trained": self.cfg.train.epochs,
            "checkpoint": str(checkpoint_path),
            "history": history,
        }
        save_json(
            result,
            Path(self.cfg.train.output_dir)
            / f"{self.cfg.model.name}_full_history.json",
        )
        save_csv(
            history,
            Path(self.cfg.train.output_dir)
            / f"{self.cfg.model.name}_full_history.csv",
        )
        return {
            **last_metrics,
            "epochs_trained": self.cfg.train.epochs,
            "checkpoint": str(checkpoint_path),
        }

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        loss_meter = AverageMeter()
        top1_meter = AverageMeter()

        for step, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(self.device.type, enabled=self.amp_enabled):
                logits = self._forward_batch(images, batch)
                loss = self.criterion(logits, labels)

            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite train output: model={self.cfg.model.name} "
                    f"epoch={epoch + 1} step={step + 1} "
                    f"logits_finite={bool(torch.isfinite(logits).all())} "
                    f"loss={float(loss.detach())}"
                )

            if self.amp_enabled:
                scale_before = self.scaler.get_scale()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.cfg.train.grad_clip_norm,
                    error_if_nonfinite=False,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scaler.get_scale() >= scale_before:
                    self.optimizer_steps += 1
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.cfg.train.grad_clip_norm,
                    error_if_nonfinite=True,
                )
                self.optimizer.step()
                self.optimizer_steps += 1

            batch_size = labels.size(0)
            acc = accuracy(logits.detach(), labels, topk=(1,))
            loss_meter.update(float(loss.item()), batch_size)
            top1_meter.update(acc["top1"], batch_size)

            if step % self.cfg.train.log_interval == 0:
                print(
                    f"train epoch={epoch + 1} step={step + 1}/{len(self.train_loader)} "
                    f"loss={loss_meter.avg:.4f} top1={top1_meter.avg:.2f}"
                )

        return {"train_loss": loss_meter.avg, "train_top1": top1_meter.avg}

    @torch.no_grad()
    def evaluate(self, epoch: int = 0) -> dict[str, float]:
        if self.val_loader is None:
            raise ValueError("val_loader is required for evaluate().")

        return self.evaluate_loader(
            self.val_loader,
            split="val",
            epoch=epoch,
        )

    @torch.no_grad()
    def evaluate_loader(
        self,
        loader,
        split: str,
        epoch: int = 0,
    ) -> dict[str, float]:

        self.model.eval()
        loss_meter = AverageMeter()
        top1_meter = AverageMeter()
        top5_meter = AverageMeter()

        for step, batch in enumerate(loader):
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            with torch.amp.autocast(self.device.type, enabled=self.amp_enabled):
                logits = self._forward_batch(images, batch)
                loss = self.criterion(logits, labels)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite {split} output: model={self.cfg.model.name} "
                    f"epoch={epoch + 1} step={step + 1} "
                    f"logits_finite={bool(torch.isfinite(logits).all())} "
                    f"loss={float(loss.detach())}"
                )
            acc = accuracy(logits, labels, topk=(1, 5))

            batch_size = labels.size(0)
            loss_meter.update(float(loss.item()), batch_size)
            top1_meter.update(acc["top1"], batch_size)
            top5_meter.update(acc["top5"], batch_size)

            if step % self.cfg.train.log_interval == 0:
                epoch_text = "" if split == "test" else f" epoch={epoch + 1}"
                print(
                    f"{split}{epoch_text} step={step + 1}/{len(loader)} "
                    f"loss={loss_meter.avg:.4f} top1={top1_meter.avg:.2f} "
                    f"top5={top5_meter.avg:.2f}"
                )

        epoch_text = "" if split == "test" else f" epoch={epoch + 1}"
        print(
            f"{split}{epoch_text} "
            f"loss={loss_meter.avg:.4f} top1={top1_meter.avg:.2f} "
            f"top5={top5_meter.avg:.2f}"
        )
        return {
            f"{split}_loss": loss_meter.avg,
            f"{split}_top1": top1_meter.avg,
            f"{split}_top5": top5_meter.avg,
        }
