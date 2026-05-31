from __future__ import annotations

import argparse

import torch
from torch import nn

from config import build_config
from data_utils import build_dataloader, summarize_batch
from trainer import Trainer
from utils import get_device, set_seed


class SmokeClassifier(nn.Module):
    def __init__(self, num_classes: int = 200) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CUB_200_2011 training entrypoint")
    parser.add_argument("--mode", choices=["smoke", "train"], default="smoke")
    parser.add_argument("--model", default="smoke")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--resize-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--use-bbox-crop", action="store_true")
    parser.add_argument("--no-parts", action="store_true")
    return parser.parse_args()


def build_model(cfg) -> nn.Module:
    if cfg.model.name == "smoke":
        return SmokeClassifier(cfg.model.num_classes)
    raise ValueError(
        f"Model '{cfg.model.name}' is not implemented yet. "
        "Block B will add resnet50_baseline."
    )


def run_smoke(cfg) -> None:
    train_loader = build_dataloader(cfg, "train", shuffle=False)
    val_loader = build_dataloader(cfg, "val", shuffle=False)

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    print("train_dataset_size", len(train_loader.dataset))
    print("val_dataset_size", len(val_loader.dataset))
    print("train_batch", summarize_batch(train_batch))
    print("val_batch", summarize_batch(val_batch))

    model = build_model(cfg)
    device = get_device(cfg.train.device)
    model.to(device)
    images = train_batch["image"].to(device)
    logits = model(images)
    print("smoke_logits_shape", tuple(logits.shape))


def run_train(cfg) -> None:
    train_loader = build_dataloader(cfg, "train")
    val_loader = build_dataloader(cfg, "val", shuffle=False)
    model = build_model(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    trainer = Trainer(
        model=model,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=get_device(cfg.train.device),
    )
    trainer.fit()


def main() -> None:
    args = parse_args()
    cfg = build_config(
        root=args.data_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        resize_size=args.resize_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        use_bbox_crop=args.use_bbox_crop,
        return_parts=not args.no_parts,
        name=args.model,
    )

    set_seed(cfg.train.seed)

    if args.mode == "smoke":
        run_smoke(cfg)
    else:
        run_train(cfg)


if __name__ == "__main__":
    main()
