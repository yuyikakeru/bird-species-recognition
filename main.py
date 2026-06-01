from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
from torch import nn

from config import build_config
from data_utils import build_dataloader, summarize_batch
from model import build_resnet50_baseline
from trainer import Trainer
from utils import get_device, mean_std, save_csv, save_json, set_seed


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
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seeds for repeated runs, e.g. 42,2024,3407.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Name for this repeated experiment under log/<model>/ and ckpt/<model>/.",
    )
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--scheduler", choices=["cosine", "none"], default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--early-stop-min-delta", type=float, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--use-bbox-crop", action="store_true")
    parser.add_argument("--no-parts", action="store_true")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.set_defaults(pretrained=None)
    return parser.parse_args()


def parse_seed_list(seed: int, seeds: str | None) -> list[int]:
    if not seeds:
        return [seed]
    parsed = [int(item.strip()) for item in seeds.split(",") if item.strip()]
    if not parsed:
        raise ValueError("--seeds was provided but no valid integer seed was found.")
    return parsed


def build_model(cfg) -> nn.Module:
    if cfg.model.name == "smoke":
        return SmokeClassifier(cfg.model.num_classes)
    if cfg.model.name in {"resnet50", "resnet50_baseline"}:
        return build_resnet50_baseline(
            num_classes=cfg.model.num_classes,
            pretrained=cfg.model.pretrained,
        )
    raise ValueError(
        f"Model '{cfg.model.name}' is not implemented yet. "
        "Available models: smoke, resnet50_baseline."
    )


def build_optimizer(cfg, model: nn.Module) -> torch.optim.Optimizer:
    if cfg.train.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.train.lr,
            momentum=cfg.train.momentum,
            weight_decay=cfg.train.weight_decay,
        )
    if cfg.train.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {cfg.train.optimizer}")


def build_scheduler(cfg, optimizer: torch.optim.Optimizer):
    if cfg.train.scheduler == "none":
        return None
    if cfg.train.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(cfg.train.epochs, 1),
        )
    raise ValueError(f"Unsupported scheduler: {cfg.train.scheduler}")


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
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)
    trainer = Trainer(
        model=model,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=get_device(cfg.train.device),
    )
    return trainer.fit()


def run_repeated_train(cfg, seeds: list[int], run_name: str | None = None) -> None:
    experiment_name = run_name or "seeds_" + "_".join(str(seed) for seed in seeds)
    base_output_dir = Path(cfg.train.output_dir) / cfg.model.name / experiment_name
    base_ckpt_dir = Path(cfg.train.ckpt_dir) / cfg.model.name / experiment_name
    rows = []

    for run_index, seed in enumerate(seeds, start=1):
        run_cfg = copy.deepcopy(cfg)
        run_cfg.train.seed = seed
        run_cfg.train.output_dir = base_output_dir / f"seed_{seed}"
        run_cfg.train.ckpt_dir = base_ckpt_dir / f"seed_{seed}"

        print(f"run={run_index}/{len(seeds)} seed={seed}")
        set_seed(seed)
        metrics = run_train(run_cfg)
        row = {"run": run_index, "seed": seed, **metrics}
        rows.append(row)

    if len(rows) == 1:
        print("final_metrics", rows[0])
        return

    top1_mean, top1_std = mean_std(row["best_top1"] for row in rows)
    val_top1_mean, val_top1_std = mean_std(row["val_top1"] for row in rows)
    val_top5_mean, val_top5_std = mean_std(row["val_top5"] for row in rows)
    summary = {
        "model": cfg.model.name,
        "seeds": seeds,
        "runs": rows,
        "best_top1_mean": top1_mean,
        "best_top1_std": top1_std,
        "last_val_top1_mean": val_top1_mean,
        "last_val_top1_std": val_top1_std,
        "last_val_top5_mean": val_top5_mean,
        "last_val_top5_std": val_top5_std,
    }

    save_json(summary, base_output_dir / "repeat_summary.json")
    save_csv(rows, base_output_dir / "repeat_summary.csv")
    print("repeat_summary", summary)


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
        seed=args.seed,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        momentum=args.momentum,
        label_smoothing=args.label_smoothing,
        scheduler=args.scheduler,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        device=args.device,
        use_bbox_crop=args.use_bbox_crop,
        return_parts=not args.no_parts,
        pretrained=args.pretrained,
        name=args.model,
    )

    seeds = parse_seed_list(cfg.train.seed, args.seeds)
    if args.mode == "smoke":
        set_seed(seeds[0])
        cfg.train.seed = seeds[0]

    if args.mode == "smoke":
        run_smoke(cfg)
    else:
        run_repeated_train(cfg, seeds, args.run_name)


if __name__ == "__main__":
    main()
