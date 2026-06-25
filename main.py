from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

import torch
from torch import nn

from config import SUPPORTED_MODELS, build_config, validate_config
from data_utils import build_dataloader
from model import (
    build_resnet50_baseline,
    build_swinv2_tiny_baseline,
    build_swinv2_tiny_fpn,
    build_swinv2_tiny_fpn_parts,
    build_swinv2_tiny_fpn_relation,
)
from trainer import Trainer
from utils import RunLock, get_device, mean_std, save_csv, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CUB_200_2011 training entrypoint")
    parser.add_argument(
        "--mode",
        choices=["train", "summarize", "pipeline", "test"],
        required=True,
    )
    parser.add_argument("--model", choices=SUPPORTED_MODELS, required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--resize-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--head-lr-mult",
        type=float,
        default=None,
        help="Learning-rate multiplier for newly added model heads when supported.",
    )
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
    parser.add_argument(
        "--search-space",
        default=None,
        help="JSON file containing training hyperparameter candidates.",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=None,
        help="Final checkpoints to evaluate in test mode.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--ckpt-dir", default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--scheduler", choices=["cosine", "none"], default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--use-bbox-crop", action="store_true")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--fpn-channels", type=int, default=None)
    parser.add_argument("--num-parts", type=int, default=None)
    parser.add_argument("--part-window-size", type=int, default=None)
    parser.add_argument("--relation-heads", type=int, default=None)
    parser.add_argument("--bilinear-dim", type=int, default=None)
    parser.set_defaults(pretrained=None)
    return parser.parse_args()


def parse_seed_list(seed: int, seeds: str | None) -> list[int]:
    if not seeds:
        return [seed]
    parsed = [int(item.strip()) for item in seeds.split(",") if item.strip()]
    if not parsed:
        raise ValueError("--seeds was provided but no valid integer seed was found.")
    return parsed


def load_search_candidates(path: str | None) -> list[dict]:
    if path is None:
        return [{"id": "current_config"}]

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    candidates = payload.get("candidates") if isinstance(payload, dict) else payload
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Search space must contain a non-empty candidates list.")
    if not all(isinstance(candidate, dict) for candidate in candidates):
        raise ValueError("Every search candidate must be a JSON object.")
    return candidates


def apply_candidate_overrides(cfg, candidate: dict):
    candidate_cfg = copy.deepcopy(cfg)
    allowed_keys = {
        "batch_size",
        "lr",
        "head_lr_mult",
        "weight_decay",
        "momentum",
        "label_smoothing",
        "scheduler",
        "grad_clip_norm",
        "fpn_channels",
        "num_parts",
        "part_window_size",
        "relation_heads",
        "bilinear_dim",
    }
    for key, value in candidate.items():
        if key == "id":
            continue
        if key not in allowed_keys:
            raise ValueError(
                f"Unsupported search-space option: '{key}'. "
                f"Supported options: {', '.join(sorted(allowed_keys))}."
            )
        if hasattr(candidate_cfg.data, key):
            setattr(candidate_cfg.data, key, value)
        elif hasattr(candidate_cfg.train, key):
            setattr(candidate_cfg.train, key, value)
        else:
            raise KeyError(f"Unknown search-space option: {key}")
    return validate_config(candidate_cfg)


def candidate_slug(candidate: dict, index: int) -> str:
    raw_name = str(candidate.get("id", f"candidate_{index}"))
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_name).strip("._")
    return f"{index:02d}_{safe_name or 'candidate'}"


def build_model(cfg) -> nn.Module:
    builders = {
        "resnet50_baseline": lambda: build_resnet50_baseline(
            num_classes=cfg.model.num_classes,
            pretrained=cfg.model.pretrained,
        ),
        "swinv2_tiny": lambda: build_swinv2_tiny_baseline(
            num_classes=cfg.model.num_classes,
            pretrained=cfg.model.pretrained,
            image_size=cfg.data.image_size,
        ),
        "swinv2_tiny_fpn": lambda: build_swinv2_tiny_fpn(
            num_classes=cfg.model.num_classes,
            pretrained=cfg.model.pretrained,
            image_size=cfg.data.image_size,
            fpn_channels=cfg.model.fpn_channels,
        ),
        "swinv2_tiny_fpn_parts": lambda: build_swinv2_tiny_fpn_parts(
            num_classes=cfg.model.num_classes,
            pretrained=cfg.model.pretrained,
            image_size=cfg.data.image_size,
            fpn_channels=cfg.model.fpn_channels,
            num_parts=cfg.model.num_parts,
            part_window_size=cfg.model.part_window_size,
        ),
        "swinv2_tiny_fpn_relation": lambda: build_swinv2_tiny_fpn_relation(
            num_classes=cfg.model.num_classes,
            pretrained=cfg.model.pretrained,
            image_size=cfg.data.image_size,
            fpn_channels=cfg.model.fpn_channels,
            num_parts=cfg.model.num_parts,
            part_window_size=cfg.model.part_window_size,
            relation_heads=cfg.model.relation_heads,
            bilinear_dim=cfg.model.bilinear_dim,
        ),
    }
    return builders[cfg.model.name]()


def build_optimizer(cfg, model: nn.Module) -> torch.optim.Optimizer:
    if hasattr(model, "get_optimizer_param_groups"):
        trainable_params = model.get_optimizer_param_groups(
            base_lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
            head_lr_mult=cfg.train.head_lr_mult,
        )
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]

    if not trainable_params:
        raise ValueError("No trainable parameters found for optimizer.")
    if cfg.train.optimizer == "sgd":
        return torch.optim.SGD(
            trainable_params,
            lr=cfg.train.lr,
            momentum=cfg.train.momentum,
            weight_decay=cfg.train.weight_decay,
        )
    if cfg.train.optimizer == "adamw":
        return torch.optim.AdamW(
            trainable_params,
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


def build_training_trainer(cfg, train_split: str, with_validation: bool) -> Trainer:
    train_loader = build_dataloader(cfg, train_split)
    val_loader = (
        build_dataloader(cfg, "val", shuffle=False) if with_validation else None
    )
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
    return trainer


def run_train(cfg) -> dict:
    return build_training_trainer(cfg, "train", with_validation=True).fit()


def run_full_train(cfg) -> dict:
    return build_training_trainer(cfg, "train_full", with_validation=False).fit_fixed()


def run_test_checkpoint(
    cfg,
    checkpoint_path: Path,
    test_loader=None,
    save_metrics: bool = True,
) -> dict:
    if test_loader is None:
        test_loader = build_dataloader(cfg, "test", shuffle=False)
    device = get_device(cfg.train.device)
    eval_cfg = copy.deepcopy(cfg)
    eval_cfg.model.pretrained = False
    model = build_model(eval_cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Checkpoint does not contain model weights: {checkpoint_path}")
    model.load_state_dict(checkpoint["model"])
    trainer = Trainer(
        model=model,
        cfg=eval_cfg,
        val_loader=test_loader,
        criterion=nn.CrossEntropyLoss(),
        device=device,
    )
    metrics = trainer.evaluate_loader(
        test_loader,
        split="test",
        epoch=max(cfg.train.epochs - 1, 0),
    )
    if save_metrics:
        save_json(metrics, Path(cfg.train.output_dir) / "test_metrics.json")
    return metrics


def summarize_test_rows(model_name: str, rows: list[dict]) -> dict:
    top1_mean, top1_std = mean_std(row["test_top1"] for row in rows)
    top5_mean, top5_std = mean_std(row["test_top5"] for row in rows)
    return {
        "model": model_name,
        "test_top1_mean": top1_mean,
        "test_top1_std": top1_std,
        "test_top5_mean": top5_mean,
        "test_top5_std": top5_std,
        "runs": rows,
    }


def run_test_checkpoints(cfg, checkpoint_paths: list[str]) -> None:
    test_loader = build_dataloader(cfg, "test", shuffle=False)
    rows = []
    for checkpoint_value in checkpoint_paths:
        checkpoint_path = Path(checkpoint_value)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
        print(f"official_test checkpoint={checkpoint_path}")
        metrics = run_test_checkpoint(
            cfg,
            checkpoint_path,
            test_loader=test_loader,
            save_metrics=False,
        )
        rows.append(
            {
                "seed": checkpoint_path.parent.name,
                "checkpoint": str(checkpoint_path),
                **metrics,
            }
        )

    summary = summarize_test_rows(cfg.model.name, rows)
    output_dir = Path(cfg.train.output_dir)
    save_json(summary, output_dir / f"{cfg.model.name}_official_test_summary.json")
    save_csv(rows, output_dir / f"{cfg.model.name}_official_test_summary.csv")
    print("official_test_summary", summary)


def run_selection_pipeline(
    cfg,
    seeds: list[int],
    run_name: str | None,
    search_space: str | None,
) -> None:
    candidates = load_search_candidates(search_space)
    experiment_name = run_name or "selection_" + "_".join(map(str, seeds))
    base_output_dir = Path(cfg.train.output_dir) / "pipeline" / experiment_name
    base_ckpt_dir = Path(cfg.train.ckpt_dir) / "pipeline" / experiment_name
    search_rows = []
    candidate_summaries = []

    with RunLock(base_output_dir / ".run.lock"):
        for candidate_index, candidate in enumerate(candidates, start=1):
            slug = candidate_slug(candidate, candidate_index)
            candidate_cfg = apply_candidate_overrides(cfg, candidate)
            candidate_rows = []

            for seed in seeds:
                run_cfg = copy.deepcopy(candidate_cfg)
                run_cfg.train.seed = seed
                run_cfg.train.output_dir = (
                    base_output_dir / "search" / slug / f"seed_{seed}"
                )
                run_cfg.train.ckpt_dir = (
                    base_ckpt_dir / "search" / slug / f"seed_{seed}"
                )
                print(f"search candidate={slug} seed={seed}")
                set_seed(seed)
                metrics = run_train(run_cfg)
                row = {
                    "candidate": slug,
                    "candidate_index": candidate_index,
                    "batch_size": run_cfg.data.batch_size,
                    "lr": run_cfg.train.lr,
                    "weight_decay": run_cfg.train.weight_decay,
                    "head_lr_mult": run_cfg.train.head_lr_mult,
                    "momentum": run_cfg.train.momentum,
                    "label_smoothing": run_cfg.train.label_smoothing,
                    "scheduler": run_cfg.train.scheduler,
                    "grad_clip_norm": run_cfg.train.grad_clip_norm,
                    "fpn_channels": run_cfg.model.fpn_channels,
                    "num_parts": run_cfg.model.num_parts,
                    "part_window_size": run_cfg.model.part_window_size,
                    "relation_heads": run_cfg.model.relation_heads,
                    "bilinear_dim": run_cfg.model.bilinear_dim,
                    "seed": seed,
                    "best_val_top1": metrics["best_top1"],
                    "best_epoch": metrics["best_epoch"],
                    "last_val_top1": metrics["val_top1"],
                    "last_val_top5": metrics["val_top5"],
                }
                candidate_rows.append(row)
                search_rows.append(row)

            mean_top1, std_top1 = mean_std(
                row["best_val_top1"] for row in candidate_rows
            )
            selected_epoch = candidate_cfg.train.epochs
            candidate_summaries.append(
                {
                    "candidate": slug,
                    "candidate_index": candidate_index,
                    "hyperparameters": candidate,
                    "mean_best_val_top1": mean_top1,
                    "std_best_val_top1": std_top1,
                    "selected_epoch": selected_epoch,
                    "runs": candidate_rows,
                }
            )

        selected = max(
            candidate_summaries,
            key=lambda row: (
                row["mean_best_val_top1"],
                -row["std_best_val_top1"],
                -row["candidate_index"],
            ),
        )
        selection = {
            "selection_metric": "mean_best_val_top1",
            "tie_breakers": ["lower_std", "earlier_candidate"],
            "epoch_rule": "fixed_config_epochs",
            "val_ratio": cfg.data.val_ratio,
            "split_seed": cfg.data.split_seed,
            "model": cfg.model.name,
            "optimizer": cfg.train.optimizer,
            "selected_candidate": selected["candidate"],
            "selected_hyperparameters": selected["hyperparameters"],
            "selected_epoch": selected["selected_epoch"],
            "candidates": candidate_summaries,
        }
        save_json(selection, base_output_dir / "selection.json")
        save_csv(search_rows, base_output_dir / "search_summary.csv")
        print("selection", selection)

        final_cfg = apply_candidate_overrides(cfg, selected["hyperparameters"])
        final_cfg.train.epochs = selected["selected_epoch"]
        test_loader = build_dataloader(final_cfg, "test", shuffle=False)
        final_rows = []
        for seed in seeds:
            run_cfg = copy.deepcopy(final_cfg)
            run_cfg.train.seed = seed
            run_cfg.train.output_dir = base_output_dir / "final" / f"seed_{seed}"
            run_cfg.train.ckpt_dir = base_ckpt_dir / "final" / f"seed_{seed}"
            print(
                f"full_train model={run_cfg.model.name} seed={seed} "
                f"epochs={run_cfg.train.epochs}"
            )
            set_seed(seed)
            train_metrics = run_full_train(run_cfg)
            checkpoint_path = Path(train_metrics["checkpoint"])
            test_metrics = run_test_checkpoint(
                run_cfg,
                checkpoint_path,
                test_loader=test_loader,
            )
            final_rows.append(
                {
                    "seed": seed,
                    "model": run_cfg.model.name,
                    "epochs": run_cfg.train.epochs,
                    "train_loss": train_metrics["train_loss"],
                    "train_top1": train_metrics["train_top1"],
                    **test_metrics,
                }
            )

        final_summary = {
            **summarize_test_rows(cfg.model.name, final_rows),
            "selected_candidate": selected["candidate"],
            "selected_hyperparameters": selected["hyperparameters"],
            "selected_epoch": selected["selected_epoch"],
        }
        save_json(final_summary, base_output_dir / "final_test_summary.json")
        save_csv(final_rows, base_output_dir / "final_test_summary.csv")
        print("final_test_summary", final_summary)


def save_repeat_summary(cfg, seeds: list[int], rows: list[dict], base_output_dir: Path) -> None:
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


def run_repeated_train(cfg, seeds: list[int], run_name: str | None = None) -> None:
    experiment_name = run_name or "seeds_" + "_".join(str(seed) for seed in seeds)
    base_output_dir = Path(cfg.train.output_dir) / cfg.model.name / experiment_name
    base_ckpt_dir = Path(cfg.train.ckpt_dir) / cfg.model.name / experiment_name
    rows = []

    with RunLock(base_output_dir / ".run.lock"):
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

        save_repeat_summary(cfg, seeds, rows, base_output_dir)


def run_summarize(cfg, seeds: list[int], run_name: str | None = None) -> None:
    experiment_name = run_name or "seeds_" + "_".join(str(seed) for seed in seeds)
    base_output_dir = Path(cfg.train.output_dir) / cfg.model.name / experiment_name
    rows = []

    for run_index, seed in enumerate(seeds, start=1):
        history_path = (
            base_output_dir
            / f"seed_{seed}"
            / f"{cfg.model.name}_history.json"
        )
        if not history_path.exists():
            raise FileNotFoundError(f"Missing history file: {history_path}")

        with history_path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)

        history = result.get("history", [])
        if not history:
            raise ValueError(f"History is empty: {history_path}")

        last_metrics = history[-1]
        row = {
            "run": run_index,
            "seed": seed,
            "train_loss": last_metrics["train_loss"],
            "train_top1": last_metrics["train_top1"],
            "val_loss": last_metrics["val_loss"],
            "val_top1": last_metrics["val_top1"],
            "val_top5": last_metrics["val_top5"],
            "best_top1": result["best_top1"],
            "best_epoch": result["best_epoch"],
        }
        rows.append(row)

    save_repeat_summary(cfg, seeds, rows, base_output_dir)


def main() -> None:
    args = parse_args()
    cfg = build_config(
        root=args.data_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        resize_size=args.resize_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        epochs=args.epochs,
        lr=args.lr,
        head_lr_mult=args.head_lr_mult,
        seed=args.seed,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        momentum=args.momentum,
        label_smoothing=args.label_smoothing,
        scheduler=args.scheduler,
        grad_clip_norm=args.grad_clip_norm,
        device=args.device,
        output_dir=args.output_dir,
        ckpt_dir=args.ckpt_dir,
        use_bbox_crop=args.use_bbox_crop,
        pretrained=args.pretrained,
        fpn_channels=args.fpn_channels,
        num_parts=args.num_parts,
        part_window_size=args.part_window_size,
        relation_heads=args.relation_heads,
        bilinear_dim=args.bilinear_dim,
        name=args.model,
    )

    seeds = parse_seed_list(cfg.train.seed, args.seeds)
    if args.mode == "train":
        run_repeated_train(cfg, seeds, args.run_name)
    elif args.mode == "summarize":
        run_summarize(cfg, seeds, args.run_name)
    elif args.mode == "pipeline":
        run_selection_pipeline(cfg, seeds, args.run_name, args.search_space)
    else:
        if not args.checkpoints:
            raise ValueError("--checkpoints is required in test mode.")
        run_test_checkpoints(cfg, args.checkpoints)


if __name__ == "__main__":
    main()

