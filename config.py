from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CUB_ROOT = PROJECT_ROOT / "datasets" / "CUB_200_2011" / "CUB_200_2011"
SUPPORTED_MODELS = (
    "resnet50_baseline",
    "swinv2_tiny",
    "swinv2_tiny_fpn",
    "swinv2_tiny_fpn_parts",
    "swinv2_tiny_fpn_relation",
)


@dataclass
class DataConfig:
    root: Path = DEFAULT_CUB_ROOT
    image_size: int = 224
    resize_size: int = 256
    batch_size: int = 32
    num_workers: int = 8
    val_ratio: float = 0.2
    split_seed: int = 42
    use_bbox_crop: bool = False
    bbox_margin: float = 0.2
    pin_memory: bool = True


@dataclass
class TrainConfig:
    epochs: int = 50
    lr: float = 0.01
    weight_decay: float = 1e-4
    head_lr_mult: float = 5.0
    optimizer: str = "sgd"
    momentum: float = 0.9
    label_smoothing: float = 0.1
    scheduler: str = "cosine"
    seed: int = 42
    device: str = "auto"
    amp: bool = True
    log_interval: int = 20
    grad_clip_norm: float = 1.0
    output_dir: Path = PROJECT_ROOT / "log"
    ckpt_dir: Path = PROJECT_ROOT / "ckpt"


@dataclass
class ModelConfig:
    name: str = "resnet50_baseline"
    num_classes: int = 200
    pretrained: bool = True
    fpn_channels: int = 256
    num_parts: int = 6
    part_window_size: int = 3
    relation_heads: int = 4
    bilinear_dim: int = 256


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


def build_config(**overrides: object) -> ExperimentConfig:
    cfg = ExperimentConfig()

    for key, value in overrides.items():
        if value is None:
            continue
        if hasattr(cfg.data, key):
            setattr(cfg.data, key, value)
        elif hasattr(cfg.train, key):
            setattr(cfg.train, key, value)
        elif hasattr(cfg.model, key):
            setattr(cfg.model, key, value)
        else:
            raise KeyError(f"Unknown config option: {key}")

    cfg.data.root = Path(cfg.data.root)
    cfg.train.output_dir = Path(cfg.train.output_dir)
    cfg.train.ckpt_dir = Path(cfg.train.ckpt_dir)
    return validate_config(cfg)


def validate_config(cfg: ExperimentConfig) -> ExperimentConfig:
    positive_values = {
        "image_size": cfg.data.image_size,
        "resize_size": cfg.data.resize_size,
        "batch_size": cfg.data.batch_size,
        "epochs": cfg.train.epochs,
        "lr": cfg.train.lr,
        "grad_clip_norm": cfg.train.grad_clip_norm,
        "num_classes": cfg.model.num_classes,
    }
    if "fpn" in cfg.model.name:
        positive_values["fpn_channels"] = cfg.model.fpn_channels
    if cfg.model.name in {"swinv2_tiny_fpn_parts", "swinv2_tiny_fpn_relation"}:
        positive_values["num_parts"] = cfg.model.num_parts
        positive_values["part_window_size"] = cfg.model.part_window_size
    if cfg.model.name == "swinv2_tiny_fpn_relation":
        positive_values["relation_heads"] = cfg.model.relation_heads
        positive_values["bilinear_dim"] = cfg.model.bilinear_dim
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    if cfg.data.num_workers < 0:
        raise ValueError(f"num_workers cannot be negative: {cfg.data.num_workers}")
    if not cfg.data.use_bbox_crop and cfg.data.resize_size < cfg.data.image_size:
        raise ValueError("resize_size must be at least image_size")
    if not 0.0 < cfg.data.val_ratio < 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1: {cfg.data.val_ratio}")
    if cfg.data.bbox_margin < 0:
        raise ValueError(f"bbox_margin cannot be negative: {cfg.data.bbox_margin}")
    if cfg.train.weight_decay < 0:
        raise ValueError(f"weight_decay cannot be negative: {cfg.train.weight_decay}")
    if cfg.train.head_lr_mult <= 0:
        raise ValueError("head_lr_mult must be positive")
    if cfg.train.momentum < 0:
        raise ValueError("momentum cannot be negative")
    if not 0.0 <= cfg.train.label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")
    if (
        cfg.model.name in {"swinv2_tiny_fpn_parts", "swinv2_tiny_fpn_relation"}
        and cfg.model.part_window_size % 2 == 0
    ):
        raise ValueError("part_window_size must be odd")
    if (
        cfg.model.name == "swinv2_tiny_fpn_relation"
        and cfg.model.fpn_channels % cfg.model.relation_heads != 0
    ):
        raise ValueError("relation_heads must divide fpn_channels")
    if cfg.model.name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {cfg.model.name}")
    if cfg.train.optimizer not in {"sgd", "adamw"}:
        raise ValueError(f"Unsupported optimizer: {cfg.train.optimizer}")
    if cfg.train.scheduler not in {"cosine", "none"}:
        raise ValueError(f"Unsupported scheduler: {cfg.train.scheduler}")
    return cfg
