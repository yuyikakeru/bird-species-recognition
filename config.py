from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CUB_ROOT = PROJECT_ROOT / "datasets" / "CUB_200_2011" / "CUB_200_2011"


@dataclass
class DataConfig:
    root: Path = DEFAULT_CUB_ROOT
    image_size: int = 224
    resize_size: int = 256
    batch_size: int = 16
    num_workers: int = 4
    use_bbox_crop: bool = False
    bbox_margin: float = 0.2
    return_parts: bool = True
    pin_memory: bool = True


@dataclass
class TrainConfig:
    epochs: int = 80
    lr: float = 0.01
    weight_decay: float = 1e-4
    optimizer: str = "sgd"
    momentum: float = 0.9
    label_smoothing: float = 0.1
    scheduler: str = "cosine"
    seed: int = 42
    device: str = "auto"
    amp: bool = True
    log_interval: int = 20
    early_stop_patience: int = 5
    early_stop_min_delta: float = 0.1
    max_train_batches: Optional[int] = None
    max_val_batches: Optional[int] = None
    output_dir: Path = PROJECT_ROOT / "log"
    ckpt_dir: Path = PROJECT_ROOT / "ckpt"


@dataclass
class ModelConfig:
    name: str = "smoke"
    num_classes: int = 200
    pretrained: bool = True
    fpn_channels: int = 256


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
    return cfg
