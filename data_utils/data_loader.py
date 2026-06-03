from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .transform import build_transforms


class CUBDataset(Dataset):
    """CUB_200_2011 dataset with optional bbox crop and part metadata."""

    def __init__(
        self,
        root: Path | str,
        split: str,
        transform=None,
        use_bbox_crop: bool = False,
        bbox_margin: float = 0.2,
        return_parts: bool = True,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.use_bbox_crop = use_bbox_crop
        self.bbox_margin = bbox_margin
        self.return_parts = return_parts

        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")

        self.images = self._read_id_to_str(self.root / "images.txt")
        self.labels = {
            image_id: int(label) - 1
            for image_id, label in self._read_id_to_str(
                self.root / "image_class_labels.txt"
            ).items()
        }
        self.classes = {
            int(class_id) - 1: name
            for class_id, name in self._read_id_to_str(self.root / "classes.txt").items()
        }
        self.split_flags = {
            image_id: int(flag)
            for image_id, flag in self._read_id_to_str(
                self.root / "train_test_split.txt"
            ).items()
        }
        self.bboxes = self._read_bboxes(self.root / "bounding_boxes.txt")
        self.parts = self._read_parts(self.root / "parts" / "part_locs.txt")

        expected_flag = 1 if split == "train" else 0
        self.image_ids = [
            image_id
            for image_id in sorted(self.images)
            if self.split_flags[image_id] == expected_flag
        ]

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> Dict[str, object]:
        image_id = self.image_ids[index]
        relative_path = self.images[image_id]
        image_path = self.root / "images" / relative_path

        image = Image.open(image_path).convert("RGB")
        bbox = self.bboxes.get(image_id, (0.0, 0.0, 0.0, 0.0))

        if self.use_bbox_crop:
            image = self._crop_with_bbox(image, bbox)

        if self.transform is not None:
            image = self.transform(image)

        label = self.labels[image_id]
        sample: Dict[str, object] = {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "image_id": torch.tensor(image_id, dtype=torch.long),
            "path": str(image_path),
            "class_name": self.classes[label],
            "bbox": torch.tensor(bbox, dtype=torch.float32),
        }

        if self.return_parts:
            sample["parts"] = torch.tensor(
                self.parts.get(image_id, self._empty_parts()),
                dtype=torch.float32,
            )

        return sample

    @staticmethod
    def _read_id_to_str(path: Path) -> Dict[int, str]:
        values: Dict[int, str] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                item_id, value = line.split(maxsplit=1)
                values[int(item_id)] = value
        return values

    @staticmethod
    def _read_bboxes(path: Path) -> Dict[int, Tuple[float, float, float, float]]:
        values: Dict[int, Tuple[float, float, float, float]] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                image_id, x, y, width, height = line.strip().split()
                values[int(image_id)] = (float(x), float(y), float(width), float(height))
        return values

    @staticmethod
    def _empty_parts() -> List[List[float]]:
        return [[0.0, 0.0, 0.0] for _ in range(15)]

    @classmethod
    def _read_parts(cls, path: Path) -> Dict[int, List[List[float]]]:
        if not path.exists():
            return {}

        values: Dict[int, List[List[float]]] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                image_id, part_id, x, y, visible = line.strip().split()
                image_id_int = int(image_id)
                part_index = int(part_id) - 1
                values.setdefault(image_id_int, cls._empty_parts())
                values[image_id_int][part_index] = [
                    float(x),
                    float(y),
                    float(visible),
                ]
        return values

    def _crop_with_bbox(
        self, image: Image.Image, bbox: Tuple[float, float, float, float]
    ) -> Image.Image:
        x, y, width, height = bbox
        if width <= 0 or height <= 0:
            return image

        image_width, image_height = image.size
        margin_x = width * self.bbox_margin
        margin_y = height * self.bbox_margin
        left = max(0, int(x - margin_x))
        top = max(0, int(y - margin_y))
        right = min(image_width, int(x + width + margin_x))
        bottom = min(image_height, int(y + height + margin_y))
        return image.crop((left, top, right, bottom))


def build_dataset(cfg, split: str) -> CUBDataset:
    transform = build_transforms(
        split,
        cfg.data.image_size,
        cfg.data.resize_size,
        use_bbox_crop=cfg.data.use_bbox_crop,
    )
    return CUBDataset(
        root=cfg.data.root,
        split=split,
        transform=transform,
        use_bbox_crop=cfg.data.use_bbox_crop,
        bbox_margin=cfg.data.bbox_margin,
        return_parts=cfg.data.return_parts,
    )


def build_dataloader(cfg, split: str, shuffle: Optional[bool] = None) -> DataLoader:
    dataset = build_dataset(cfg, split)
    if shuffle is None:
        shuffle = split == "train"

    return DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=shuffle,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )


def summarize_batch(batch: Dict[str, object]) -> Dict[str, object]:
    summary = {
        "image_shape": tuple(batch["image"].shape),
        "label_shape": tuple(batch["label"].shape),
        "first_image_id": int(batch["image_id"][0]),
        "first_label": int(batch["label"][0]),
        "first_path": batch["path"][0],
        "has_bbox": "bbox" in batch,
        "has_parts": "parts" in batch,
    }
    if "bbox" in batch:
        summary["bbox_shape"] = tuple(batch["bbox"].shape)
    if "parts" in batch:
        summary["parts_shape"] = tuple(batch["parts"].shape)
    return summary
