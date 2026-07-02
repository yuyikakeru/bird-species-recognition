from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .transform import build_transforms


class CUBDataset(Dataset):
    """CUB_200_2011 classification dataset with an optional bbox crop."""

    def __init__(
        self,
        root: Path | str,
        split: str,
        transform=None,
        cam_root: Path | str | None = None,
        use_bbox_crop: bool = False,
        bbox_margin: float = 0.2,
        val_ratio: float = 0.2,
        split_seed: int = 42,
    ) -> None:
        self.root = Path(root)
        self.cam_root = Path(cam_root) if cam_root else None
        self.split = split
        self.transform = transform
        self.use_bbox_crop = use_bbox_crop
        self.bbox_margin = bbox_margin
        self.val_ratio = val_ratio
        self.split_seed = split_seed

        if split not in {"train", "val", "train_full", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        if not 0.0 < val_ratio < 1.0:
            raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")

        required_files = [
            self.root / "images.txt",
            self.root / "image_class_labels.txt",
            self.root / "train_test_split.txt",
        ]
        if use_bbox_crop:
            required_files.append(self.root / "bounding_boxes.txt")
        missing_files = [str(path) for path in required_files if not path.is_file()]
        if missing_files:
            raise FileNotFoundError(
                "CUB dataset is incomplete; missing: " + ", ".join(missing_files)
            )

        self.images = self._read_id_to_str(self.root / "images.txt")
        self.labels = {
            image_id: int(label) - 1
            for image_id, label in self._read_id_to_str(
                self.root / "image_class_labels.txt"
            ).items()
        }
        self.split_flags = {
            image_id: int(flag)
            for image_id, flag in self._read_id_to_str(
                self.root / "train_test_split.txt"
            ).items()
        }
        self.bboxes = (
            self._read_bboxes(self.root / "bounding_boxes.txt")
            if use_bbox_crop
            else {}
        )

        image_ids = set(self.images)
        if image_ids != set(self.labels) or image_ids != set(self.split_flags):
            raise ValueError(
                "images.txt, image_class_labels.txt, and train_test_split.txt "
                "must contain the same image IDs."
            )
        if use_bbox_crop and not image_ids.issubset(self.bboxes):
            raise ValueError("bounding_boxes.txt is missing one or more image IDs.")

        official_train_ids = [
            image_id for image_id in sorted(self.images) if self.split_flags[image_id] == 1
        ]
        official_test_ids = [
            image_id for image_id in sorted(self.images) if self.split_flags[image_id] == 0
        ]

        if split == "test":
            self.image_ids = official_test_ids
        elif split == "train_full":
            self.image_ids = official_train_ids
        else:
            train_ids, val_ids = self._stratified_train_val_split(official_train_ids)
            self.image_ids = train_ids if split == "train" else val_ids

    def __len__(self) -> int:
        return len(self.image_ids)

    def _stratified_train_val_split(
        self, image_ids: list[int]
    ) -> tuple[list[int], list[int]]:
        ids_by_label: dict[int, list[int]] = defaultdict(list)
        for image_id in image_ids:
            ids_by_label[self.labels[image_id]].append(image_id)

        rng = random.Random(self.split_seed)
        train_ids: list[int] = []
        val_ids: list[int] = []
        for label in sorted(ids_by_label):
            class_ids = ids_by_label[label]
            rng.shuffle(class_ids)
            val_count = min(
                max(1, round(len(class_ids) * self.val_ratio)),
                len(class_ids) - 1,
            )
            val_ids.extend(class_ids[:val_count])
            train_ids.extend(class_ids[val_count:])

        return sorted(train_ids), sorted(val_ids)

    def __getitem__(self, index: int) -> dict[str, object]:
        image_id = self.image_ids[index]
        relative_path = self.images[image_id]
        image_path = self.root / "images" / relative_path

        with Image.open(image_path) as source:
            image = source.convert("RGB")

        if self.use_bbox_crop:
            image = self._crop_with_bbox(image, self.bboxes[image_id])

        cam_mask = None
        if self.cam_root is not None:
            cam_path = self._cam_path(relative_path)
            if not cam_path.is_file():
                raise FileNotFoundError(f"Missing CAM mask: {cam_path}")
            with Image.open(cam_path) as source:
                cam_mask = source.convert("L")

        if self.transform is not None:
            transformed = self.transform(image, cam_mask)
            if cam_mask is None:
                image = transformed
            else:
                image, cam_mask = transformed

        item = {
            "image": image,
            "label": self.labels[image_id],
            "image_id": image_id,
        }
        if cam_mask is not None:
            item["cam_mask"] = cam_mask
        return item

    def _cam_path(self, relative_path: str) -> Path:
        image_relative_path = Path(relative_path)
        return self.cam_root / image_relative_path.with_suffix(".png")

    @staticmethod
    def _read_id_to_str(path: Path) -> dict[int, str]:
        values: dict[int, str] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                item_id, value = line.split(maxsplit=1)
                values[int(item_id)] = value
        return values

    @staticmethod
    def _read_bboxes(path: Path) -> dict[int, tuple[float, float, float, float]]:
        values: dict[int, tuple[float, float, float, float]] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                image_id, x, y, width, height = line.strip().split()
                values[int(image_id)] = (float(x), float(y), float(width), float(height))
        return values

    def _crop_with_bbox(
        self,
        image: Image.Image,
        bbox: tuple[float, float, float, float],
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
        cam_root=cfg.data.cam_root,
        use_bbox_crop=cfg.data.use_bbox_crop,
        bbox_margin=cfg.data.bbox_margin,
        val_ratio=cfg.data.val_ratio,
        split_seed=cfg.data.split_seed,
    )


def build_dataloader(cfg, split: str, shuffle: bool | None = None) -> DataLoader:
    dataset = build_dataset(cfg, split)
    if shuffle is None:
        shuffle = split in {"train", "train_full"}

    return DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=shuffle,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.num_workers > 0,
    )
