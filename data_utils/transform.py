from __future__ import annotations

from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(split: str, image_size: int = 448, resize_size: int = 512):
    if split == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.02,
                ),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    if split in {"val", "test"}:
        return transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    raise ValueError(f"Unsupported split: {split}")
