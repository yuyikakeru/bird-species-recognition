from __future__ import annotations

import torch
from torchvision import transforms
from torchvision.transforms import functional as TF


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageTransform:
    """Image preprocessing for training and evaluation."""

    def __init__(
        self,
        split: str,
        image_size: int,
        resize_size: int,
        use_bbox_crop: bool,
    ) -> None:
        self.training = split in {"train", "train_full"}
        self.image_size = image_size
        self.resize_size = resize_size
        self.use_bbox_crop = use_bbox_crop
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.02,
        )

    def __call__(self, image, mask=None):
        if self.training:
            if self.use_bbox_crop:
                image = TF.resize(
                    image,
                    [self.image_size, self.image_size],
                    antialias=True,
                )
                if mask is not None:
                    mask = TF.resize(
                        mask,
                        [self.image_size, self.image_size],
                        interpolation=transforms.InterpolationMode.BILINEAR,
                        antialias=True,
                    )
            else:
                top, left, height, width = transforms.RandomResizedCrop.get_params(
                    image,
                    scale=(0.65, 1.0),
                    ratio=(3.0 / 4.0, 4.0 / 3.0),
                )
                image = TF.resized_crop(
                    image,
                    top,
                    left,
                    height,
                    width,
                    [self.image_size, self.image_size],
                    antialias=True,
                )
                if mask is not None:
                    mask = TF.resized_crop(
                        mask,
                        top,
                        left,
                        height,
                        width,
                        [self.image_size, self.image_size],
                        interpolation=transforms.InterpolationMode.BILINEAR,
                        antialias=True,
                    )

            if bool(torch.rand(()) < 0.5):
                image = TF.hflip(image)
                if mask is not None:
                    mask = TF.hflip(mask)
            image = self.color_jitter(image)
        elif self.use_bbox_crop:
            image = TF.resize(
                image,
                [self.image_size, self.image_size],
                antialias=True,
            )
            if mask is not None:
                mask = TF.resize(
                    mask,
                    [self.image_size, self.image_size],
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True,
                )
        else:
            image = TF.resize(image, self.resize_size, antialias=True)
            image = TF.center_crop(image, [self.image_size, self.image_size])
            if mask is not None:
                mask = TF.resize(
                    mask,
                    self.resize_size,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True,
                )
                mask = TF.center_crop(mask, [self.image_size, self.image_size])

        image_tensor = TF.normalize(
            TF.to_tensor(image),
            IMAGENET_MEAN,
            IMAGENET_STD,
        )
        if mask is None:
            return image_tensor

        mask_tensor = TF.to_tensor(mask).clamp(0.0, 1.0)
        return image_tensor, mask_tensor


def build_transforms(
    split: str,
    image_size: int = 448,
    resize_size: int = 512,
    use_bbox_crop: bool = False,
) -> ImageTransform:
    if split not in {"train", "val", "train_full", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    return ImageTransform(split, image_size, resize_size, use_bbox_crop)
