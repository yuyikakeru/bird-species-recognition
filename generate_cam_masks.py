from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF
from tqdm import tqdm

from config import DEFAULT_CUB_ROOT, build_config
from data_utils.transform import IMAGENET_MEAN, IMAGENET_STD
from main import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate smoothed Grad-CAM masks.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--model",
        default="swinv2_tiny",
        choices=["swinv2_tiny"],
    )
    parser.add_argument("--data-root", default=str(DEFAULT_CUB_ROOT))
    parser.add_argument("--output-root", default="datasets/CUB_200_2011/cam_masks")
    parser.add_argument("--split", choices=["train_full", "test"], default="train_full")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--resize-size", type=int, default=256)
    parser.add_argument("--gradcam-weight", type=float, default=0.65)
    parser.add_argument("--energy-weight", type=float, default=0.35)
    parser.add_argument("--spread-kernel", type=int, default=31)
    parser.add_argument("--smooth-kernel", type=int, default=15)
    return parser.parse_args()


def read_id_to_str(path: Path) -> dict[int, str]:
    values: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item_id, value = line.split(maxsplit=1)
            values[int(item_id)] = value
    return values


def normalize_map(mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    mask = mask - mask.amin(dim=(-2, -1), keepdim=True)
    return mask / mask.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)


def refine_mask_with_image(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = normalize_map(mask.clamp(0.0, 1.0))
    gray = image.float().mean(dim=1, keepdim=True)
    dx = F.pad((gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs(), (0, 0, 0, 1))
    edge = normalize_map(dx + dy)
    contrast = normalize_map(
        (image.float() - image.float().mean(dim=(-2, -1), keepdim=True))
        .square()
        .mean(dim=1, keepdim=True)
        .sqrt()
    )
    saturation = normalize_map(
        image.float().max(dim=1, keepdim=True).values
        - image.float().min(dim=1, keepdim=True).values
    )
    detail = normalize_map(0.45 * edge + 0.35 * contrast + 0.20 * saturation)
    refined = mask * (0.70 + 0.30 * detail)
    refined = F.avg_pool2d(refined, kernel_size=5, stride=1, padding=2)
    refined = normalize_map(refined)
    return normalize_map(0.65 * refined + 0.35 * mask).clamp(0.0, 1.0)


def odd_kernel(value: int) -> int:
    value = max(1, value)
    return value if value % 2 == 1 else value + 1


def cam_class_index(logits: torch.Tensor, label: int, split: str) -> int:
    if split == "test":
        return int(logits.argmax(dim=1).item())
    return label


def eval_crop(image: Image.Image, image_size: int, resize_size: int):
    original_width, original_height = image.size
    resized = TF.resize(image, resize_size, antialias=True)
    resized_width, resized_height = resized.size
    left = int(round((resized_width - image_size) / 2.0))
    top = int(round((resized_height - image_size) / 2.0))
    crop = TF.center_crop(resized, [image_size, image_size])
    return crop, (original_width, original_height, resized_width, resized_height, left, top)


def paste_crop_mask_to_original(mask: torch.Tensor, meta) -> Image.Image:
    original_width, original_height, resized_width, resized_height, left, top = meta
    mask_image = TF.to_pil_image(mask.cpu().clamp(0, 1))
    canvas = Image.new("L", (resized_width, resized_height), 0)
    canvas.paste(mask_image, (left, top))
    return canvas.resize((original_width, original_height), Image.Resampling.BILINEAR)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    images = read_id_to_str(data_root / "images.txt")
    labels = {
        image_id: int(label) - 1
        for image_id, label in read_id_to_str(data_root / "image_class_labels.txt").items()
    }
    splits = {
        image_id: int(flag)
        for image_id, flag in read_id_to_str(data_root / "train_test_split.txt").items()
    }
    if args.split == "train_full":
        image_ids = [image_id for image_id in sorted(images) if splits[image_id] == 1]
    else:
        image_ids = [image_id for image_id in sorted(images) if splits[image_id] == 0]
    if args.max_images is not None:
        image_ids = image_ids[: args.max_images]

    cfg = build_config(
        name=args.model,
        pretrained=False,
        image_size=args.image_size,
        resize_size=args.resize_size,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])

    spread_kernel = odd_kernel(args.spread_kernel)
    smooth_kernel = odd_kernel(args.smooth_kernel)
    spread_padding = spread_kernel // 2
    smooth_padding = smooth_kernel // 2

    for image_id in tqdm(image_ids, desc="cam"):
        relative_path = Path(images[image_id])
        image = Image.open(data_root / "images" / relative_path).convert("RGB")
        crop, meta = eval_crop(image, args.image_size, args.resize_size)
        crop_tensor = TF.to_tensor(crop).unsqueeze(0).to(device)
        image_tensor = TF.normalize(
            crop_tensor.squeeze(0),
            IMAGENET_MEAN,
            IMAGENET_STD,
        ).unsqueeze(0).to(device)

        model.zero_grad(set_to_none=True)
        if args.model == "swinv2_tiny":
            backbone = model.backbone
            feature_map = backbone.patch_embed(image_tensor)
            for layer in backbone.layers:
                feature_map = layer(feature_map)
            feature_map.retain_grad()
            logits = backbone.forward_head(backbone.norm(feature_map))
            fused = feature_map.permute(0, 3, 1, 2).contiguous()
            target_class = cam_class_index(logits, labels[image_id], args.split)
            score = logits[0, target_class]
        else:
            features = model(image_tensor, return_features=True)
            fused = features.get("part_feature_map")
            if fused is None:
                raise RuntimeError(f"{args.model} does not expose a part feature map.")
            fused.retain_grad()
            target_class = cam_class_index(features["logits"], labels[image_id], args.split)
            score = features["logits"][0, target_class]
        score.backward()

        if args.model == "swinv2_tiny":
            gradients = feature_map.grad.permute(0, 3, 1, 2).contiguous()
        else:
            gradients = fused.grad
        weights = gradients.mean(dim=(-2, -1), keepdim=True)
        gradcam = torch.relu((weights * fused).sum(dim=1, keepdim=True))
        gradcam = normalize_map(gradcam)

        energy = normalize_map(fused.detach().float().square().mean(dim=1, keepdim=True).sqrt())
        mask = args.gradcam_weight * gradcam.detach() + args.energy_weight * energy
        mask = normalize_map(mask)
        mask = F.interpolate(
            mask,
            size=(args.image_size, args.image_size),
            mode="bilinear",
            align_corners=False,
        )
        mask = F.max_pool2d(mask, kernel_size=spread_kernel, stride=1, padding=spread_padding)
        mask = F.avg_pool2d(mask, kernel_size=smooth_kernel, stride=1, padding=smooth_padding)
        mask = refine_mask_with_image(crop_tensor, mask)
        mask = normalize_map(mask)[0]

        output_path = output_root / relative_path.with_suffix(".png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        paste_crop_mask_to_original(mask, meta).save(output_path)


if __name__ == "__main__":
    main()
