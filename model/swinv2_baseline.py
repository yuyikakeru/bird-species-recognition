from __future__ import annotations

from torch import nn


def build_swinv2_tiny_backbone(
    num_classes: int,
    pretrained: bool,
    image_size: int,
) -> nn.Module:
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError(
            "SwinV2-Tiny requires timm. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    model_name = "swinv2_tiny_window16_256"
    try:
        backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )
    except Exception as exc:
        if pretrained:
            raise RuntimeError(
                "Failed to load ImageNet-1K pretrained SwinV2-Tiny weights. "
                "Check network access or rerun with --no-pretrained."
            ) from exc
        raise

    if image_size != 256:
        try:
            backbone.set_input_size(img_size=(image_size, image_size))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to adapt {model_name} to {image_size}x{image_size} input."
            ) from exc
    return backbone


class SwinV2TinyBaseline(nn.Module):
    """ImageNet-1K pretrained SwinV2-Tiny classifier for CUB_200_2011."""

    def __init__(
        self,
        num_classes: int = 200,
        pretrained: bool = True,
        image_size: int = 448,
    ) -> None:
        super().__init__()
        self.backbone = build_swinv2_tiny_backbone(
            num_classes=num_classes,
            pretrained=pretrained,
            image_size=image_size,
        )

    def forward(self, x):
        return self.backbone(x)


def build_swinv2_tiny_baseline(
    num_classes: int = 200,
    pretrained: bool = True,
    image_size: int = 448,
) -> SwinV2TinyBaseline:
    return SwinV2TinyBaseline(
        num_classes=num_classes,
        pretrained=pretrained,
        image_size=image_size,
    )
