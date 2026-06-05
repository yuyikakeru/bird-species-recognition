from __future__ import annotations

from torch import nn


class SwinV2TinyBaseline(nn.Module):
    """ImageNet-1K pretrained SwinV2-Tiny classifier for CUB_200_2011."""

    def __init__(
        self,
        num_classes: int = 200,
        pretrained: bool = True,
        image_size: int = 448,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise RuntimeError(
                "SwinV2-T requires the timm package. Install dependencies with "
                "`pip install -r requirements.txt`."
            ) from exc

        model_name = "swinv2_tiny_window16_256"
        try:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes,
            )
        except Exception as exc:
            if pretrained:
                raise RuntimeError(
                    "Failed to load ImageNet-1K pretrained SwinV2-T weights. "
                    "Check network access or rerun with --no-pretrained."
                ) from exc
            raise

        if image_size != 256:
            try:
                self.backbone.set_input_size(img_size=(image_size, image_size))
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to adapt {model_name} from 256x256 pretraining "
                    f"to {image_size}x{image_size} input. Choose an image size "
                    "whose SwinV2 feature maps can be partitioned into windows."
                ) from exc

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
