from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )


class SwinV2TinyFPN(nn.Module):
    """SwinV2-Tiny with a lightweight C2/C3/C4 FPN classifier."""

    def __init__(
        self,
        num_classes: int = 200,
        pretrained: bool = True,
        image_size: int = 224,
        fpn_channels: int = 256,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise RuntimeError(
                "SwinV2-T FPN requires the timm package. Install dependencies with "
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
                    f"to {image_size}x{image_size} input."
                ) from exc

        self.lateral_c2 = ConvNormAct(192, fpn_channels, kernel_size=1)
        self.lateral_c3 = ConvNormAct(384, fpn_channels, kernel_size=1)
        self.lateral_c4 = ConvNormAct(768, fpn_channels, kernel_size=1)
        self.fusion = nn.Sequential(
            ConvNormAct(fpn_channels * 3, fpn_channels, kernel_size=3),
            ConvNormAct(fpn_channels, fpn_channels, kernel_size=3),
        )
        self.fpn_norm = nn.LayerNorm(fpn_channels)
        self.fpn_drop = nn.Dropout(p=0.1)
        self.fpn_classifier = nn.Linear(fpn_channels, num_classes)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

        self._init_residual_head()

    def _init_residual_head(self) -> None:
        nn.init.trunc_normal_(self.fpn_classifier.weight, std=0.02)
        nn.init.zeros_(self.fpn_classifier.bias)

    @staticmethod
    def _to_nchw(feature: torch.Tensor) -> torch.Tensor:
        return feature.permute(0, 3, 1, 2).contiguous()

    def _extract_stages(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.backbone.patch_embed(x)
        c2 = c3 = c4 = None
        for index, layer in enumerate(self.backbone.layers):
            x = layer(x)
            if index == 1:
                c2 = x
            elif index == 2:
                c3 = x
            elif index == 3:
                c4 = x

        if c2 is None or c3 is None or c4 is None:
            raise RuntimeError("Failed to collect C2/C3/C4 SwinV2 stage features.")
        return c2, c3, c4

    def _forward_once(self, x: torch.Tensor) -> torch.Tensor:
        c2, c3, c4 = self._extract_stages(x)
        baseline_logits = self.backbone.forward_head(self.backbone.norm(c4))

        p2 = self.lateral_c2(self._to_nchw(c2))
        p3 = self.lateral_c3(self._to_nchw(c3))
        p4 = self.lateral_c4(self._to_nchw(c4))
        target_size = p2.shape[-2:]
        p3 = F.interpolate(p3, size=target_size, mode="bilinear", align_corners=False)
        p4 = F.interpolate(p4, size=target_size, mode="bilinear", align_corners=False)

        fused = self.fusion(torch.cat([p2, p3, p4], dim=1))
        fpn_feature = F.adaptive_avg_pool2d(fused, output_size=1).flatten(1)
        fpn_feature = self.fpn_drop(self.fpn_norm(fpn_feature))
        fpn_logits = self.fpn_classifier(fpn_feature)
        return baseline_logits + self.residual_scale * fpn_logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_once(x)


def build_swinv2_tiny_fpn(
    num_classes: int = 200,
    pretrained: bool = True,
    image_size: int = 224,
    fpn_channels: int = 256,
) -> SwinV2TinyFPN:
    return SwinV2TinyFPN(
        num_classes=num_classes,
        pretrained=pretrained,
        image_size=image_size,
        fpn_channels=fpn_channels,
    )
