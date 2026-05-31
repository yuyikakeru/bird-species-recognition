from __future__ import annotations

from torch import nn
from torchvision import models


class ResNet50Baseline(nn.Module):
    """ImageNet-pretrained ResNet50 classifier for CUB_200_2011."""

    def __init__(self, num_classes: int = 200, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        try:
            self.backbone = models.resnet50(weights=weights)
        except Exception as exc:
            if pretrained:
                raise RuntimeError(
                    "Failed to load ImageNet pretrained ResNet50 weights. "
                    "Check network access or rerun with --no-pretrained."
                ) from exc
            raise
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def build_resnet50_baseline(
    num_classes: int = 200,
    pretrained: bool = True,
) -> ResNet50Baseline:
    return ResNet50Baseline(num_classes=num_classes, pretrained=pretrained)
