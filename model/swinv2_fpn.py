from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .swinv2_baseline import build_swinv2_tiny_backbone


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
    ) -> None:
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
    """Fuse SwinV2 stage-3 and stage-4 features with a lightweight FPN head."""

    def __init__(
        self,
        num_classes: int = 200,
        pretrained: bool = True,
        image_size: int = 224,
        fpn_channels: int = 256,
    ) -> None:
        super().__init__()
        if fpn_channels < 1:
            raise ValueError(f"fpn_channels must be positive, got {fpn_channels}")

        self.backbone = build_swinv2_tiny_backbone(
            num_classes=num_classes,
            pretrained=pretrained,
            image_size=image_size,
        )
        self.fpn_channels = fpn_channels
        self.baseline_logit_scale = 1.0
        self.fpn_logit_scale = 1.0

        self.stage3_lateral = ConvNormAct(384, fpn_channels, kernel_size=1)
        self.stage4_lateral = ConvNormAct(768, fpn_channels, kernel_size=1)
        self.fpn_smooth = ConvNormAct(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.fpn_attention = nn.Conv2d(fpn_channels, 1, kernel_size=1)
        self.fpn_classifier = nn.Linear(fpn_channels, num_classes)
        self._init_fpn_head()

    def _init_fpn_head(self) -> None:
        nn.init.zeros_(self.fpn_attention.weight)
        nn.init.zeros_(self.fpn_attention.bias)
        nn.init.trunc_normal_(self.fpn_classifier.weight, std=0.02)
        nn.init.zeros_(self.fpn_classifier.bias)

    def get_optimizer_param_groups(
        self,
        base_lr: float,
        weight_decay: float,
        head_lr_mult: float = 5.0,
    ) -> list[dict[str, object]]:
        groups: dict[tuple[float, float], dict[str, object]] = {}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            is_head = not name.startswith("backbone.")
            lr = base_lr * (head_lr_mult if is_head else 1.0)
            decay = 0.0 if self._should_skip_weight_decay(name, param) else weight_decay
            key = (lr, decay)
            groups.setdefault(key, {"params": [], "lr": lr, "weight_decay": decay})
            groups[key]["params"].append(param)
        return list(groups.values())

    @staticmethod
    def _should_skip_weight_decay(name: str, param: torch.nn.Parameter) -> bool:
        lowered = name.lower()
        return (
            param.ndim <= 1
            or lowered.endswith(".bias")
            or "norm" in lowered
            or "relative_position_bias" in lowered
            or "absolute_pos_embed" in lowered
            or "logit_scale" in lowered
        )

    @staticmethod
    def _to_nchw(features: torch.Tensor) -> torch.Tensor:
        return features.permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _attention_pool(
        feature_map: torch.Tensor,
        score_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, channels, height, width = feature_map.shape
        weights = torch.softmax(score_map.flatten(2).squeeze(1), dim=1)
        flattened = feature_map.flatten(2).transpose(1, 2)
        pooled = torch.sum(
            flattened * weights.unsqueeze(-1).to(feature_map.dtype),
            dim=1,
        )
        return pooled, weights.reshape(batch_size, 1, height, width)

    def _extract_stage_features(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone.patch_embed(x)
        stage3 = None
        for index, layer in enumerate(self.backbone.layers):
            features = layer(features)
            if index == 2:
                stage3 = features

        if stage3 is None:
            raise RuntimeError("Failed to collect SwinV2 stage-3 features.")

        stage4 = features
        baseline_logits = self.backbone.forward_head(self.backbone.norm(stage4))
        return baseline_logits, stage3, stage4

    def _fuse_features(
        self,
        stage3: torch.Tensor,
        stage4: torch.Tensor,
    ) -> torch.Tensor:
        stage3_map = self.stage3_lateral(self._to_nchw(stage3))
        stage4_map = self.stage4_lateral(self._to_nchw(stage4))
        stage4_map = F.interpolate(
            stage4_map,
            size=stage3_map.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.fpn_smooth(stage3_map + stage4_map)

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        baseline_logits, stage3, stage4 = self._extract_stage_features(x)
        fused_map = self._fuse_features(stage3, stage4)
        fpn_score_map = self.fpn_attention(fused_map)
        fpn_feature, fpn_attention = self._attention_pool(fused_map, fpn_score_map)
        fpn_logits = self.fpn_classifier(fpn_feature)
        logits = self.baseline_logit_scale * baseline_logits + self.fpn_logit_scale * fpn_logits

        return {
            "logits": logits,
            "baseline_logits": baseline_logits,
            "baseline_logit_scale": torch.full_like(fpn_logits[:, :1], self.baseline_logit_scale),
            "stage3_features": stage3,
            "stage4_features": stage4,
            "fpn_feature_map": fused_map,
            "fpn_score_map": fpn_score_map,
            "fpn_attention": fpn_attention,
            "fpn_feature": fpn_feature,
            "fpn_logits": fpn_logits,
            "fpn_gate": torch.full_like(fpn_logits[:, :1], self.fpn_logit_scale),
        }

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        features = self.forward_features(x)
        if return_features:
            return features
        return features["logits"]


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
