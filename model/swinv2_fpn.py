from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .swinv2_baseline import build_swinv2_tiny_backbone


def make_group_norm(num_channels: int) -> nn.GroupNorm:
    for num_groups in (32, 16, 8, 4, 2):
        if num_channels % num_groups == 0:
            return nn.GroupNorm(num_groups, num_channels)
    return nn.GroupNorm(1, num_channels)


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
            make_group_norm(out_channels),
            nn.GELU(),
        )


class AttentionPool2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden_channels = max(channels // 4, 32)
        self.score = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.score(x).flatten(2).softmax(dim=-1)
        values = x.flatten(2)
        return torch.sum(values * weights, dim=-1)


class SwinV2TinyFPN(nn.Module):
    """SwinV2-Tiny with top-down C2/C3/C4 fusion for fine-grained cues."""

    def __init__(
        self,
        num_classes: int = 200,
        pretrained: bool = True,
        image_size: int = 224,
        fpn_channels: int = 256,
    ) -> None:
        super().__init__()
        self.backbone = build_swinv2_tiny_backbone(
            num_classes=num_classes,
            pretrained=pretrained,
            image_size=image_size,
        )

        self.lateral_c2 = ConvNormAct(192, fpn_channels, kernel_size=1)
        self.lateral_c3 = ConvNormAct(384, fpn_channels, kernel_size=1)
        self.lateral_c4 = ConvNormAct(768, fpn_channels, kernel_size=1)
        self.smooth_p2 = ConvNormAct(fpn_channels, fpn_channels, kernel_size=3)
        self.smooth_p3 = ConvNormAct(fpn_channels, fpn_channels, kernel_size=3)
        self.smooth_p4 = ConvNormAct(fpn_channels, fpn_channels, kernel_size=3)
        self.fusion = nn.Sequential(
            ConvNormAct(fpn_channels * 3, fpn_channels, kernel_size=3),
            ConvNormAct(fpn_channels, fpn_channels, kernel_size=3),
        )
        self.pool_p2 = AttentionPool2d(fpn_channels)
        self.pool_p3 = AttentionPool2d(fpn_channels)
        self.pool_p4 = AttentionPool2d(fpn_channels)
        self.pool_fused = AttentionPool2d(fpn_channels)
        self.scale_mixer = nn.Sequential(
            nn.LayerNorm(fpn_channels * 4),
            nn.Linear(fpn_channels * 4, fpn_channels * 2),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(fpn_channels * 2, fpn_channels),
        )
        self.fpn_norm = nn.LayerNorm(fpn_channels)
        self.fpn_drop = nn.Dropout(p=0.1)
        self.fpn_classifier = nn.Linear(fpn_channels, num_classes)
        self.residual_gate = nn.Sequential(
            nn.LayerNorm(fpn_channels),
            nn.Linear(fpn_channels, max(fpn_channels // 4, 32)),
            nn.GELU(),
            nn.Linear(max(fpn_channels // 4, 32), 1),
        )
        self.residual_gate_bias = nn.Parameter(torch.tensor(-2.0))

        self._init_residual_head()

    def _init_residual_head(self) -> None:
        nn.init.trunc_normal_(self.fpn_classifier.weight, std=0.02)
        nn.init.zeros_(self.fpn_classifier.bias)
        final_gate = self.residual_gate[-1]
        nn.init.zeros_(final_gate.weight)
        nn.init.zeros_(final_gate.bias)

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

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return logits and spatial feature maps for downstream FG modules."""
        c2, c3, c4 = self._extract_stages(x)
        baseline_logits = self.backbone.forward_head(self.backbone.norm(c4))

        c2 = self._to_nchw(c2)
        c3 = self._to_nchw(c3)
        c4 = self._to_nchw(c4)

        lateral_p2 = self.lateral_c2(c2)
        lateral_p3 = self.lateral_c3(c3)
        lateral_p4 = self.lateral_c4(c4)

        p4 = self.smooth_p4(lateral_p4)
        p3 = lateral_p3 + F.interpolate(
            p4,
            size=lateral_p3.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        p3 = self.smooth_p3(p3)
        p2 = lateral_p2 + F.interpolate(
            p3,
            size=lateral_p2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        p2 = self.smooth_p2(p2)

        target_size = p2.shape[-2:]
        p3_up = F.interpolate(p3, size=target_size, mode="bilinear", align_corners=False)
        p4_up = F.interpolate(p4, size=target_size, mode="bilinear", align_corners=False)

        fused = self.fusion(torch.cat([p2, p3_up, p4_up], dim=1))
        fpn_feature = torch.cat(
            [
                self.pool_p2(p2),
                self.pool_p3(p3),
                self.pool_p4(p4),
                self.pool_fused(fused),
            ],
            dim=1,
        )
        fpn_feature = self.scale_mixer(fpn_feature)
        fpn_feature = self.fpn_drop(self.fpn_norm(fpn_feature))
        fpn_logits = self.fpn_classifier(fpn_feature)
        residual_gate = torch.sigmoid(
            self.residual_gate(fpn_feature) + self.residual_gate_bias
        )
        logits = baseline_logits + residual_gate * fpn_logits

        return {
            "logits": logits,
            "baseline_logits": baseline_logits,
            "fpn_logits": fpn_logits,
            "residual_gate": residual_gate,
            "c2": c2,
            "c3": c3,
            "c4": c4,
            "p2": p2,
            "p3": p3,
            "p4": p4,
            "fused": fused,
            "fpn_feature": fpn_feature,
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
