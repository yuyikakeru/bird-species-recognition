from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .swinv2_fpn import ConvNormAct, SwinV2TinyFPN


class SwinV2TinyFPNParts(SwinV2TinyFPN):
    """CAM-guided Top-K parts on fused 14x14 FPN features."""

    def __init__(
        self,
        num_classes: int = 200,
        pretrained: bool = True,
        image_size: int = 224,
        fpn_channels: int = 256,
        num_parts: int = 6,
        part_window_size: int = 3,
        part_warmup_epochs: int = 10,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            pretrained=pretrained,
            image_size=image_size,
            fpn_channels=fpn_channels,
        )
        if num_parts < 1:
            raise ValueError(f"num_parts must be positive, got {num_parts}")
        if part_window_size < 1 or part_window_size % 2 == 0:
            raise ValueError(
                "part_window_size must be a positive odd integer, "
                f"got {part_window_size}"
            )
        if part_warmup_epochs < 0:
            raise ValueError(
                "part_warmup_epochs cannot be negative, "
                f"got {part_warmup_epochs}"
            )

        self.num_parts = num_parts
        self.part_window_size = part_window_size
        self.part_warmup_epochs = part_warmup_epochs
        self.current_epoch = 0
        self.foreground_floor = 0.05
        self.part_logit_scale = 1.0

        hidden_channels = max(fpn_channels // 4, 32)
        self.part_score = nn.Sequential(
            ConvNormAct(fpn_channels, hidden_channels, kernel_size=1),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        self.part_pool_logits = nn.Parameter(torch.zeros(num_parts))
        self.part_classifier = nn.Linear(fpn_channels, num_classes)
        self._init_part_head()

    def _init_part_head(self) -> None:
        nn.init.zeros_(self.part_score[-1].weight)
        nn.init.zeros_(self.part_score[-1].bias)
        nn.init.trunc_normal_(self.part_classifier.weight, std=0.02)
        nn.init.zeros_(self.part_classifier.bias)

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def _use_soft_part_warmup(self) -> bool:
        return (
            self.training
            and self.part_warmup_epochs > 0
            and 0 < self.current_epoch <= self.part_warmup_epochs
        )

    @staticmethod
    def _normalize_map(response: torch.Tensor) -> torch.Tensor:
        response = response.float()
        flat = response.flatten(2)
        min_value = flat.min(dim=-1).values[..., None, None]
        max_value = flat.max(dim=-1).values[..., None, None]
        normalized = (response - min_value) / (max_value - min_value).clamp_min(1e-6)
        return normalized.to(response.dtype)

    def _attention_from_inputs(
        self,
        feature_map: torch.Tensor,
        cam_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, bool]:
        if cam_mask is not None:
            attention = F.interpolate(
                cam_mask.float(),
                size=feature_map.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            return self._normalize_map(attention).to(feature_map.dtype), True

        raise ValueError(
            "SwinV2 FPN parts models require cam_mask during training and evaluation. "
            "Generate CAM masks for the active split and pass --cam-root."
        )

    def _score_foreground_map(
        self,
        feature_map: torch.Tensor,
        attention: torch.Tensor,
        used_cam: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        foreground_map = feature_map * attention.clamp_min(self.foreground_floor)
        raw_score = torch.sigmoid(self.part_score(foreground_map))
        if used_cam:
            selection_attention = attention.clamp_min(self.foreground_floor)
            score_map = raw_score * selection_attention
        else:
            score_map = raw_score
        return score_map, foreground_map

    def _select_part_candidates(
        self,
        feature_map: torch.Tensor,
        score_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, channels, height, width = feature_map.shape
        required_candidates = min(self.num_parts, height * width)
        candidate_scores, candidate_indices = torch.topk(
            score_map.flatten(2).squeeze(1),
            k=required_candidates,
            dim=1,
        )

        y = candidate_indices.div(width, rounding_mode="floor").to(feature_map.dtype)
        x = (candidate_indices % width).to(feature_map.dtype)
        coords = torch.stack(
            [
                x * (2.0 / max(width - 1, 1)) - 1.0,
                y * (2.0 / max(height - 1, 1)) - 1.0,
            ],
            dim=-1,
        )

        flattened = feature_map.flatten(2).transpose(1, 2)
        gather_indices = candidate_indices.unsqueeze(-1).expand(-1, -1, channels)
        candidate_features = torch.gather(flattened, 1, gather_indices)
        return candidate_features, coords, candidate_scores, candidate_indices

    @staticmethod
    def _soft_attention_pool_parts(
        feature_map: torch.Tensor,
        score_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, channels, height, width = feature_map.shape
        attention_weights = torch.softmax(score_map.flatten(2).squeeze(1), dim=1)
        flattened = feature_map.flatten(2).transpose(1, 2)
        pooled = torch.sum(
            flattened * attention_weights.unsqueeze(-1).to(feature_map.dtype),
            dim=1,
        )
        return pooled, attention_weights.reshape(batch_size, 1, height, width)

    def _part_value_map(
        self,
        fused_map: torch.Tensor,
        foreground_attention: torch.Tensor,
        foreground_map: torch.Tensor,
    ) -> torch.Tensor:
        return foreground_map

    def forward_features(
        self,
        x: torch.Tensor,
        cam_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        features = super().forward_features(x)
        fused_map = features["fpn_feature_map"]
        foreground_attention, used_cam = self._attention_from_inputs(
            fused_map,
            cam_mask,
        )
        score_map, foreground_map = self._score_foreground_map(
            fused_map,
            foreground_attention,
            used_cam,
        )
        part_value_map = self._part_value_map(
            fused_map,
            foreground_attention,
            foreground_map,
        )
        part_features, part_coords, part_scores, part_indices = self._select_part_candidates(
            part_value_map,
            score_map,
        )
        pool_weights = torch.softmax(
            self.part_pool_logits[: part_features.size(1)],
            dim=0,
        ).to(part_features.dtype)
        soft_pooled_parts, soft_part_weights = self._soft_attention_pool_parts(
            part_value_map,
            score_map,
        )
        warmup_active = self._use_soft_part_warmup()
        if warmup_active:
            pooled_parts = soft_pooled_parts
        else:
            pooled_parts = torch.sum(part_features * pool_weights.view(1, -1, 1), dim=1)

        part_logits = self.part_classifier(pooled_parts)
        logits = features["logits"] + self.part_logit_scale * part_logits
        features.update(
            {
                "logits": logits,
                "foreground_attention": foreground_attention,
                "cam_prior": foreground_attention if used_cam else None,
                "score_map": score_map,
                "selection_attention": score_map,
                "predicted_foreground": score_map,
                "part_feature_map": fused_map,
                "foreground_map": foreground_map,
                "part_value_map": part_value_map,
                "foreground_feature": pooled_parts,
                "part_features": part_features,
                "part_coords": part_coords,
                "part_scores": part_scores,
                "part_indices": part_indices,
                "part_pool_weights": pool_weights,
                "soft_part_weights": soft_part_weights,
                "part_warmup_active": torch.full_like(
                    part_logits[:, :1],
                    1.0 if warmup_active else 0.0,
                ),
                "attention_feature": pooled_parts,
                "part_logits": part_logits,
                "part_gate": torch.full_like(part_logits[:, :1], self.part_logit_scale),
            }
        )
        return features

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        cam_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        features = self.forward_features(x, cam_mask=cam_mask)
        if return_features:
            return features
        return features["logits"]


def build_swinv2_tiny_fpn_parts(
    num_classes: int = 200,
    pretrained: bool = True,
    image_size: int = 224,
    fpn_channels: int = 256,
    num_parts: int = 6,
    part_window_size: int = 3,
    part_warmup_epochs: int = 10,
) -> SwinV2TinyFPNParts:
    return SwinV2TinyFPNParts(
        num_classes=num_classes,
        pretrained=pretrained,
        image_size=image_size,
        fpn_channels=fpn_channels,
        num_parts=num_parts,
        part_window_size=part_window_size,
        part_warmup_epochs=part_warmup_epochs,
    )
