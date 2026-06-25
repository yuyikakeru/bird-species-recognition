from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .swinv2_fpn import ConvNormAct, SwinV2TinyFPN


class SwinV2TinyFPNParts(SwinV2TinyFPN):
    """Foreground-aware FPN with response-based candidate region sampling."""

    def __init__(
        self,
        num_classes: int = 200,
        pretrained: bool = True,
        image_size: int = 224,
        fpn_channels: int = 256,
        num_parts: int = 6,
        part_window_size: int = 3,
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

        hidden_channels = max(fpn_channels // 4, 32)
        self.num_parts = num_parts
        self.part_window_size = part_window_size
        self.foreground_score = nn.Sequential(
            ConvNormAct(fpn_channels, hidden_channels, kernel_size=1),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        self.part_score = nn.Sequential(
            ConvNormAct(fpn_channels, hidden_channels, kernel_size=1),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        self.part_mixer = nn.Sequential(
            nn.LayerNorm(fpn_channels * 3),
            nn.Linear(fpn_channels * 3, fpn_channels * 2),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(fpn_channels * 2, fpn_channels),
        )
        self.part_norm = nn.LayerNorm(fpn_channels)
        self.part_classifier = nn.Linear(fpn_channels, num_classes)
        self.part_gate = nn.Sequential(
            nn.LayerNorm(fpn_channels),
            nn.Linear(fpn_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, 1),
        )
        self.part_gate_bias = nn.Parameter(torch.tensor(-2.0))
        self._init_part_head()

    def _init_part_head(self) -> None:
        nn.init.trunc_normal_(self.part_classifier.weight, std=0.02)
        nn.init.zeros_(self.part_classifier.bias)
        nn.init.zeros_(self.part_gate[-1].weight)
        nn.init.zeros_(self.part_gate[-1].bias)

    def _topk_candidates(
        self,
        feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, channels, height, width = feature.shape
        padding = self.part_window_size // 2
        local_feature = F.avg_pool2d(
            feature,
            kernel_size=self.part_window_size,
            stride=1,
            padding=padding,
        )
        response_map = self.part_score(feature)
        local_max = F.max_pool2d(
            response_map,
            kernel_size=self.part_window_size,
            stride=1,
            padding=padding,
        )
        response_map = response_map.masked_fill(response_map < local_max, float("-inf"))
        response = response_map.flatten(2)
        candidate_scores, candidate_indices = torch.topk(
            response,
            k=min(self.num_parts, height * width),
            dim=-1,
        )
        candidate_indices = candidate_indices.squeeze(1)
        candidate_scores = candidate_scores.squeeze(1)

        y = candidate_indices.div(width, rounding_mode="floor").to(feature.dtype)
        x = (candidate_indices % width).to(feature.dtype)
        coords = torch.stack(
            [
                x * (2.0 / max(width - 1, 1)) - 1.0,
                y * (2.0 / max(height - 1, 1)) - 1.0,
            ],
            dim=-1,
        )

        flattened_feature = local_feature.flatten(2).transpose(1, 2)
        gather_indices = candidate_indices.unsqueeze(-1).expand(-1, -1, channels)
        candidate_features = torch.gather(flattened_feature, 1, gather_indices)
        return candidate_features, coords, candidate_scores

    def forward_features(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        features = super().forward_features(x)
        fused = features["fused"]

        foreground_attention = torch.sigmoid(self.foreground_score(fused))
        foreground_map = fused * (0.5 + foreground_attention)
        foreground_weights = foreground_attention.flatten(2)
        foreground_weights = foreground_weights / foreground_weights.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-6)
        foreground_feature = torch.sum(
            foreground_map.flatten(2) * foreground_weights,
            dim=-1,
        )

        part_features, part_coords, part_scores = self._topk_candidates(foreground_map)
        part_weights = torch.softmax(part_scores, dim=1).unsqueeze(-1)
        pooled_parts = torch.sum(part_features * part_weights, dim=1)
        max_parts = part_features.max(dim=1).values
        attention_feature = self.part_mixer(
            torch.cat(
                [foreground_feature, pooled_parts, max_parts],
                dim=1,
            )
        )
        attention_feature = self.part_norm(attention_feature)
        part_logits = self.part_classifier(attention_feature)
        part_gate = torch.sigmoid(
            self.part_gate(attention_feature) + self.part_gate_bias
        )
        logits = features["logits"] + part_gate * part_logits

        features.update(
            {
                "logits": logits,
                "foreground_attention": foreground_attention,
                "foreground_map": foreground_map,
                "foreground_feature": foreground_feature,
                "part_features": part_features,
                "part_coords": part_coords,
                "part_scores": part_scores,
                "attention_feature": attention_feature,
                "part_logits": part_logits,
                "part_gate": part_gate,
            }
        )
        return features

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        features = self.forward_features(x)
        if return_features:
            return features
        return features["logits"]


class SwinV2TinyFPNRelation(SwinV2TinyFPNParts):
    """Candidate-region relation transformer with low-rank bilinear pooling."""

    def __init__(
        self,
        num_classes: int = 200,
        pretrained: bool = True,
        image_size: int = 224,
        fpn_channels: int = 256,
        num_parts: int = 6,
        part_window_size: int = 3,
        relation_heads: int = 4,
        bilinear_dim: int = 256,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            pretrained=pretrained,
            image_size=image_size,
            fpn_channels=fpn_channels,
            num_parts=num_parts,
            part_window_size=part_window_size,
        )
        if relation_heads < 1 or fpn_channels % relation_heads != 0:
            raise ValueError(
                "relation_heads must divide fpn_channels, got "
                f"{relation_heads} and {fpn_channels}"
            )
        if bilinear_dim < 1:
            raise ValueError(f"bilinear_dim must be positive, got {bilinear_dim}")

        self.spatial_position = nn.Sequential(
            nn.Linear(2, fpn_channels),
            nn.GELU(),
            nn.Linear(fpn_channels, fpn_channels),
        )
        relation_layer = nn.TransformerEncoderLayer(
            d_model=fpn_channels,
            nhead=relation_heads,
            dim_feedforward=fpn_channels * 2,
            dropout=0.2,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.relation_encoder = nn.TransformerEncoder(relation_layer, num_layers=1)
        self.relation_norm = nn.LayerNorm(fpn_channels)
        self.bilinear_left = nn.Linear(fpn_channels, bilinear_dim)
        self.bilinear_right = nn.Linear(fpn_channels, bilinear_dim)
        self.bilinear_output = nn.Linear(bilinear_dim, fpn_channels)
        self.pairwise_mixer = nn.Sequential(
            nn.LayerNorm(fpn_channels * 2),
            nn.Linear(fpn_channels * 2, fpn_channels),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(fpn_channels, fpn_channels),
            nn.LayerNorm(fpn_channels),
        )
        self.relation_mixer = nn.Sequential(
            nn.LayerNorm(fpn_channels * 4),
            nn.Linear(fpn_channels * 4, fpn_channels * 2),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(fpn_channels * 2, fpn_channels),
            nn.LayerNorm(fpn_channels),
        )
        hidden_channels = max(fpn_channels // 4, 32)
        self.relation_classifier = nn.Linear(fpn_channels, num_classes)
        self.relation_gate = nn.Sequential(
            nn.LayerNorm(fpn_channels),
            nn.Linear(fpn_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, 1),
        )
        self.relation_gate_bias = nn.Parameter(torch.tensor(-2.0))
        self._init_relation_head()

    def _init_relation_head(self) -> None:
        nn.init.trunc_normal_(self.relation_classifier.weight, std=0.02)
        nn.init.zeros_(self.relation_classifier.bias)
        nn.init.zeros_(self.relation_gate[-1].weight)
        nn.init.zeros_(self.relation_gate[-1].bias)

    def forward_features(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        features = super().forward_features(x)
        foreground_token = features["foreground_feature"].unsqueeze(1)
        part_tokens = features["part_features"]
        tokens = torch.cat([foreground_token, part_tokens], dim=1)

        part_position = self.spatial_position(features["part_coords"])
        global_position = torch.zeros_like(part_position[:, :1])
        encoded = self.relation_encoder(
            tokens + torch.cat([global_position, part_position], dim=1),
        )
        relation_feature = self.relation_norm(encoded[:, 0])

        related_parts = encoded[:, 1:]
        left = self.bilinear_left(related_parts).mean(dim=1)
        right = self.bilinear_right(related_parts).mean(dim=1)
        bilinear_dtype = left.dtype
        bilinear_feature = left.float() * right.float()
        bilinear_feature = torch.sign(bilinear_feature) * torch.sqrt(
            bilinear_feature.abs() + 1e-6
        )
        bilinear_feature = F.normalize(bilinear_feature, dim=1, eps=1e-6)
        bilinear_feature = bilinear_feature.to(bilinear_dtype)
        bilinear_feature = self.bilinear_output(bilinear_feature)

        num_parts = related_parts.size(1)
        if num_parts > 1:
            pair_i, pair_j = torch.triu_indices(
                num_parts,
                num_parts,
                offset=1,
                device=related_parts.device,
            )
            left_parts = related_parts[:, pair_i]
            right_parts = related_parts[:, pair_j]
            pairwise_tokens = torch.cat(
                [
                    (left_parts - right_parts).abs(),
                    left_parts * right_parts,
                ],
                dim=-1,
            )
            pairwise_feature = self.pairwise_mixer(pairwise_tokens).mean(dim=1)
        else:
            pairwise_feature = torch.zeros_like(relation_feature)

        relation_output = self.relation_mixer(
            torch.cat(
                [
                    relation_feature,
                    bilinear_feature,
                    pairwise_feature,
                    features["attention_feature"],
                ],
                dim=1,
            )
        )
        relation_logits = self.relation_classifier(relation_output)
        relation_gate = torch.sigmoid(
            self.relation_gate(relation_output) + self.relation_gate_bias
        )
        logits = features["logits"] + relation_gate * relation_logits
        features.update(
            {
                "logits": logits,
                "relation_tokens": encoded,
                "relation_feature": relation_feature,
                "bilinear_feature": bilinear_feature,
                "pairwise_feature": pairwise_feature,
                "relation_output": relation_output,
                "relation_logits": relation_logits,
                "relation_gate": relation_gate,
            }
        )
        return features


def build_swinv2_tiny_fpn_parts(
    num_classes: int = 200,
    pretrained: bool = True,
    image_size: int = 224,
    fpn_channels: int = 256,
    num_parts: int = 6,
    part_window_size: int = 3,
) -> SwinV2TinyFPNParts:
    return SwinV2TinyFPNParts(
        num_classes=num_classes,
        pretrained=pretrained,
        image_size=image_size,
        fpn_channels=fpn_channels,
        num_parts=num_parts,
        part_window_size=part_window_size,
    )


def build_swinv2_tiny_fpn_relation(
    num_classes: int = 200,
    pretrained: bool = True,
    image_size: int = 224,
    fpn_channels: int = 256,
    num_parts: int = 6,
    part_window_size: int = 3,
    relation_heads: int = 4,
    bilinear_dim: int = 256,
) -> SwinV2TinyFPNRelation:
    return SwinV2TinyFPNRelation(
        num_classes=num_classes,
        pretrained=pretrained,
        image_size=image_size,
        fpn_channels=fpn_channels,
        num_parts=num_parts,
        part_window_size=part_window_size,
        relation_heads=relation_heads,
        bilinear_dim=bilinear_dim,
    )
