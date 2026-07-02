from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _load_timm_convnextv2_tiny(
    *,
    pretrained: bool,
    num_classes: int = 0,
    features_only: bool = False,
) -> nn.Module:
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError(
            "ConvNeXtV2-Tiny requires timm. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    try:
        kwargs = {
            "pretrained": pretrained,
            "num_classes": num_classes,
            "features_only": features_only,
        }
        if features_only:
            kwargs["out_indices"] = (1, 2, 3)
        return timm.create_model("convnextv2_tiny", **kwargs)
    except Exception as exc:
        if pretrained:
            raise RuntimeError(
                "Failed to load ImageNet pretrained ConvNeXtV2-Tiny weights. "
                "Check network access or rerun with --no-pretrained."
            ) from exc
        raise


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        groups: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape,
            dtype=x.dtype,
            device=x.device,
        )
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class DiscriminativeContextAttention(nn.Module):
    """Residual branch that pools discriminative context from the final feature map."""

    def __init__(self, channels: int, hidden_ratio: int = 4) -> None:
        super().__init__()
        hidden_channels = max(channels // hidden_ratio, 64)
        self.local_context = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial_score = nn.Conv2d(channels, 1, kernel_size=1)
        self.out_norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.local_context(x) * self.channel_gate(x)
        score_map = self.spatial_score(context)
        weights = torch.softmax(score_map.flatten(2), dim=-1)
        tokens = (x + context).flatten(2).transpose(1, 2)
        pooled = torch.sum(tokens * weights.transpose(1, 2), dim=1)
        return self.out_norm(pooled), score_map


class RegionTokenPooling(nn.Module):
    """Pool several DCA-guided discriminative region tokens from stage-3 features."""

    def __init__(
        self,
        stage3_channels: int,
        stage4_channels: int,
        dca_channels: int,
        region_channels: int = 256,
        num_regions: int = 4,
    ) -> None:
        super().__init__()
        if region_channels < 1:
            raise ValueError(f"region_channels must be positive, got {region_channels}")
        if num_regions < 1:
            raise ValueError(f"num_regions must be positive, got {num_regions}")

        self.num_regions = num_regions
        self.stage3_lateral = ConvNormAct(stage3_channels, region_channels, kernel_size=1)
        self.stage4_lateral = ConvNormAct(stage4_channels, region_channels, kernel_size=1)
        self.region_smooth = nn.Sequential(
            ConvNormAct(region_channels, region_channels, kernel_size=3, padding=1),
            ConvNormAct(region_channels, region_channels, kernel_size=3, padding=1),
        )
        self.region_score = nn.Conv2d(region_channels, num_regions, kernel_size=1)
        self.region_embed = nn.Parameter(torch.zeros(1, num_regions, region_channels))
        self.global_proj = nn.Linear(dca_channels, region_channels)
        self.token_dropout = nn.Dropout(p=0.1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=region_channels,
            nhead=4,
            dim_feedforward=region_channels * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.token_mixer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.out_norm = nn.LayerNorm(region_channels)
        self.output_dim = region_channels
        nn.init.trunc_normal_(self.region_embed, std=0.02)

    @staticmethod
    def _standardize_score(score: torch.Tensor) -> torch.Tensor:
        flat_score = score.flatten(2)
        mean = flat_score.mean(dim=2, keepdim=True).unsqueeze(-1)
        std = flat_score.std(dim=2, keepdim=True, unbiased=False).unsqueeze(-1)
        return ((score - mean) / std.clamp_min(1e-6)).clamp(-5.0, 5.0)

    def forward(
        self,
        stage3: torch.Tensor,
        stage4: torch.Tensor,
        dca_feature: torch.Tensor,
        attention_map: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_size = stage3.shape[-2:]
        stage3_map = self.stage3_lateral(stage3)
        stage4_map = F.interpolate(
            self.stage4_lateral(stage4),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        region_map = self.region_smooth(stage3_map + stage4_map)
        batch_size, channels, height, width = region_map.shape
        region_logits = self.region_score(region_map)
        if attention_map is not None:
            attention_score = F.interpolate(
                attention_map,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            region_logits = region_logits + 0.5 * self._standardize_score(attention_score.float()).to(region_logits.dtype)

        region_weights = torch.softmax(region_logits.flatten(2), dim=-1)
        region_tokens = torch.bmm(
            region_weights,
            region_map.flatten(2).transpose(1, 2),
        )
        region_tokens = region_tokens + self.region_embed.to(region_tokens.dtype)
        global_token = self.global_proj(dca_feature).unsqueeze(1)
        tokens = torch.cat([global_token, self.token_dropout(region_tokens)], dim=1)
        mixed_tokens = self.token_mixer(tokens)
        region_feature = self.out_norm(mixed_tokens[:, 0] + mixed_tokens[:, 1:].mean(dim=1))
        return region_feature, region_map, region_weights.reshape(batch_size, self.num_regions, height, width)


def region_attention_diversity_loss(region_attention: torch.Tensor) -> torch.Tensor:
    batch_size, num_regions, height, width = region_attention.shape
    attention = region_attention.flatten(2)
    attention = F.normalize(attention, p=2, dim=2)
    similarity = torch.bmm(attention, attention.transpose(1, 2))
    eye = torch.eye(num_regions, device=region_attention.device, dtype=torch.bool)
    eye = eye.unsqueeze(0).expand(batch_size, -1, -1)
    off_diagonal = similarity.masked_select(~eye)
    return off_diagonal.pow(2).mean()


class ConvNeXtV2TinyBase(nn.Module):
    """ImageNet pretrained ConvNeXtV2-Tiny classifier for CUB_200_2011."""

    def __init__(
        self,
        num_classes: int = 200,
        pretrained: bool = True,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        del image_size
        self.backbone = _load_timm_convnextv2_tiny(
            pretrained=pretrained,
            num_classes=num_classes,
            features_only=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ConvNeXtV2TinyDCA(nn.Module):
    """ConvNeXtV2-Tiny baseline plus a DCA residual classification branch."""

    def __init__(
        self,
        num_classes: int = 200,
        pretrained: bool = True,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        del image_size
        self.backbone = _load_timm_convnextv2_tiny(
            pretrained=pretrained,
            features_only=True,
        )
        final_channels = self.backbone.feature_info.channels()[-1]
        self.baseline_norm = nn.LayerNorm(final_channels)
        self.baseline_classifier = nn.Linear(final_channels, num_classes)
        self.dca = DiscriminativeContextAttention(final_channels)
        self.dca_classifier = nn.Linear(final_channels, num_classes)
        self.baseline_logit_scale = 1.0
        self.dca_logit_scale = 1.0
        self._init_residual_head()

    def _init_residual_head(self) -> None:
        nn.init.zeros_(self.dca_classifier.weight)
        nn.init.zeros_(self.dca_classifier.bias)

    def get_optimizer_param_groups(
        self,
        base_lr: float,
        weight_decay: float,
        head_lr_mult: float = 5.0,
    ) -> list[dict[str, object]]:
        return _build_param_groups(self, base_lr, weight_decay, head_lr_mult)

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        stage2, stage3, stage4 = self.backbone(x)
        baseline_feature = self.baseline_norm(stage4.mean(dim=(2, 3)))
        baseline_logits = self.baseline_classifier(baseline_feature)
        dca_feature, dca_score_map = self.dca(stage4)
        dca_logits = self.dca_classifier(dca_feature)
        logits = self.baseline_logit_scale * baseline_logits + self.dca_logit_scale * dca_logits
        return {
            "logits": logits,
            "baseline_logits": baseline_logits,
            "baseline_feature": baseline_feature,
            "baseline_logit_scale": torch.full_like(baseline_logits[:, :1], self.baseline_logit_scale),
            "stage2_features": stage2,
            "stage3_features": stage3,
            "stage4_features": stage4,
            "dca_feature": dca_feature,
            "dca_score_map": dca_score_map,
            "dca_logits": dca_logits,
            "dca_logit_scale": torch.full_like(dca_logits[:, :1], self.dca_logit_scale),
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


class ConvNeXtV2TinyDCARegion(nn.Module):
    """ConvNeXtV2-Tiny with DCA and discriminative region residual branches."""

    def __init__(
        self,
        num_classes: int = 200,
        pretrained: bool = True,
        image_size: int = 224,
        fpn_channels: int = 256,
    ) -> None:
        super().__init__()
        del image_size
        self.backbone = _load_timm_convnextv2_tiny(
            pretrained=pretrained,
            features_only=True,
        )
        _, stage3_channels, stage4_channels = self.backbone.feature_info.channels()
        self.baseline_norm = nn.LayerNorm(stage4_channels)
        self.baseline_classifier = nn.Linear(stage4_channels, num_classes)
        self.dca = DiscriminativeContextAttention(stage4_channels)
        self.dca_classifier = nn.Linear(stage4_channels, num_classes)
        self.region_pool = RegionTokenPooling(
            stage3_channels=stage3_channels,
            stage4_channels=stage4_channels,
            dca_channels=stage4_channels,
            region_channels=fpn_channels,
            num_regions=4,
        )
        self.region_classifier = nn.Linear(self.region_pool.output_dim, num_classes)
        self.region_dropout = nn.Dropout(p=0.1)
        self.region_drop_path = DropPath(drop_prob=0.1)
        self.region_diversity_weight = 0.05
        self.baseline_logit_scale = 1.0
        self.dca_logit_scale = 1.0
        self.region_logit_scale = 1.0
        self._init_residual_heads()

    def _init_residual_heads(self) -> None:
        nn.init.zeros_(self.dca_classifier.weight)
        nn.init.zeros_(self.dca_classifier.bias)
        nn.init.zeros_(self.region_classifier.weight)
        nn.init.zeros_(self.region_classifier.bias)

    def get_optimizer_param_groups(
        self,
        base_lr: float,
        weight_decay: float,
        head_lr_mult: float = 5.0,
    ) -> list[dict[str, object]]:
        return _build_param_groups(self, base_lr, weight_decay, head_lr_mult)

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        stage2, stage3, stage4 = self.backbone(x)
        baseline_feature = self.baseline_norm(stage4.mean(dim=(2, 3)))
        baseline_logits = self.baseline_classifier(baseline_feature)
        dca_feature, dca_score_map = self.dca(stage4)
        dca_logits = self.dca_classifier(dca_feature)
        region_feature, region_map, region_attention = self.region_pool(
            stage3,
            stage4,
            dca_feature=dca_feature,
            attention_map=dca_score_map,
        )
        region_feature = self.region_drop_path(self.region_dropout(region_feature))
        region_logits = self.region_classifier(region_feature)
        region_diversity_loss = region_attention_diversity_loss(region_attention)
        logits = (
            self.baseline_logit_scale * baseline_logits
            + self.dca_logit_scale * dca_logits
            + self.region_logit_scale * region_logits
        )
        return {
            "logits": logits,
            "baseline_logits": baseline_logits,
            "baseline_feature": baseline_feature,
            "baseline_logit_scale": torch.full_like(baseline_logits[:, :1], self.baseline_logit_scale),
            "stage2_features": stage2,
            "stage3_features": stage3,
            "stage4_features": stage4,
            "dca_feature": dca_feature,
            "dca_score_map": dca_score_map,
            "dca_logits": dca_logits,
            "dca_logit_scale": torch.full_like(dca_logits[:, :1], self.dca_logit_scale),
            "region_map": region_map,
            "region_attention": region_attention,
            "region_feature": region_feature,
            "region_logits": region_logits,
            "region_diversity_loss": region_diversity_loss,
            "aux_loss": self.region_diversity_weight * region_diversity_loss,
            "region_logit_scale": torch.full_like(region_logits[:, :1], self.region_logit_scale),
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


def _build_param_groups(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    head_lr_mult: float,
) -> list[dict[str, object]]:
    groups: dict[tuple[float, float], dict[str, object]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_head = not name.startswith("backbone.")
        lr = base_lr * (head_lr_mult if is_head else 1.0)
        decay = 0.0 if _should_skip_weight_decay(name, param) else weight_decay
        key = (lr, decay)
        groups.setdefault(key, {"params": [], "lr": lr, "weight_decay": decay})
        groups[key]["params"].append(param)
    return list(groups.values())


def _should_skip_weight_decay(name: str, param: torch.nn.Parameter) -> bool:
    lowered = name.lower()
    return (
        param.ndim <= 1
        or lowered.endswith(".bias")
        or "norm" in lowered
        or "gamma" in lowered
        or "grn" in lowered
    )


def build_convnextv2_tiny(
    num_classes: int = 200,
    pretrained: bool = True,
    image_size: int = 224,
) -> ConvNeXtV2TinyBase:
    return ConvNeXtV2TinyBase(
        num_classes=num_classes,
        pretrained=pretrained,
        image_size=image_size,
    )


def build_convnextv2_tiny_dca(
    num_classes: int = 200,
    pretrained: bool = True,
    image_size: int = 224,
) -> ConvNeXtV2TinyDCA:
    return ConvNeXtV2TinyDCA(
        num_classes=num_classes,
        pretrained=pretrained,
        image_size=image_size,
    )


def build_convnextv2_tiny_dca_region(
    num_classes: int = 200,
    pretrained: bool = True,
    image_size: int = 224,
    fpn_channels: int = 256,
) -> ConvNeXtV2TinyDCARegion:
    return ConvNeXtV2TinyDCARegion(
        num_classes=num_classes,
        pretrained=pretrained,
        image_size=image_size,
        fpn_channels=fpn_channels,
    )
