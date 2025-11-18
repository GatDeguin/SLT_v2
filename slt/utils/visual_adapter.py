"""Multimodal adapter utilities shared by training and inference."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

KEYPOINT_CHANNELS = 3


@dataclass
class AdapterConfig:
    hidden_size: int = 1024
    keypoint_hidden: int = 512
    fusion_layers: int = 4
    fusion_heads: int = 8
    dropout: float = 0.1
    keypoint_layers: int = 4
    keypoint_heads: int = 8


class _TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, *, num_layers: int, nhead: int, dropout: float) -> None:
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(self.encoder(self.dropout(x)))


class KeypointEncoder(nn.Module):
    def __init__(self, config: AdapterConfig, num_points: int) -> None:
        super().__init__()
        input_dim = num_points * KEYPOINT_CHANNELS
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, config.keypoint_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.keypoint_hidden, config.keypoint_hidden),
        )
        self.temporal = _TransformerEncoder(
            config.keypoint_hidden,
            num_layers=config.keypoint_layers,
            nhead=config.keypoint_heads,
            dropout=config.dropout,
        )

    def forward(self, kp_btnc: torch.Tensor) -> torch.Tensor:
        if kp_btnc is None:
            raise ValueError("KeypointEncoder received None input")
        B, T, N, C = kp_btnc.shape
        x = kp_btnc.reshape(B, T, N * C)
        x = self.proj(x)
        return self.temporal(x)


class FusionAdapter(nn.Module):
    """Keypoint-only adapter mirroring the inference helper implementation."""

    def __init__(self, config: AdapterConfig, num_points: int) -> None:
        super().__init__()
        self.kp_enc = KeypointEncoder(config, num_points)
        self.kp_proj = nn.Linear(config.keypoint_hidden, config.hidden_size)
        self.keypoint_bias = nn.Parameter(torch.zeros(config.hidden_size))
        self.fusion = _TransformerEncoder(
            config.hidden_size,
            num_layers=config.fusion_layers,
            nhead=config.fusion_heads,
            dropout=config.dropout,
        )
        self.out_norm = nn.LayerNorm(config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self,
        keypoints: torch.Tensor,
        frames: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if keypoints is None:
            raise ValueError("FusionAdapter expects keypoints tensor")
        kp = self.kp_enc(keypoints)  # [B,T,H_k]
        kp = self.kp_proj(kp)  # [B,T,D]
        kp = kp + self.keypoint_bias.view(1, 1, -1)
        fused = self.fusion(kp)
        pooled = fused.mean(dim=1)
        pooled = self.out_norm(pooled)
        return self.out_proj(pooled)


def _import_visual_backbones() -> Tuple[type, ...]:
    try:  # pragma: no cover - optional heavy dependency
        from models.visual_backbone import (
            SpaMoFusion,
            SpaMoFusionConfig,
            ViTBackbone,
            VideoMAEBackbone,
        )
    except Exception as exc:  # pragma: no cover - surfacing clearer guidance
        raise ModuleNotFoundError(
            "VisualFusionAdapter requires the optional visual backbones."
            " Install the project with timm/transformers[video] extras to enable it."
        ) from exc

    return SpaMoFusion, SpaMoFusionConfig, ViTBackbone, VideoMAEBackbone


class VisualFusionAdapter(nn.Module):
    """Fuse keypoints with spatial/motion cues before bridging into SONAR."""

    def __init__(
        self,
        config: AdapterConfig,
        num_points: int,
        *,
        vit_name: str = "vit_base_patch16_224",
        vit_checkpoint: Optional[Union[str, Path]] = None,
        freeze_vit: bool = False,
        videomae_name: str = "MCG-NJU/videomae-base",
        videomae_checkpoint: Optional[Union[str, Path]] = None,
        freeze_videomae: bool = False,
        freeze_keypoint: bool = False,
        fusion_config: Optional[SpaMoFusionConfig] = None,
    ) -> None:
        super().__init__()
        SpaMoFusion, SpaMoFusionConfig, ViTBackbone, VideoMAEBackbone = _import_visual_backbones()

        fusion_config = fusion_config or SpaMoFusionConfig(
            hidden_size=config.hidden_size,
            num_layers=config.fusion_layers,
            num_heads=config.fusion_heads,
            dropout=config.dropout,
        )
        self.kp_enc = KeypointEncoder(config, num_points)
        if freeze_keypoint:
            for param in self.kp_enc.parameters():
                param.requires_grad = False
        vit_ckpt = Path(vit_checkpoint) if vit_checkpoint is not None else None
        vmae_ckpt = Path(videomae_checkpoint) if videomae_checkpoint is not None else None
        self.vit = ViTBackbone(
            vit_name,
            pretrained=True,
            checkpoint_path=vit_ckpt,
            freeze=freeze_vit,
        )
        self.videomae = VideoMAEBackbone(
            videomae_name,
            pretrained=True,
            checkpoint_path=vmae_ckpt,
            freeze=freeze_videomae,
        )
        self.fusion = SpaMoFusion(
            key_dim=config.keypoint_hidden,
            spatial_dim=self.vit.output_dim,
            motion_dim=self.videomae.hidden_size,
            config=fusion_config,
        )
        self.out_norm = nn.LayerNorm(config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self,
        keypoints: Optional[torch.Tensor],
        frames: Optional[torch.Tensor],
    ) -> torch.Tensor:
        kp = self.kp_enc(keypoints) if keypoints is not None else None
        spatial = self.vit(frames) if frames is not None else None
        motion = self.videomae(frames) if frames is not None else None
        fused = self.fusion(kp, spatial, motion)
        pooled = fused.mean(dim=1)
        pooled = self.out_norm(pooled)
        return self.out_proj(pooled)
