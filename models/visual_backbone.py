"""Visual backbones used by SONAR-SLT adapters.

This module bundles light-weight wrappers around ``timm`` and ``transformers``
implementations so the rest of the code base can operate on tensors shaped
``(batch, time, channels, height, width)``. All classes expose a ``forward``
method that returns temporally-aligned features suitable for fusion with the
sign keypoint stream.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

try:  # pragma: no cover - optional dependency
    import timm
except Exception:  # pragma: no cover - optional dependency
    timm = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from transformers import VideoMAEConfig, VideoMAEModel
except Exception:  # pragma: no cover - optional dependency
    VideoMAEModel = None  # type: ignore[assignment]
    VideoMAEConfig = None  # type: ignore[assignment]


class ViTBackbone(nn.Module):
    """Wrapper around a ViT image encoder from ``timm``.

    Parameters
    ----------
    model_name:
        Name of the timm ViT architecture to instantiate.
    pretrained:
        Whether to load ImageNet pre-trained weights when constructing the
        backbone.
    checkpoint_path:
        Optional path to a checkpoint with fine-tuned weights. When provided,
        it is loaded after the base model is created.
    freeze:
        If ``True`` the backbone parameters are frozen.
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        *,
        pretrained: bool = True,
        checkpoint_path: Optional[Path] = None,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        if timm is None:
            raise ImportError("timm is required to use ViTBackbone")
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        self.output_dim = getattr(self.model, "num_features", None) or getattr(
            self.model, "embed_dim", 768
        )
        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            self.model.load_state_dict(state_dict, strict=False)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.eval() if freeze else self.model.train()

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5:
            raise ValueError("ViTBackbone expects tensors shaped (B, T, C, H, W)")
        batch, time, channels, height, width = frames.shape
        if channels != 3:
            raise ValueError("ViTBackbone expects RGB inputs with 3 channels")
        flat = frames.view(batch * time, channels, height, width)
        features = self.model(flat)  # (B*T, D)
        features = features.view(batch, time, -1)
        return features


class VideoMAEBackbone(nn.Module):
    """VideoMAE wrapper that exposes per-frame features."""

    def __init__(
        self,
        model_name: str = "MCG-NJU/videomae-base",
        *,
        pretrained: bool = True,
        checkpoint_path: Optional[Path] = None,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        if VideoMAEModel is None:
            raise ImportError("transformers[video] is required to use VideoMAEBackbone")
        if pretrained:
            self.model = VideoMAEModel.from_pretrained(model_name)
        else:
            if VideoMAEConfig is None:
                raise ImportError("transformers[video] is required to build VideoMAE models")
            config = VideoMAEConfig.from_pretrained(model_name)
            self.model = VideoMAEModel(config)
        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            self.model.load_state_dict(state_dict, strict=False)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.eval() if freeze else self.model.train()
        self.hidden_size = int(self.model.config.hidden_size)
        self.patch_size = int(self.model.config.patch_size)
        self.tubelet_size = int(getattr(self.model.config, "tubelet_size", 2))

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5:
            raise ValueError("VideoMAEBackbone expects tensors shaped (B, T, C, H, W)")
        batch, time, channels, height, width = frames.shape
        if channels != 3:
            raise ValueError("VideoMAEBackbone expects RGB inputs with 3 channels")
        pixel_values = frames.contiguous()
        outputs = self.model(pixel_values=pixel_values, output_hidden_states=False)
        hidden = outputs.last_hidden_state  # (B, tokens, hidden)
        use_cls_token = getattr(self.model.config, "use_cls_token", None)
        if use_cls_token is None:
            use_cls_token = hasattr(self.model, "cls_token")
        if use_cls_token:
            hidden = hidden[:, 1:, :]
        patches_per_frame = (height // self.patch_size) * (width // self.patch_size)
        tokens_per_frame = max(patches_per_frame, 1)
        if hidden.size(1) % tokens_per_frame != 0:
            raise ValueError(
                "VideoMAE token count %d not divisible by patches per frame %d"
                % (hidden.size(1), tokens_per_frame)
            )
        temporal_tokens = hidden.size(1) // tokens_per_frame
        hidden = hidden.view(batch, temporal_tokens, tokens_per_frame, self.hidden_size)
        pooled = hidden.mean(dim=2)
        if temporal_tokens == time:
            return pooled
        repeated = pooled.repeat_interleave(self.tubelet_size, dim=1)
        if repeated.size(1) < time:
            pad = time - repeated.size(1)
            last = repeated[:, -1:, :].expand(batch, pad, self.hidden_size)
            repeated = torch.cat([repeated, last], dim=1)
        return repeated[:, :time, :]


@dataclass
class SpaMoFusionConfig:
    hidden_size: int = 1024
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1


class SpaMoFusion(nn.Module):
    """Fuse spatial (ViT) and motion (VideoMAE) descriptors with keypoints."""

    def __init__(
        self,
        key_dim: int,
        spatial_dim: int,
        motion_dim: int,
        config: SpaMoFusionConfig,
    ) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.key_proj = nn.Identity() if key_dim == hidden else nn.Linear(key_dim, hidden)
        self.spatial_proj = (
            nn.Identity() if spatial_dim == hidden else nn.Linear(spatial_dim, hidden)
        )
        self.motion_proj = (
            nn.Identity() if motion_dim == hidden else nn.Linear(motion_dim, hidden)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.num_heads,
            dim_feedforward=hidden * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(hidden)
        self.modality_embeddings = nn.Parameter(torch.randn(3, hidden))

    def forward(
        self,
        keypoints: Optional[torch.Tensor] = None,
        spatial: Optional[torch.Tensor] = None,
        motion: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        streams = []
        if keypoints is not None:
            kp = self.key_proj(keypoints)
            streams.append(kp + self.modality_embeddings[0].view(1, 1, -1))
        if spatial is not None:
            sp = self.spatial_proj(spatial)
            streams.append(sp + self.modality_embeddings[1].view(1, 1, -1))
        if motion is not None:
            mo = self.motion_proj(motion)
            streams.append(mo + self.modality_embeddings[2].view(1, 1, -1))
        if not streams:
            raise ValueError("SpaMoFusion requires at least one modality input")
        tokens = torch.cat(streams, dim=1)
        tokens = self.dropout(tokens)
        fused = self.encoder(tokens)
        return self.layer_norm(fused)
