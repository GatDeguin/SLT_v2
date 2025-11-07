"""Inference script for SONAR-SLT using single signer videos and keypoints.

This module implements a self-contained pipeline that mirrors the data
pre-processing and multimodal fusion strategy described in the SONAR-SLT paper.
The script expects MediaPipe holistic keypoints exported with
``extract_rois_v2.py`` and raw videos. Keypoints and frame crops are projected
into a shared latent space and subsequently aligned with the SONAR text decoder
(`M2M100ForConditionalGeneration`).

The implementation focuses on reproducibility and clarity:

* Keypoints are normalised frame-wise to remove translation and scale
  variations, retaining visibility/confidence channels.
* Video frames are sampled uniformly, resized to 224x224 and embedded with a
  lightweight ResNet18 backbone. The weights default to the ImageNet initialisa-
  tion and can be overridden through a checkpoint.
* Both streams go through temporal Transformer encoders followed by a fusion
  Transformer. The fused representation is projected onto the 1024-dim SONAR
  space and fed directly into the decoder as a single "pseudo token" sequence.

The adapter weights that map sign representations to SONAR space are expected
in PyTorch ``state_dict`` format. They can be trained following the training
recipe in the paper and loaded with ``--adapter-checkpoint``.

Example usage::

    python sonar_slt_inference.py \
        --videos-dir data/single_signer/videos \
        --keypoints-dir data/single_signer/kp \
        --adapter-checkpoint checkpoints/sonar_slt_adapter.pt \
        --output translations.jsonl

The output is a JSONL file where each line contains the sample identifier, the
chosen language tags and the generated translation.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    BaseModelOutput,
    M2M100ForConditionalGeneration,
)

try:  # pragma: no cover - optional dependency
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from torchvision import models
    from torchvision.models import ResNet18_Weights
except Exception:  # pragma: no cover - optional dependency
    models = None  # type: ignore[assignment]
    ResNet18_Weights = None  # type: ignore[assignment]

LOGGER = logging.getLogger("sonar_slt_inference")

VIDEO_EXTENSIONS = (".mp4", ".mkv", ".mov", ".avi", ".webm")
KEYPOINT_TOTAL_LANDMARKS = 33 + 468 + 2 * 21  # pose + face + 2 hands
KEYPOINT_CHANNELS = 3  # (x, y, confidence)
DEFAULT_FRAME_SIZE = 224
DEFAULT_NUM_FRAMES = 96
DEFAULT_KEYPOINT_FRAMES = 192
DEFAULT_TRANSFORMER_HEADS = 8
DEFAULT_TRANSFORMER_LAYERS = 4
DEFAULT_TRANSFORMER_DROPOUT = 0.1


@dataclass
class GenerationConfig:
    """Configuration parameters for the translation decoder."""

    target_lang: str = "spa_Latn"
    max_new_tokens: int = 64
    num_beams: int = 4
    temperature: float = 1.0
    repetition_penalty: float = 1.1


@dataclass
class AdapterConfig:
    """Hyper-parameters for the multimodal adapter."""

    hidden_size: int = 1024
    video_hidden: int = 512
    keypoint_hidden: int = 512
    fusion_layers: int = DEFAULT_TRANSFORMER_LAYERS
    fusion_heads: int = DEFAULT_TRANSFORMER_HEADS
    dropout: float = DEFAULT_TRANSFORMER_DROPOUT
    keypoint_layers: int = DEFAULT_TRANSFORMER_LAYERS
    keypoint_heads: int = DEFAULT_TRANSFORMER_HEADS
    video_layers: int = DEFAULT_TRANSFORMER_LAYERS
    video_heads: int = DEFAULT_TRANSFORMER_HEADS


def _sinusoidal_positional_encoding(length: int, dim: int, device: torch.device) -> torch.Tensor:
    """Generate sinusoidal positional encodings."""

    position = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32, device=device)
        * (-math.log(10000.0) / dim)
    )
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class _TransformerEncoder(nn.Module):
    """Wrapper around ``nn.TransformerEncoder`` with sinusoidal embeddings."""

    def __init__(
        self,
        d_model: int,
        *,
        num_layers: int,
        nhead: int,
        dropout: float,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError("Transformer inputs must have shape (batch, time, dim)")
        positions = _sinusoidal_positional_encoding(
            inputs.size(1), inputs.size(2), inputs.device
        )
        hidden = inputs + positions.unsqueeze(0)
        hidden = self.dropout(hidden)
        hidden = self.encoder(hidden)
        return self.layer_norm(hidden)


class KeypointEncoder(nn.Module):
    """Temporal encoder for MediaPipe holistic keypoints."""

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__()
        input_dim = KEYPOINT_TOTAL_LANDMARKS * KEYPOINT_CHANNELS
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

    def forward(self, keypoints: torch.Tensor) -> torch.Tensor:
        """Encode keypoints into temporal embeddings.

        Args:
            keypoints: Tensor shaped ``(batch, frames, landmarks, channels)``.
        """

        if keypoints.ndim != 4:
            raise ValueError(
                "Keypoints tensor must be 4-D (batch, time, landmarks, channels)."
            )
        batch, frames, landmarks, channels = keypoints.shape
        if landmarks != KEYPOINT_TOTAL_LANDMARKS or channels != KEYPOINT_CHANNELS:
            raise ValueError(
                "Unexpected keypoint shape: expected (%d, %d) landmarks/channels, got %s"
                % (KEYPOINT_TOTAL_LANDMARKS, KEYPOINT_CHANNELS, keypoints.shape)
            )
        flattened = keypoints.view(batch, frames, -1)
        embedded = self.proj(flattened)
        return self.temporal(embedded)


class _IdentityBackbone(nn.Module):
    """Fallback backbone when torchvision is not available."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(inputs)


class VideoEncoder(nn.Module):
    """ResNet-based temporal encoder for RGB frames."""

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__()
        hidden = config.video_hidden
        if models is None or ResNet18_Weights is None:
            LOGGER.warning(
                "torchvision is not available; falling back to a linear frame encoder"
            )
            self._use_torchvision = False
            self.backbone = _IdentityBackbone(DEFAULT_FRAME_SIZE * DEFAULT_FRAME_SIZE * 3, hidden)
        else:  # pragma: no cover - requires torchvision
            weights = ResNet18_Weights.IMAGENET1K_V1
            backbone = models.resnet18(weights=weights)
            modules = list(backbone.children())[:-1]
            self.backbone = nn.Sequential(*modules)
            self.frame_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.frame_norm = nn.LayerNorm(512)
            self.frame_proj = nn.Linear(512, hidden)
            self._use_torchvision = True
            mean = torch.tensor(weights.meta["mean"], dtype=torch.float32)
            std = torch.tensor(weights.meta["std"], dtype=torch.float32)
            self.register_buffer("norm_mean", mean.view(1, 3, 1, 1), persistent=False)
            self.register_buffer("norm_std", std.view(1, 3, 1, 1), persistent=False)
        self.temporal = _TransformerEncoder(
            hidden,
            num_layers=config.video_layers,
            nhead=config.video_heads,
            dropout=config.dropout,
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode a batch of frames.

        Args:
            frames: Tensor shaped ``(batch, time, 3, H, W)`` in RGB and [0, 1] range.
        """

        if frames.ndim != 5:
            raise ValueError("Frames tensor must be 5-D (batch, time, channels, H, W)")
        batch, time, channels, height, width = frames.shape
        if channels != 3:
            raise ValueError("Expected 3-channel RGB frames, got %d" % channels)

        if not getattr(self, "_use_torchvision", False):
            flat = frames.view(batch, time, -1)
            embedded = self.backbone(flat)
            return self.temporal(embedded)

        frames = frames.view(batch * time, channels, height, width)
        frames = (frames - self.norm_mean) / self.norm_std
        features = self.backbone(frames)
        features = self.frame_pool(features).view(batch, time, -1)
        features = self.frame_norm(features)
        features = self.frame_proj(features)
        return self.temporal(features)


class FusionAdapter(nn.Module):
    """Fuse keypoint and video embeddings and project to SONAR space."""

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__()
        self.keypoint_encoder = KeypointEncoder(config)
        self.video_encoder = VideoEncoder(config)
        self.fusion = _TransformerEncoder(
            config.hidden_size,
            num_layers=config.fusion_layers,
            nhead=config.fusion_heads,
            dropout=config.dropout,
        )
        self.key_proj = nn.Linear(config.keypoint_hidden, config.hidden_size)
        self.video_proj = nn.Linear(config.video_hidden, config.hidden_size)
        self.output_norm = nn.LayerNorm(config.hidden_size)
        self.output_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self,
        keypoints: Optional[torch.Tensor],
        frames: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if keypoints is None and frames is None:
            raise ValueError("At least one modality (keypoints or frames) must be provided")

        streams: List[torch.Tensor] = []
        if keypoints is not None:
            key_emb = self.keypoint_encoder(keypoints)
            streams.append(self.key_proj(key_emb))
        if frames is not None:
            vid_emb = self.video_encoder(frames)
            streams.append(self.video_proj(vid_emb))

        fused = torch.cat(streams, dim=1)
        fused = self.fusion(fused)
        pooled = fused.mean(dim=1)
        pooled = self.output_norm(pooled)
        return self.output_proj(pooled)


def _load_keypoints(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        with np.load(path) as data:
            if "keypoints" in data:
                arr = data["keypoints"]
            elif "frames" in data:
                arr = data["frames"]
            else:
                raise KeyError(f"{path} does not contain 'keypoints' or 'frames'")
    elif path.suffix == ".npy":
        arr = np.load(path)
    else:
        raise ValueError(f"Unsupported keypoint format: {path.suffix}")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        return arr
    if arr.ndim == 2:
        length = arr.shape[0]
        return arr.reshape(length, KEYPOINT_TOTAL_LANDMARKS, KEYPOINT_CHANNELS)
    raise ValueError(f"Unexpected keypoint tensor shape: {arr.shape}")


def _normalise_keypoints(keypoints: np.ndarray) -> np.ndarray:
    coords = keypoints[..., :2]
    conf = keypoints[..., 2:3]
    center = coords.mean(axis=-2, keepdims=True)
    centered = coords - center
    norms = np.linalg.norm(centered, axis=-1)
    max_norm = np.max(norms, axis=-1, keepdims=True)
    max_norm = np.clip(max_norm, 1e-4, None)
    normalised = centered / max_norm[..., None]
    return np.concatenate([normalised, conf], axis=-1)


def _pad_or_sample(
    array: np.ndarray,
    target_frames: int,
    *,
    axis: int = 0,
) -> np.ndarray:
    length = array.shape[axis]
    if length == target_frames:
        return array
    if length == 0:
        raise ValueError("Cannot pad/sample an empty sequence along axis %d" % axis)
    if length > target_frames:
        idxs = np.linspace(0, length - 1, num=target_frames)
        idxs = np.rint(idxs).astype(int)
        return np.take(array, idxs, axis=axis)
    pad_count = target_frames - length
    pad_idx = np.full(pad_count, length - 1, dtype=int)
    pad_values = np.take(array, pad_idx, axis=axis)
    return np.concatenate([array, pad_values], axis=axis)


def _read_video_frames(path: Path) -> Tuple[np.ndarray, float]:
    if cv2 is None:
        raise RuntimeError("OpenCV is required to read videos")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():  # pragma: no cover - hardware dependent
        raise RuntimeError(f"Unable to open video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frames: List[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (DEFAULT_FRAME_SIZE, DEFAULT_FRAME_SIZE))
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"Video {path} did not contain readable frames")
    stacked = np.stack(frames, axis=0)
    stacked = stacked.astype(np.float32) / 255.0
    return stacked, float(fps)


def _align_modalities(
    keypoints: np.ndarray,
    frames: Optional[np.ndarray],
    *,
    num_kp_frames: int,
    num_video_frames: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    kp = _pad_or_sample(keypoints, num_kp_frames, axis=0)
    video = None
    if frames is not None:
        video = _pad_or_sample(frames, num_video_frames, axis=0)
    return kp, video


def _discover_samples(
    videos_dir: Path,
    keypoints_dir: Path,
) -> List[Tuple[str, Optional[Path], Path]]:
    kp_files = sorted(
        [
            path
            for path in keypoints_dir.rglob("*")
            if path.is_file() and path.suffix in {".npz", ".npy"}
        ]
    )
    samples: List[Tuple[str, Optional[Path], Path]] = []
    video_lookup: Dict[str, Path] = {}
    for extension in VIDEO_EXTENSIONS:
        for video_path in videos_dir.rglob(f"*{extension}"):
            video_lookup[video_path.stem] = video_path
    for kp_path in kp_files:
        stem = kp_path.stem
        video_path = video_lookup.get(stem)
        samples.append((stem, video_path, kp_path))
    return samples


@dataclass
class SampleResult:
    sample_id: str
    translation: str
    target_lang: str
    num_frames: int
    num_keypoint_frames: int


def run_inference(
    model: M2M100ForConditionalGeneration,
    tokenizer: AutoTokenizer,
    adapter: FusionAdapter,
    samples: Sequence[Tuple[str, Optional[Path], Path]],
    generation_config: GenerationConfig,
    *,
    device: torch.device,
    num_kp_frames: int,
    num_video_frames: int,
) -> List[SampleResult]:
    adapter.eval()
    model.eval()

    results: List[SampleResult] = []
    for sample_id, video_path, kp_path in samples:
        LOGGER.info("Processing %s", sample_id)
        keypoints = _load_keypoints(kp_path)
        keypoints = _normalise_keypoints(keypoints)
        frames_array: Optional[np.ndarray] = None
        if video_path is not None:
            try:
                frames_array, _ = _read_video_frames(video_path)
            except Exception as exc:  # pragma: no cover - IO dependent
                LOGGER.warning("Video %s could not be read: %s", video_path, exc)
                frames_array = None

        keypoints, frames_array = _align_modalities(
            keypoints, frames_array, num_kp_frames=num_kp_frames, num_video_frames=num_video_frames
        )

        kp_tensor = (
            torch.from_numpy(keypoints)
            .unsqueeze(0)
            .to(device=device, dtype=torch.float32)
        )
        frames_tensor: Optional[torch.Tensor] = None
        if frames_array is not None:
            frames_tensor = torch.from_numpy(frames_array).permute(0, 3, 1, 2).unsqueeze(0)
            frames_tensor = frames_tensor.to(device=device, dtype=torch.float32)

        with torch.inference_mode():
            embedding = adapter(kp_tensor, frames_tensor)
            encoder_outputs = BaseModelOutput(last_hidden_state=embedding.unsqueeze(1))
            forced_bos_id = tokenizer.lang_code_to_id[generation_config.target_lang]
            generated = model.generate(
                encoder_outputs=encoder_outputs,
                forced_bos_token_id=forced_bos_id,
                max_new_tokens=generation_config.max_new_tokens,
                num_beams=generation_config.num_beams,
                temperature=generation_config.temperature,
                repetition_penalty=generation_config.repetition_penalty,
            )
        text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        results.append(
            SampleResult(
                sample_id=sample_id,
                translation=text,
                target_lang=generation_config.target_lang,
                num_frames=int(frames_array.shape[0]) if frames_array is not None else 0,
                num_keypoint_frames=int(keypoints.shape[0]),
            )
        )
    return results


def _load_adapter_checkpoint(adapter: FusionAdapter, checkpoint: Optional[Path]) -> None:
    if checkpoint is None:
        LOGGER.warning("No adapter checkpoint provided; using random initialisation")
        return
    state_dict = torch.load(checkpoint, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    adapter.load_state_dict(state_dict)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SONAR-SLT inference script")
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=Path("single_signer/videos"),
        help="Directory with input videos.",
    )
    parser.add_argument(
        "--keypoints-dir",
        type=Path,
        default=Path("single_signer/kp"),
        help="Directory with keypoint tensors (npz/npy).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="mtmlt/sonar-nllb-200-1.3B",
        help="Hugging Face identifier for the SONAR decoder.",
    )
    parser.add_argument(
        "--adapter-checkpoint",
        type=Path,
        default=None,
        help="Path to the trained adapter state_dict.",
    )
    parser.add_argument(
        "--target-lang",
        type=str,
        default="spa_Latn",
        help="Target language code compatible with the SONAR tokenizer.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=4,
        help="Beam size for generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Softmax temperature during generation.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Repetition penalty for the decoder.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to run inference on.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSONL file to store translations.",
    )
    parser.add_argument(
        "--num-video-frames",
        type=int,
        default=DEFAULT_NUM_FRAMES,
        help="Number of frames per video clip after sampling/padding.",
    )
    parser.add_argument(
        "--num-keypoint-frames",
        type=int,
        default=DEFAULT_KEYPOINT_FRAMES,
        help="Number of keypoint frames after sampling/padding.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(args)


def main(cli_args: Optional[Sequence[str]] = None) -> None:
    args = parse_args(cli_args)
    _setup_logging(args.verbose)

    if not args.keypoints_dir.exists():
        raise FileNotFoundError(f"Keypoints directory not found: {args.keypoints_dir}")
    if not args.videos_dir.exists():
        LOGGER.warning("Videos directory %s does not exist", args.videos_dir)

    device = torch.device(args.device)
    adapter_config = AdapterConfig()
    adapter = FusionAdapter(adapter_config).to(device)
    _load_adapter_checkpoint(adapter, args.adapter_checkpoint)

    LOGGER.info("Loading SONAR decoder model: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(args.model_name).to(device)
    if args.target_lang not in tokenizer.lang_code_to_id:
        raise ValueError(
            f"Target language '{args.target_lang}' is not supported by the tokenizer"
        )
    generation_config = GenerationConfig(
        target_lang=args.target_lang,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
    )

    samples = _discover_samples(args.videos_dir, args.keypoints_dir)
    if not samples:
        raise RuntimeError(
            "No samples found. Ensure keypoints are stored as .npz/.npy files in the directory."
        )

    results = run_inference(
        model,
        tokenizer,
        adapter,
        samples,
        generation_config,
        device=device,
        num_kp_frames=args.num_keypoint_frames,
        num_video_frames=args.num_video_frames,
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as fh:
            for item in results:
                json.dump(item.__dict__, fh, ensure_ascii=False)
                fh.write("\n")
        LOGGER.info("Saved %d translations to %s", len(results), args.output)
    else:
        for item in results:
            print(f"{item.sample_id}\t{item.target_lang}\t{item.translation}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
