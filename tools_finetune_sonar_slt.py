#!/usr/bin/env python3
r"""
SONAR‑SLT — Fine‑tuning ("paper‑perfect") on LSA using keypoints (+ meta_2.csv)
--------------------------------------------------------------------------------

This script implements the training objective described in SONAR‑SLT
(semantic alignment + decoder cross‑entropy + auto‑encoding), adapted to
keypoint inputs extracted with MediaPipe (npz/npy, shape (T, N, C)).

It is **dataset‑agnostic** and expects a CSV with at least two columns:
  • id / video_id / video  (any of these works)
  • texto / text / sentence / transcript (target sentence in Spanish)

Default layout assumes MediaPipe Holistic (33 body + 468 face + 21 + 21 hands)
producing >=543 landmarks with (x, y, [conf]) per frame.

Example (Windows PowerShell):

  python tools/finetune_sonar_slt.py `
     --keypoints-dir single_signer\kp\keypoints `
     --csv meta_2.csv `
     --out runs\sonar_slt_finetune_lsa `
     --model-name mtmlt/sonar-nllb-200-1.3B `
     --tgt-lang spa_Latn `
     --epochs 10 --batch-size 8 --accum 2 --lr 3e-4 --T 128 --half --device auto

Notes
-----
- If your keypoints are in **single_signer/ke/keypoints**, pass that path explicitly
  or rely on the fallback resolver (the script will try /kp/ then /ke/).
- The SONAR HF port is expected at `mtmlt/sonar-nllb-200-1.3B`. If unavailable,
  you may try a compatible NLLB‑200 checkpoint; decoding/embeddings still work.
- We freeze the decoder by default and *only* train the multimodal adapter +
  a small bridge (1024→d_model). This mirrors the paper’s PEFT spirit.
- Losses follow the paper (equations 5–11):
    L_joint = λ_sem L_sem  +  λ_ce L_ce  +  λ_ae L_ae  +  λ_nce L_nce
  with strong MSE magnitude regularizer and cosine term.
- Optional InfoNCE alignment can be enabled with --lam-nce / --nce-temperature,
  using a queue of recent textual embeddings as extra negatives.

Dependencies
------------
  pip install torch timm "transformers[video]" peft numpy pandas opencv-python rich

Outputs
-------
  • {out}/checkpoints/adapter_stepXXXX.pt       # adapter + bridge weights
  • {out}/train_log.jsonl                        # per‑step metrics
  • {out}/preds_dev.jsonl (optional quick eval)

"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import types
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from transformers.modeling_outputs import BaseModelOutput

try:  # pragma: no cover - optional dependency
    from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict
except Exception:  # pragma: no cover - gracefully handle absence during tests
    def _missing_peft(*_: object, **__: object) -> None:
        raise ModuleNotFoundError(
            "Optional dependency 'peft' is required for LoRA fine-tuning."
            " Install it with `pip install peft`."
        )

    # Provide lightweight shims so module import succeeds without peft.
    def LoraConfig(*args: object, **kwargs: object) -> None:  # type: ignore[misc]
        _missing_peft()

    TaskType = types.SimpleNamespace(SEQ_2_SEQ_LM="seq2seq_lm")  # type: ignore[assignment]

    def get_peft_model(*args: object, **kwargs: object) -> None:  # type: ignore[misc]
        _missing_peft()

    def get_peft_model_state_dict(*args: object, **kwargs: object) -> None:  # type: ignore[misc]
        _missing_peft()

try:  # pragma: no cover - optional dependency
    from models.text_pooler import TextPooler
except Exception:  # pragma: no cover - keep import-time optionality for tests
    class TextPooler:  # type: ignore[dead-code]
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ModuleNotFoundError(
                "Local import 'models.text_pooler.TextPooler' not available."
                " Ensure project modules are on PYTHONPATH when running training."
            )

try:  # pragma: no cover - optional dependency
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - optional dependency
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    ImageFont = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from slt.utils.visual_adapter import (
        AdapterConfig,
        KEYPOINT_CHANNELS as ADAPTER_KEYPOINT_CHANNELS,
        VisualFusionAdapter,
    )
except Exception:  # pragma: no cover - ensure module import during tests
    ADAPTER_KEYPOINT_CHANNELS = 3

    class AdapterConfig:  # type: ignore[dead-code]
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ModuleNotFoundError(
                "Optional visual adapter dependencies are missing."
                " Install project in editable mode to train the adapter."
            )

    class VisualFusionAdapter:  # type: ignore[dead-code]
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ModuleNotFoundError(
                "Optional visual adapter dependencies are missing."
                " Install project in editable mode to train the adapter."
            )

try:  # pragma: no cover - optional dependency
    from slt.data.lsa_t_multistream import (
        _resolve_mediapipe_connections,
        _resolve_mediapipe_layout,
    )
except Exception:  # pragma: no cover - keep training usable without dataset extras
    _resolve_mediapipe_connections = None  # type: ignore[assignment]
    _resolve_mediapipe_layout = None  # type: ignore[assignment]

from sonar_generation import generate_from_hidden_states
from tools_eval_sonar_slt import (
    DEFAULT_METRICS,
    Example as EvalExample,
    compute_metrics as eval_compute_metrics,
)
from tools_keypoint_utils import extract_confidence_channel

# -----------------------------
# Constants / Layout helpers
# -----------------------------
KEYPOINT_CHANNELS = ADAPTER_KEYPOINT_CHANNELS  # (x, y, conf)
DEFAULT_T = 128
DEFAULT_IMG_SIZE = 224
DEFAULT_CLIP_FRAMES = 16
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".mov", ".avi", ".webm")
DEFAULT_D_MODEL = 1024
RNG = random.Random(123)

ID_ALIASES = ("id", "video_id", "video")
TEXT_ALIASES = ("texto", "text", "sentence", "transcript")


# -----------------------------
# Keypoint utilities (normalise + sample) — mirrors repo inference helpers
# -----------------------------
def reshape_flat_keypoints(
    kp: np.ndarray, num_landmarks: int = 0, *, expected_channels: int = KEYPOINT_CHANNELS
) -> np.ndarray:
    """Mirror inference reshape heuristic for flattened MediaPipe dumps."""

    if kp.ndim != 2:
        return kp

    feat_dim = kp.shape[1]

    candidate_points: Optional[int] = None
    if num_landmarks > 0 and feat_dim % num_landmarks == 0:
        candidate_points = num_landmarks
    elif feat_dim % 543 == 0:
        candidate_points = 543
    elif feat_dim % expected_channels == 0:
        candidate_points = feat_dim // expected_channels
    else:
        for channels in range(expected_channels + 1, expected_channels + 4):
            if channels <= 0:
                continue
            if feat_dim % channels == 0:
                candidate_points = feat_dim // channels
                break

    if candidate_points is None or candidate_points <= 0:
        raise ValueError(
            "Cannot infer landmark count from flattened keypoints with feature "
            f"dimension {feat_dim}"
        )

    inferred_channels = feat_dim // candidate_points
    if candidate_points * inferred_channels != feat_dim:
        raise ValueError(
            "Flattened keypoints feature dimension does not evenly divide into "
            f"{candidate_points} landmarks and {inferred_channels} channels"
        )
    if inferred_channels < expected_channels:
        raise ValueError(
            f"Expected at least {expected_channels} channels but found {inferred_channels}"
        )

    reshaped = kp.reshape(-1, candidate_points, inferred_channels)
    if inferred_channels > expected_channels:
        if expected_channels == 3 and inferred_channels >= 3:
            coords = reshaped[..., :2]
            confidence = reshaped[..., -1:]
            reshaped = np.concatenate([coords, confidence], axis=-1)
        else:
            reshaped = reshaped[..., :expected_channels]

    return reshaped


def normalise_keypoints(arr: np.ndarray) -> np.ndarray:
    """Center by mean landmark and scale to unit max‑radius; keep confidence.
    Expects (T, N, C≥2). Returns same shape (T, N, 3).
    """
    coords = arr[..., :2]
    conf = extract_confidence_channel(arr, expected_channels=KEYPOINT_CHANNELS)
    center = coords.mean(axis=-2, keepdims=True)
    centered = coords - center
    norms = np.linalg.norm(centered, axis=-1)
    max_norm = np.max(norms, axis=-1, keepdims=True)
    max_norm = np.clip(max_norm, 1e-4, None)
    normalised = centered / max_norm[..., None]
    return np.concatenate([normalised, conf], axis=-1)


def pad_or_sample(array: np.ndarray, target_frames: int, axis: int = 0) -> np.ndarray:
    length = array.shape[axis]
    if length == target_frames:
        return array
    if length == 0:
        raise ValueError("Cannot pad/sample an empty keypoint sequence")
    if length > target_frames:
        idxs = np.linspace(0, length - 1, num=target_frames)
        idxs = np.rint(idxs).astype(int)
        return np.take(array, idxs, axis=axis)
    pad_count = target_frames - length
    pad_idx = np.full(pad_count, length - 1, dtype=int)
    pad_values = np.take(array, pad_idx, axis=axis)
    return np.concatenate([array, pad_values], axis=axis)


# -----------------------------
# CSV loader
# -----------------------------
@dataclass
class CsvRow:
    vid: str
    text: str


def _read_meta_csv(path: Path) -> List[CsvRow]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=";")
        rows = list(reader)
    out: List[CsvRow] = []
    for row in rows:
        vid = None
        for k in ID_ALIASES:
            if k in row and str(row[k]).strip():
                vid = str(row[k]).strip()
                break
        txt = None
        for k in TEXT_ALIASES:
            if k in row and str(row[k]).strip():
                txt = str(row[k]).strip()
                break
        if not vid or not txt:
            continue
        out.append(CsvRow(vid, txt))
    if not out:
        raise ValueError(f"CSV {path} did not contain required columns {ID_ALIASES} + {TEXT_ALIASES}")
    return out


# -----------------------------
# Dataset
# -----------------------------
class KPTextDataset(Dataset):
    def __init__(
        self,
        keypoints_dir: Path,
        csv_path: Path,
        T: int = DEFAULT_T,
        *,
        shuffle: bool = True,
        video_dir: Optional[Path] = None,
        clip_frames: int = DEFAULT_CLIP_FRAMES,
        frame_size: int = DEFAULT_IMG_SIZE,
    ) -> None:
        self.keypoints_dir = keypoints_dir
        self.video_dir = video_dir
        if self.video_dir is not None and cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required to load paired videos")
        self.items = _read_meta_csv(csv_path)
        if shuffle:
            RNG.shuffle(self.items)
        self.T = int(T)
        self.clip_frames = int(clip_frames)
        self.frame_size = int(frame_size)

    def __len__(self) -> int:
        return len(self.items)

    def _load_np(self, path: Path) -> np.ndarray:
        if path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=True) as data:
                if "keypoints" in data:
                    arr = data["keypoints"]
                elif "frames" in data:
                    arr = data["frames"]
                else:
                    raise KeyError(f"{path} lacks 'keypoints' or 'frames'")
                return np.asarray(arr, dtype=np.float32)
        return np.asarray(np.load(path), dtype=np.float32)

    def __getitem__(self, idx: int):
        row = self.items[idx]
        base = self.keypoints_dir / row.vid
        kp_path: Optional[Path] = None
        for ext in (".npz", ".npy"):
            cand = base.with_suffix(ext)
            if cand.exists():
                kp_path = cand
                break
        if kp_path is None:
            raise FileNotFoundError(f"Keypoints not found for id={row.vid} in {self.keypoints_dir}")
        keypoints = self._load_np(kp_path)
        keypoints = reshape_flat_keypoints(
            keypoints, num_landmarks=0, expected_channels=KEYPOINT_CHANNELS
        )
        keypoints = normalise_keypoints(keypoints)
        keypoints = pad_or_sample(keypoints, self.T, axis=0)
        video_tensor: Optional[torch.Tensor] = None
        if self.video_dir is not None:
            video_path = self._resolve_video_path(row.vid)
            if video_path is None:
                raise FileNotFoundError(
                    f"Video not found for id={row.vid} in {self.video_dir}"
                )
            video_np = self._load_video(video_path)
            video_np = pad_or_sample(video_np, self.clip_frames, axis=0)
            video_tensor = torch.from_numpy(video_np).to(torch.float32)
        return {
            "id": row.vid,
            "keypoints": torch.from_numpy(keypoints).to(torch.float32),  # (T, N, C)
            "video": video_tensor,
            "text": row.text,
        }

    def _resolve_video_path(self, stem: str) -> Optional[Path]:
        if self.video_dir is None:
            return None

        raw = stem.strip()
        if not raw:
            return None

        candidates: List[str] = []

        def _push(value: str) -> None:
            normalised = value.strip()
            if normalised and normalised not in candidates:
                candidates.append(normalised)

        _push(raw)
        _push(raw.lower())

        raw_path = Path(raw)
        if raw_path.suffix:
            _push(raw_path.stem)
            _push(raw_path.stem.lower())

        for name in candidates:
            literal = self.video_dir / name
            if literal.exists():
                return literal

            # Only append extensions when the candidate does not already provide one.
            if Path(name).suffix:
                continue

            for ext in VIDEO_EXTENSIONS:
                candidate = self.video_dir / f"{name}{ext}"
                if candidate.exists():
                    return candidate

        return None

    def _load_video(self, path: Path) -> np.ndarray:
        if cv2 is None:
            raise RuntimeError("OpenCV is required to read video data")
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():  # pragma: no cover - hardware dependent
            raise RuntimeError(f"Unable to open video: {path}")
        frames: List[np.ndarray] = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.frame_size, self.frame_size))
            frames.append(frame.astype(np.float32) / 255.0)
        capture.release()
        if not frames:
            raise RuntimeError(f"Video {path} did not contain readable frames")
        arr = np.stack(frames, axis=0)  # (T, H, W, C)
        arr = arr.transpose(0, 3, 1, 2)  # (T, C, H, W)
        return arr

    def load_video_frame(
        self,
        vid: str,
        *,
        frame_index: int,
        keypoint_frames: int,
    ) -> Optional[np.ndarray]:
        """Return a single clip-aligned frame for ``vid`` if videos are available."""

        if self.video_dir is None:
            return None

        video_path = self._resolve_video_path(vid)
        if video_path is None:
            return None

        video_np = self._load_video(video_path)
        if video_np.size == 0 or video_np.shape[0] == 0:
            return None

        try:
            video_np = pad_or_sample(video_np, self.clip_frames, axis=0)
        except ValueError:
            return None

        if video_np.shape[0] == 0:
            return None

        if keypoint_frames > 1 and video_np.shape[0] > 1:
            ratio = frame_index / float(max(keypoint_frames - 1, 1))
            approx_idx = int(round(ratio * (video_np.shape[0] - 1)))
        else:
            approx_idx = 0
        approx_idx = max(0, min(approx_idx, video_np.shape[0] - 1))

        return video_np[approx_idx]


# -----------------------------
# Model definitions live in `slt.utils.visual_adapter`
# -----------------------------


def _render_keypoint_preview(
    frame: np.ndarray,
    *,
    layout: Dict[str, Sequence[int]],
    connections: Dict[str, Sequence[Tuple[int, int]]],
    width: int,
    height: int,
    confidence_threshold: float,
    text: Optional[str] = None,
    video_frame: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Render a preview image with keypoints and optional reference text."""

    if cv2 is None:  # pragma: no cover - optional dependency
        raise RuntimeError("OpenCV is required to render keypoint previews")

    if frame.ndim != 2 or frame.shape[1] < 2:
        raise ValueError("Expected frame with shape (N, C>=2) for keypoint preview")

    width = max(int(width), 1)
    height = max(int(height), 1)

    coords = np.asarray(frame[..., :2], dtype=np.float32)
    conf = frame[..., 2] if frame.shape[1] > 2 else np.ones(frame.shape[0], dtype=np.float32)
    conf = np.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0)

    # Default to treating coordinates as being in the [0, 1] range, as produced by
    # `LsaTMultiStream._sample_keypoints`. Fallback to the legacy [-1, 1]
    # interpretation when we observe values outside the expected [0, 1] bounds.
    xs = coords[:, 0] * (width - 1)
    ys = coords[:, 1] * (height - 1)
    finite_coords = coords[np.isfinite(coords)]
    if finite_coords.size:
        min_coord = float(finite_coords.min())
        max_coord = float(finite_coords.max())
        eps = 1e-3
        if min_coord < -eps or max_coord > 1.0 + eps:
            xs = ((coords[:, 0] + 1.0) * 0.5) * (width - 1)
            ys = ((coords[:, 1] + 1.0) * 0.5) * (height - 1)
    points = np.stack([xs, ys], axis=-1)
    points = np.nan_to_num(points, nan=-1.0, posinf=-1.0, neginf=-1.0)
    points = points.astype(np.int32)

    valid = conf >= float(confidence_threshold)

    if video_frame is not None:
        frame_np = video_frame
        if isinstance(frame_np, torch.Tensor):
            frame_np = frame_np.detach().cpu().numpy()
        frame_np = np.asarray(frame_np)
        if frame_np.ndim == 3 and frame_np.shape[0] in {1, 3}:  # (C,H,W) → (H,W,C)
            frame_np = np.transpose(frame_np, (1, 2, 0))
        frame_np = np.nan_to_num(frame_np, nan=0.0)
        if frame_np.dtype != np.uint8:
            frame_np = frame_np.astype(np.float32, copy=False)
            max_val = 255.0 if frame_np.max() > 1.0 else 1.0
            frame_np = np.clip(frame_np / max(max_val, 1e-6), 0.0, 1.0)
            frame_np = (frame_np * 255.0).astype(np.uint8)
        if frame_np.shape[0] != height or frame_np.shape[1] != width:
            frame_np = cv2.resize(frame_np, (width, height), interpolation=cv2.INTER_LINEAR)
        canvas = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
    else:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

    segment_colours = {
        "body": (0, 255, 255),
        "hand_l": (255, 144, 30),
        "hand_r": (30, 144, 255),
        "face": (144, 255, 144),
    }

    drawn: Set[int] = set()
    for segment, pairs in connections.items():
        colour = segment_colours.get(segment, (255, 255, 255))
        for start, end in pairs:
            if start >= points.shape[0] or end >= points.shape[0]:
                continue
            if not (valid[start] and valid[end]):
                continue
            cv2.line(
                canvas,
                tuple(points[start]),
                tuple(points[end]),
                colour,
                2,
                cv2.LINE_AA,
            )
            drawn.add(start)
            drawn.add(end)

    for segment, indices in layout.items():
        colour = segment_colours.get(segment, (255, 255, 255))
        for idx in indices:
            if idx >= points.shape[0] or not valid[idx]:
                continue
            cv2.circle(canvas, tuple(points[idx]), 3, colour, -1, cv2.LINE_AA)
            drawn.add(idx)

    for idx in range(points.shape[0]):
        if not valid[idx] or idx in drawn:
            continue
        cv2.circle(canvas, tuple(points[idx]), 2, (255, 255, 255), -1, cv2.LINE_AA)

    if text:
        max_width = width - 20
        if Image is not None and ImageDraw is not None and ImageFont is not None:
            rgb_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_canvas)
            draw = ImageDraw.Draw(image)

            font_size = 20
            font_path = Path(__file__).resolve().parent / "fonts" / "DejaVuSans.ttf"
            try:
                font = ImageFont.truetype(str(font_path), size=font_size)
            except (OSError, IOError):  # pragma: no cover - optional font dependency
                font = ImageFont.load_default()

            def _wrap(line: str) -> List[str]:
                words = line.split()
                if not words:
                    return [""]
                lines: List[str] = []
                current: List[str] = []
                for word in words:
                    tentative = " ".join(current + [word])
                    width_px = draw.textlength(tentative, font=font)
                    if width_px <= max_width or not current:
                        current.append(word)
                        continue
                    lines.append(" ".join(current))
                    current = [word]
                if current:
                    lines.append(" ".join(current))
                return lines

            lines = []
            for raw_line in str(text).splitlines():
                lines.extend(_wrap(raw_line.strip()))

            try:
                ascent, descent = font.getmetrics()
                line_height = ascent + descent
            except (AttributeError, TypeError):
                line_height = getattr(font, "size", font_size)
            line_height = max(int(line_height), 16)
            line_gap = max(12, int(line_height * 0.3))
            blank_gap = max(18, int(line_height * 0.6))

            y = 24
            for line in lines:
                if not line:
                    y += blank_gap
                    continue
                draw.text((10, y), line, font=font, fill=(255, 255, 255))
                y += line_height + line_gap

            canvas = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2

            def _wrap(line: str) -> List[str]:
                words = line.split()
                if not words:
                    return [""]
                lines: List[str] = []
                current: List[str] = []
                for word in words:
                    tentative = " ".join(current + [word])
                    size, _ = cv2.getTextSize(tentative, font, font_scale, thickness)
                    if size[0] <= max_width or not current:
                        current.append(word)
                        continue
                    lines.append(" ".join(current))
                    current = [word]
                if current:
                    lines.append(" ".join(current))
                return lines

            lines = []
            for raw_line in str(text).splitlines():
                lines.extend(_wrap(raw_line.strip()))

            y = 24
            for line in lines:
                if not line:
                    y += 18
                    continue
                cv2.putText(
                    canvas,
                    line,
                    (10, y),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )
                y += int(18 * font_scale) + 12

    return canvas

# -----------------------------
# Text encoder / pooling with SONAR HF port (M2M‑style mean pooling)
# -----------------------------
# -----------------------------
# Augmentations (simple keypoint perturbations, coupled to target aug future‑proof)
# -----------------------------
def jitter_keypoints(kp: torch.Tensor, noise_std: float = 0.04, drop_prob: float = 0.2) -> torch.Tensor:
    """Apply small Gaussian jitter and random frame dropout (repeat last)."""
    B, T, N, C = kp.shape
    if noise_std > 0:
        noise = torch.randn_like(kp[..., :2]) * noise_std
        kp = kp.clone()
        kp[..., :2] = kp[..., :2] + noise
    if drop_prob > 0:
        keep = torch.rand(B, T, 1, 1, device=kp.device) >= drop_prob
        last = kp[:, -1:, :, :]
        kp = torch.where(keep, kp, last)
    return kp


# -----------------------------
# Serialization helpers
# -----------------------------
def _serialise_lora_config(lora_config: object) -> Dict[str, object]:
    """Return a JSON/torch.save friendly representation of a LoRA config."""

    cfg_dict = dict(lora_config.to_dict() if hasattr(lora_config, "to_dict") else {})
    return _convert_to_primitives(cfg_dict)


def _convert_to_primitives(value: object) -> object:
    """Recursively convert values to ``torch.save`` friendly primitives."""

    if isinstance(value, Enum):
        enum_value = getattr(value, "value", None)
        if isinstance(enum_value, str):
            return enum_value
        enum_name = getattr(value, "name", None)
        if isinstance(enum_name, str):
            return enum_name
        return str(value)
    if isinstance(value, dict):
        return {k: _convert_to_primitives(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        converted = [_convert_to_primitives(v) for v in value]
        return type(value)(converted)
    return value


def _serialise_train_config(cfg: TrainConfig) -> Dict[str, object]:
    """Convert :class:`TrainConfig` into primitives accepted by ``torch.save``."""

    raw = asdict(cfg)
    serialised: Dict[str, object] = {}
    for key, value in raw.items():
        if isinstance(value, Path):
            serialised[key] = str(value)
        else:
            serialised[key] = _convert_to_primitives(value)
    return serialised


# -----------------------------
# Training loop
# -----------------------------
@dataclass
class TrainConfig:
    keypoints_dir: Path
    csv: Path
    out_dir: Path
    dev_csv: Optional[Path] = None
    video_dir: Optional[Path] = None
    model_name: str = "mtmlt/sonar-nllb-200-1.3B"
    tgt_lang: str = "spa_Latn"
    T: int = DEFAULT_T
    clip_frames: int = DEFAULT_CLIP_FRAMES
    frame_size: int = DEFAULT_IMG_SIZE
    batch_size: int = 8
    accum: int = 2
    epochs: int = 10
    lr: float = 3e-4
    bridge_lr: Optional[float] = None
    lora_lr: Optional[float] = None
    weight_decay: float = 0.01
    device: str = "auto"  # auto|cuda|cpu|mps
    half: bool = False
    save_every: int = 500
    log_every: int = 50
    eval_every: int = 500
    mse_weight: float = 7000.0   # λ_mse inside L_sem (paper impl. details)
    cos_weight: float = 2.7      # λ_cos inside L_sem
    lam_sem: float = 1.0         # λ_sem
    lam_ce: float = 0.1          # λ_ce
    lam_ae: float = 0.2          # λ_ae
    lam_nce: float = 0.0         # λ_nce (InfoNCE alignment)
    nce_temperature: float = 0.07
    text_pool_layers: int = 4
    vit_model: str = "vit_base_patch16_224"
    vit_checkpoint: Optional[Path] = None
    videomae_model: str = "MCG-NJU/videomae-base"
    videomae_checkpoint: Optional[Path] = None
    freeze_keypoint: bool = False
    freeze_vit: bool = False
    freeze_videomae: bool = False
    save_samples: bool = False
    preview_width: int = 640
    preview_height: int = 640
    preview_confidence_threshold: float = 0.2
    preview_seed: int = 1337
    sample_metrics: Tuple[str, ...] = DEFAULT_METRICS
    sample_max_new_tokens: int = 64
    sample_num_beams: int = 4
    sample_metrics_lowercase: bool = False


def _cuda_is_available() -> bool:
    """Return True if torch.cuda is available, guarding against CPU-only builds."""

    try:  # torch.cuda.is_available may raise when CUDA support is missing entirely
        return bool(torch.cuda.is_available())
    except (AssertionError, RuntimeError):
        return False


def resolve_device(name: str) -> torch.device:
    raw_name = name.strip()
    lowered = raw_name.lower()
    cuda_available = _cuda_is_available()
    mps_available = torch.backends.mps.is_available()

    if lowered == "auto":
        if cuda_available:
            return torch.device("cuda")
        if mps_available:
            return torch.device("mps")
        return torch.device("cpu")

    try:
        device = torch.device(raw_name)
        if device.type == "cuda" and not cuda_available:
            print("[device] Requested CUDA but PyTorch lacks CUDA support — using CPU instead.")
            return torch.device("cpu")
        if device.type == "mps" and not mps_available:
            print("[device] Requested MPS but backend is unavailable — using CPU instead.")
            return torch.device("cpu")
        return device
    except (TypeError, RuntimeError, ValueError, AttributeError):
        pass

    if lowered in {"cuda", "gpu"}:
        if cuda_available:
            return torch.device("cuda")
        print("[device] CUDA requested but unavailable — using CPU instead.")
    elif lowered == "mps":
        if mps_available:
            return torch.device("mps")
        print("[device] MPS requested but unavailable — using CPU instead.")
    return torch.device("cpu")


def count_landmarks(example: torch.Tensor) -> int:
    # heuristic to infer N from single example (T, N, C)
    return int(example.shape[1])


def _generate_sample_prediction(
    sample: Dict[str, object],
    *,
    adapter: nn.Module,
    bridge: nn.Module,
    decoder: nn.Module,
    tokenizer,
    device: torch.device,
    cfg: TrainConfig,
) -> str:
    modules: List[nn.Module] = [adapter, bridge, decoder]
    prev_training_states = [module.training for module in modules]
    for module in modules:
        module.eval()

    try:
        with torch.no_grad():
            kp_tensor = sample["keypoints"]
            if not isinstance(kp_tensor, torch.Tensor):
                raise TypeError("Sample is missing keypoints tensor for preview generation")
            kp_tensor = kp_tensor.unsqueeze(0).to(device)

            video_tensor = sample.get("video") if isinstance(sample, dict) else None
            frames_tensor: Optional[torch.Tensor] = None
            if isinstance(video_tensor, torch.Tensor):
                frames_tensor = video_tensor.unsqueeze(0).to(device)

            z = adapter(kp_tensor, frames_tensor)
            z = z.to(bridge.weight.device)
            z = z.to(bridge.weight.dtype)
            mem = bridge(z).unsqueeze(1)

            tokenizer.src_lang = cfg.tgt_lang
            outputs = generate_from_hidden_states(
                decoder,
                tokenizer,
                mem,
                cfg.tgt_lang,
                generation_options={
                    "max_new_tokens": cfg.sample_max_new_tokens,
                    "num_beams": cfg.sample_num_beams,
                },
            )
            if not outputs:
                return ""
            return outputs[0]
    finally:
        for module, was_training in zip(modules, prev_training_states):
            module.train(was_training)


def _compute_batch_losses(
    *,
    cfg: TrainConfig,
    adapter: nn.Module,
    bridge: nn.Module,
    decoder: nn.Module,
    text_pool: TextPooler,
    tokenizer,
    device: torch.device,
    texts: Sequence[str],
    keypoints: torch.Tensor,
    frames_tensor: Optional[torch.Tensor],
    nce_queue: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[torch.Tensor]]:
    """Run a forward pass and return the weighted loss + metrics."""

    with torch.amp.autocast("cuda", enabled=cfg.half and device.type == "cuda"):
        z = adapter(keypoints, frames_tensor)  # [B,1024]
        mem_z = bridge(z).unsqueeze(1)  # [B,1,d_model]

        s = text_pool.encode(texts).to(mem_z.dtype)
        with torch.no_grad():
            W = bridge.weight  # [d_model,1024]
            W_f32 = W.to(torch.float32)
            W_pinv_fp32 = torch.linalg.pinv(W_f32, rcond=1e-5)
        s_fp32 = s.to(torch.float32)
        s_1024_fp32 = torch.matmul(s_fp32, W_pinv_fp32.T).contiguous()
        s_1024 = s_1024_fp32.to(z.dtype)

        mse = F.mse_loss(z, s_1024)
        cos = 1.0 - F.cosine_similarity(z, s_1024, dim=-1).mean()
        L_sem = cfg.mse_weight * mse + cfg.cos_weight * cos

        tokenizer.src_lang = cfg.tgt_lang
        batch = tokenizer(texts, return_tensors="pt", padding=True).to(device)
        labels = batch["input_ids"].clone()
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        enc_out_z = BaseModelOutput(last_hidden_state=mem_z)
        out_z = decoder(encoder_outputs=enc_out_z, labels=labels)
        L_ce = out_z.loss

        mem_s = s.unsqueeze(1)
        enc_out_s = BaseModelOutput(last_hidden_state=mem_s)
        out_s = decoder(encoder_outputs=enc_out_s, labels=labels)
        L_ae = out_s.loss

        L_nce = torch.zeros((), device=z.device, dtype=z.dtype)
        s_norm: Optional[torch.Tensor] = None
        if cfg.lam_nce > 0:
            z_norm = F.normalize(z.to(torch.float32), dim=-1)
            s_norm = F.normalize(s_1024_fp32, dim=-1)
            negatives = [s_norm]
            if nce_queue is not None and nce_queue.numel() > 0:
                negatives.append(nce_queue)
            all_targets = torch.cat(negatives, dim=0)
            logits = torch.matmul(z_norm, all_targets.t()) / max(cfg.nce_temperature, 1e-6)
            labels_nce = torch.arange(z_norm.size(0), device=z.device)
            L_nce = F.cross_entropy(logits, labels_nce).to(z.dtype)

        loss = (
            cfg.lam_sem * L_sem
            + cfg.lam_ce * L_ce
            + cfg.lam_ae * L_ae
            + cfg.lam_nce * L_nce
        )

    metrics = {
        "loss": loss,
        "L_sem": L_sem,
        "mse": mse,
        "cos": cos,
        "L_ce": L_ce,
        "L_ae": L_ae,
        "L_nce": L_nce,
    }
    return loss, metrics, s_norm


def _evaluate_on_loader(
    dev_dl: DataLoader,
    *,
    cfg: TrainConfig,
    adapter: nn.Module,
    bridge: nn.Module,
    decoder: nn.Module,
    text_pool: TextPooler,
    tokenizer,
    device: torch.device,
) -> Optional[Dict[str, float]]:
    """Compute average losses over ``dev_dl`` using eval mode."""

    if dev_dl is None:
        return None

    modules: List[nn.Module] = [adapter, bridge, decoder]
    prev_states = [module.training for module in modules]
    for module in modules:
        module.eval()

    try:
        totals: Dict[str, float] = {}
        num_batches = 0
        with torch.no_grad():
            for _ids, texts, kp, video in dev_dl:
                kp = kp.to(device)
                frames_tensor: Optional[torch.Tensor] = None
                if video is not None:
                    frames_tensor = video.to(device)
                _, metrics, _ = _compute_batch_losses(
                    cfg=cfg,
                    adapter=adapter,
                    bridge=bridge,
                    decoder=decoder,
                    text_pool=text_pool,
                    tokenizer=tokenizer,
                    device=device,
                    texts=texts,
                    keypoints=kp,
                    frames_tensor=frames_tensor,
                    nce_queue=None,
                )
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
                num_batches += 1

        if num_batches == 0:
            return None

        return {key: total / num_batches for key, total in totals.items()}
    finally:
        for module, was_training in zip(modules, prev_states):
            module.train(was_training)


def train(cfg: TrainConfig) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # Resolve kp folder (kp → ke fallback if not present)
    if not cfg.keypoints_dir.exists():
        alt = Path(str(cfg.keypoints_dir).replace(os.sep + "kp" + os.sep, os.sep + "ke" + os.sep))
        if alt.exists():
            cfg.keypoints_dir = alt

    # Data
    if cfg.video_dir is not None and not cfg.video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {cfg.video_dir}")

    ds = KPTextDataset(
        cfg.keypoints_dir,
        cfg.csv,
        T=cfg.T,
        video_dir=cfg.video_dir,
        clip_frames=cfg.clip_frames,
        frame_size=cfg.frame_size,
    )
    dev_ds: Optional[KPTextDataset] = None
    if cfg.dev_csv is not None:
        dev_ds = KPTextDataset(
            cfg.keypoints_dir,
            cfg.dev_csv,
            T=cfg.T,
            video_dir=cfg.video_dir,
            clip_frames=cfg.clip_frames,
            frame_size=cfg.frame_size,
            shuffle=False,
        )
    N = count_landmarks(ds[0]["keypoints"])  # landmarks

    # Model pieces
    device = resolve_device(cfg.device)
    decoder = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)
    tok = AutoTokenizer.from_pretrained(cfg.model_name)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
    )
    decoder = get_peft_model(decoder, lora_config).to(device)
    # Keep the frozen decoder weights in FP32; autocast handles mixed-precision casts.

    # Bridge maps adapter’s 1024 to decoder d_model if needed
    d_model = getattr(decoder.config, "d_model", None) or getattr(decoder.config, "hidden_size", DEFAULT_D_MODEL)
    bridge = nn.Linear(1024, d_model, bias=False).to(device)
    # Bridge remains FP32 so GradScaler can safely unscale gradients.

    adapter = VisualFusionAdapter(
        AdapterConfig(),
        num_points=N,
        vit_name=cfg.vit_model,
        vit_checkpoint=cfg.vit_checkpoint,
        freeze_vit=cfg.freeze_vit,
        videomae_name=cfg.videomae_model,
        videomae_checkpoint=cfg.videomae_checkpoint,
        freeze_videomae=cfg.freeze_videomae,
        freeze_keypoint=cfg.freeze_keypoint,
    ).to(device)
    # Adapter gradients must stay in FP32 for stable unscaling during AMP.

    text_pool = TextPooler(decoder, tok, cfg.tgt_lang, num_layers=cfg.text_pool_layers)

    # Optimiser
    adapter_params = [p for p in adapter.parameters() if p.requires_grad]
    bridge_params = [p for p in bridge.parameters() if p.requires_grad]
    lora_params = [p for p in decoder.parameters() if p.requires_grad]

    lr_adapter = cfg.lr
    lr_bridge = cfg.bridge_lr if cfg.bridge_lr is not None else cfg.lr
    lr_lora = cfg.lora_lr if cfg.lora_lr is not None else cfg.lr

    param_groups = []
    if adapter_params:
        param_groups.append({"params": adapter_params, "lr": lr_adapter})
    if bridge_params:
        param_groups.append({"params": bridge_params, "lr": lr_bridge})
    if lora_params:
        param_groups.append({"params": lora_params, "lr": lr_lora})

    if not param_groups:
        raise RuntimeError("No trainable parameters found for optimiser")

    opt = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.half and device.type == "cuda")

    # Loader
    def collate(batch):
        ids = [b["id"] for b in batch]
        texts = [b["text"] for b in batch]
        kps = torch.stack([b["keypoints"] for b in batch], dim=0)  # [B,T,N,C]
        videos: Optional[torch.Tensor] = None
        if batch[0]["video"] is not None:
            videos = torch.stack([b["video"] for b in batch], dim=0)  # [B,TV,C,H,W]
        return ids, texts, kps, videos

    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
        pin_memory=(device.type == "cuda"),
    )
    dev_dl: Optional[DataLoader] = None
    if dev_ds is not None:
        dev_dl = DataLoader(
            dev_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate,
            pin_memory=(device.type == "cuda"),
        )

    layout: Dict[str, List[int]] = {}
    connections: Dict[str, List[Tuple[int, int]]] = {}
    if _resolve_mediapipe_layout is not None and _resolve_mediapipe_connections is not None:
        try:
            layout = _resolve_mediapipe_layout(N)
            connections = _resolve_mediapipe_connections(layout)
        except Exception as exc:  # pragma: no cover - layout resolution is optional
            layout = {}
            connections = {}
            print(f"[preview] Failed to resolve MediaPipe layout: {exc}")

    preview_rng = random.Random(int(cfg.preview_seed))

    step = 0
    log_path = cfg.out_dir / "train_log.jsonl"
    dev_log_path = cfg.out_dir / "dev_log.jsonl"
    accum_counter = 0
    nce_queue: Optional[torch.Tensor] = None
    nce_queue_max = 4096
    for epoch in range(cfg.epochs):
        total_batches = len(dl)
        for batch_idx, (ids, texts, kp, video) in enumerate(dl):
            step += 1
            accum_counter += 1
            kp = kp.to(device)
            frames_tensor: Optional[torch.Tensor] = None
            if video is not None:
                frames_tensor = video.to(device)
            kp = jitter_keypoints(kp, noise_std=0.04, drop_prob=0.2)

            loss, metrics, s_norm = _compute_batch_losses(
                cfg=cfg,
                adapter=adapter,
                bridge=bridge,
                decoder=decoder,
                text_pool=text_pool,
                tokenizer=tok,
                device=device,
                texts=texts,
                keypoints=kp,
                frames_tensor=frames_tensor,
                nce_queue=nce_queue,
            )

            batches_remaining_after_current = total_batches - (batch_idx + 1)
            effective_accum = min(cfg.accum, accum_counter + batches_remaining_after_current)
            effective_accum = max(effective_accum, 1)

            scaler.scale(loss / effective_accum).backward()

            is_accum_boundary = accum_counter >= cfg.accum
            is_last_batch = (batch_idx + 1) == total_batches

            if is_accum_boundary or is_last_batch:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                accum_counter = 0

            if cfg.lam_nce > 0 and s_norm is not None:
                with torch.no_grad():
                    new_entries = s_norm.detach()
                    if nce_queue is None:
                        nce_queue = new_entries
                    else:
                        nce_queue = torch.cat([nce_queue, new_entries], dim=0)
                        if nce_queue.shape[0] > nce_queue_max:
                            nce_queue = nce_queue[-nce_queue_max :].contiguous()

            if step % cfg.log_every == 0:
                rec = {
                    "split": "train",
                    "step": step,
                    "epoch": epoch,
                    **{k: float(v.detach().cpu()) for k, v in metrics.items()},
                }
                with log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(
                    "[step {step:6d}] loss={loss:.4f}  L_sem={L_sem:.4f}  L_ce={L_ce:.4f}  L_ae={L_ae:.4f}  L_nce={L_nce:.4f}".format(
                        step=step,
                        loss=rec["loss"],
                        L_sem=rec["L_sem"],
                        L_ce=rec["L_ce"],
                        L_ae=rec["L_ae"],
                        L_nce=rec["L_nce"],
                    )
                )

            if (
                dev_dl is not None
                and cfg.eval_every > 0
                and step % cfg.eval_every == 0
            ):
                dev_metrics = _evaluate_on_loader(
                    dev_dl,
                    cfg=cfg,
                    adapter=adapter,
                    bridge=bridge,
                    decoder=decoder,
                    text_pool=text_pool,
                    tokenizer=tok,
                    device=device,
                )
                if dev_metrics is not None:
                    rec_dev = {
                        "split": "dev",
                        "step": step,
                        "epoch": epoch,
                        **{k: float(v) for k, v in dev_metrics.items()},
                    }
                    payload = json.dumps(rec_dev, ensure_ascii=False) + "\n"
                    with log_path.open("a", encoding="utf-8") as fh:
                        fh.write(payload)
                    with dev_log_path.open("a", encoding="utf-8") as fh:
                        fh.write(payload)
                    print(
                        "[dev  {step:6d}] loss={loss:.4f}  L_sem={L_sem:.4f}  L_ce={L_ce:.4f}  L_ae={L_ae:.4f}  L_nce={L_nce:.4f}".format(
                            step=step,
                            loss=rec_dev["loss"],
                            L_sem=rec_dev["L_sem"],
                            L_ce=rec_dev["L_ce"],
                            L_ae=rec_dev["L_ae"],
                            L_nce=rec_dev["L_nce"],
                        )
                    )

            if step % cfg.save_every == 0:
                ckpt = {
                    "adapter": adapter.state_dict(),
                    "bridge": bridge.state_dict(),
                    "lora": get_peft_model_state_dict(decoder),
                    "lora_config": _serialise_lora_config(lora_config),
                    "config": _serialise_train_config(cfg),
                    "d_model": d_model,
                    "num_landmarks": N,
                }
                path = cfg.out_dir / "checkpoints" / f"adapter_step{step:06d}.pt"
                torch.save(ckpt, path)
                print(f"[ckpt] Saved {path}")
                if cfg.save_samples:
                    preview_dir = cfg.out_dir / "checkpoints" / f"adapter_step{step:06d}" / "preview"
                    preview_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        sample_idx = preview_rng.randrange(len(ds))
                        sample = ds[sample_idx]
                    except Exception as exc:  # pragma: no cover - sampling is best-effort
                        print(f"[preview] Failed to sample example: {exc}")
                        sample = None

                    if sample is not None:
                        reference_path = preview_dir / "reference.txt"
                        sample_text = str(sample["text"]) if isinstance(sample, dict) else str(sample)
                        reference_text = f"id: {sample['id']}\ntext: {sample_text}\n"
                        reference_path.write_text(reference_text, encoding="utf-8")
                        print(f"[preview] Saved reference text → {reference_path}")
                        if cv2 is None:
                            print("[preview] OpenCV not available; skipped keypoints.png generation.")
                        else:
                            try:
                                kp_tensor = sample["keypoints"].detach().cpu().numpy()
                                total_frames = kp_tensor.shape[0]
                                if total_frames <= 0:
                                    raise ValueError("Sample contained no keypoint frames")
                                frame_conf = kp_tensor[..., 2] if kp_tensor.shape[-1] > 2 else np.ones(
                                    kp_tensor.shape[:2], dtype=np.float32
                                )
                                frame_scores = frame_conf.mean(axis=1)
                                frame_idx = int(np.argmax(frame_scores)) if frame_scores.size else 0
                                frame_idx = max(0, min(frame_idx, total_frames - 1))
                                kp_frame = kp_tensor[frame_idx]

                                video_frame: Optional[np.ndarray] = None
                                video_tensor = sample.get("video") if isinstance(sample, dict) else None
                                if video_tensor is not None:
                                    if isinstance(video_tensor, torch.Tensor):
                                        video_np = video_tensor.detach().cpu().numpy()
                                    else:
                                        video_np = np.asarray(video_tensor)
                                    if video_np.shape[0] == 0:
                                        raise ValueError("Sample video contained no frames")
                                    if total_frames > 1 and video_np.shape[0] > 1:
                                        ratio = frame_idx / float(max(total_frames - 1, 1))
                                        frame_idx_vid = int(round(ratio * (video_np.shape[0] - 1)))
                                    else:
                                        frame_idx_vid = 0
                                    frame_idx_vid = max(0, min(frame_idx_vid, video_np.shape[0] - 1))
                                    video_frame = video_np[frame_idx_vid]
                                elif hasattr(ds, "load_video_frame"):
                                    try:
                                        video_frame = ds.load_video_frame(
                                            sample["id"],
                                            frame_index=frame_idx,
                                            keypoint_frames=total_frames,
                                        )
                                    except Exception as exc:
                                        print(f"[preview] Failed to fetch video frame: {exc}")

                                image = _render_keypoint_preview(
                                    kp_frame,
                                    layout=layout,
                                    connections=connections,
                                    width=cfg.preview_width,
                                    height=cfg.preview_height,
                                    confidence_threshold=cfg.preview_confidence_threshold,
                                    text=f"{sample['id']}: {sample['text']}",
                                    video_frame=video_frame,
                                )
                                image_path = preview_dir / "keypoints.png"
                                cv2.imwrite(str(image_path), image)
                                print(f"[preview] Saved sample preview → {image_path}")
                            except Exception as exc:  # pragma: no cover - preview errors are non fatal
                                print(f"[preview] Failed to render keypoint preview: {exc}")

                        hypothesis: Optional[str] = None
                        hypothesis_error: Optional[str] = None
                        try:
                            hypothesis = _generate_sample_prediction(
                                sample,
                                adapter=adapter,
                                bridge=bridge,
                                decoder=decoder,
                                tokenizer=tok,
                                device=device,
                                cfg=cfg,
                            )
                        except Exception as exc:
                            hypothesis_error = str(exc)
                            print(f"[preview] Failed to generate sample hypothesis: {exc}")
                        metrics_dir_payload: Dict[str, object] = {}
                        metrics_values: Dict[str, float] = {}
                        metrics_path = preview_dir / "metrics.json"
                        if hypothesis is not None:
                            metrics_dir_payload = {
                                "id": str(sample["id"]),
                                "reference": sample_text,
                                "hypothesis": hypothesis,
                                "settings": {
                                    "max_new_tokens": cfg.sample_max_new_tokens,
                                    "num_beams": cfg.sample_num_beams,
                                    "lowercase": cfg.sample_metrics_lowercase,
                                    "metrics": list(cfg.sample_metrics),
                                },
                            }

                            reference_for_metrics = metrics_dir_payload["reference"]
                            prediction_for_metrics = hypothesis
                            if cfg.sample_metrics_lowercase:
                                reference_for_metrics = reference_for_metrics.lower()
                                prediction_for_metrics = prediction_for_metrics.lower()

                            if cfg.sample_metrics:
                                try:
                                    metrics_values = eval_compute_metrics(
                                        [
                                            EvalExample(
                                                clip_id=str(sample["id"]),
                                                reference=reference_for_metrics,
                                                prediction=prediction_for_metrics,
                                            )
                                        ],
                                        cfg.sample_metrics,
                                    )
                                except Exception as exc:
                                    print(f"[preview] Failed to compute metrics: {exc}")
                                else:
                                    metrics_dir_payload["metrics"] = metrics_values
                                    summary = ", ".join(f"{name}={value:.4f}" for name, value in metrics_values.items())
                                    if summary:
                                        print(f"[preview] Metrics → {summary}")
                                    else:
                                        print("[preview] Metrics → (none computed)")
                            else:
                                print("[preview] Metrics → (disabled)")
                            if "metrics" not in metrics_dir_payload:
                                metrics_dir_payload["metrics"] = metrics_values
                            metrics_path.write_text(
                                json.dumps(metrics_dir_payload, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8",
                            )
                            print(f"[preview] Saved metrics → {metrics_path}")
                        else:
                            failure_payload = {
                                "id": str(sample["id"]),
                                "reference": sample_text,
                                "error": hypothesis_error or "empty hypothesis",
                                "settings": {
                                    "max_new_tokens": cfg.sample_max_new_tokens,
                                    "num_beams": cfg.sample_num_beams,
                                    "lowercase": cfg.sample_metrics_lowercase,
                                    "metrics": list(cfg.sample_metrics),
                                },
                            }
                            metrics_path.write_text(
                                json.dumps(failure_payload, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8",
                            )
                            print(f"[preview] Saved metrics (generation failed) → {metrics_path}")

        if accum_counter != 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            accum_counter = 0

    # Final checkpoint
    if accum_counter != 0:
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        accum_counter = 0

    path = cfg.out_dir / "checkpoints" / f"adapter_final.pt"
    ckpt = {
        "adapter": adapter.state_dict(),
        "bridge": bridge.state_dict(),
        "lora": get_peft_model_state_dict(decoder),
        "lora_config": _serialise_lora_config(lora_config),
        "config": _serialise_train_config(cfg),
        "d_model": d_model,
        "num_landmarks": N,
    }
    torch.save(ckpt, path)
    print(f"[done] Saved final checkpoint → {path}")


def parse_args() -> TrainConfig:
    ap = argparse.ArgumentParser(description="SONAR‑SLT fine‑tuning on keypoints (LSA→es)")
    ap.add_argument("--keypoints-dir", type=Path, default=Path("single_signer/kp/keypoints"))
    ap.add_argument("--csv", type=Path, default=Path("meta_2.csv"))
    ap.add_argument("--out", type=Path, default=Path("runs/sonar_slt_finetune_lsa"))
    ap.add_argument(
        "--dev-csv",
        type=Path,
        default=None,
        help="Optional held-out CSV with the same schema for validation",
    )
    ap.add_argument("--video-dir", type=Path, default=None)
    ap.add_argument("--model-name", type=str, default="mtmlt/sonar-nllb-200-1.3B")
    ap.add_argument("--tgt-lang", type=str, default="spa_Latn")
    ap.add_argument("--T", type=int, default=DEFAULT_T)
    ap.add_argument("--clip-frames", type=int, default=DEFAULT_CLIP_FRAMES)
    ap.add_argument("--frame-size", type=int, default=DEFAULT_IMG_SIZE)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bridge-lr", type=float, default=None)
    ap.add_argument("--lora-lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        help=(
            "Compute device (e.g. 'auto', 'cpu', 'cpu:0', 'cuda', 'cuda:1', 'mps'). "
            "When set to 'auto', CUDA is preferred, then MPS, then CPU."
        ),
    )
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument(
        "--eval-every",
        type=int,
        default=500,
        help="Evaluate on --dev-csv every N optimizer steps",
    )
    ap.add_argument("--mse-weight", type=float, default=7000.0)
    ap.add_argument("--cos-weight", type=float, default=2.7)
    ap.add_argument("--lam-sem", type=float, default=1.0)
    ap.add_argument("--lam-ce", type=float, default=0.1)
    ap.add_argument("--lam-ae", type=float, default=0.2)
    ap.add_argument("--lam-nce", type=float, default=0.0)
    ap.add_argument("--nce-temperature", type=float, default=0.07)
    ap.add_argument("--vit-model", type=str, default="vit_base_patch16_224")
    ap.add_argument("--vit-checkpoint", type=Path, default=None)
    ap.add_argument("--videomae-model", type=str, default="MCG-NJU/videomae-base")
    ap.add_argument("--videomae-checkpoint", type=Path, default=None)
    ap.add_argument("--freeze-keypoint", action="store_true")
    ap.add_argument("--freeze-vit", action="store_true")
    ap.add_argument("--freeze-videomae", action="store_true")
    ap.add_argument("--text-pool-layers", type=int, default=4)
    ap.add_argument("--save-samples", action="store_true", help="Store preview samples alongside checkpoints")
    ap.add_argument("--preview-width", type=int, default=640)
    ap.add_argument("--preview-height", type=int, default=640)
    ap.add_argument(
        "--preview-confidence-threshold",
        type=float,
        default=0.2,
        help="Minimum confidence required to draw a keypoint in previews",
    )
    ap.add_argument(
        "--preview-seed",
        type=int,
        default=1337,
        help="Random seed used to pick preview samples",
    )
    ap.add_argument(
        "--sample-metrics",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Metrics to compute for saved preview samples. "
            "Defaults to tools_eval_sonar_slt.DEFAULT_METRICS when omitted."
        ),
    )
    ap.add_argument(
        "--sample-max-new-tokens",
        type=int,
        default=64,
        help="Maximum number of tokens to generate for preview hypotheses",
    )
    ap.add_argument(
        "--sample-num-beams",
        type=int,
        default=4,
        help="Number of beams to use when generating preview hypotheses",
    )
    ap.add_argument(
        "--sample-metrics-lowercase",
        action="store_true",
        help="Lowercase preview references and hypotheses before computing metrics",
    )
    args = ap.parse_args()
    return TrainConfig(
        keypoints_dir=args.keypoints_dir,
        csv=args.csv,
        out_dir=args.out,
        dev_csv=args.dev_csv,
        video_dir=args.video_dir,
        model_name=args.model_name,
        tgt_lang=args.tgt_lang,
        T=args.T,
        clip_frames=args.clip_frames,
        frame_size=args.frame_size,
        batch_size=args.batch_size,
        accum=args.accum,
        epochs=args.epochs,
        lr=args.lr,
        bridge_lr=args.bridge_lr,
        lora_lr=args.lora_lr,
        weight_decay=args.weight_decay,
        device=args.device,
        half=bool(args.half),
        save_every=args.save_every,
        log_every=args.log_every,
        eval_every=args.eval_every,
        mse_weight=args.mse_weight,
        cos_weight=args.cos_weight,
        lam_sem=args.lam_sem,
        lam_ce=args.lam_ce,
        lam_ae=args.lam_ae,
        lam_nce=args.lam_nce,
        nce_temperature=args.nce_temperature,
        text_pool_layers=args.text_pool_layers,
        vit_model=args.vit_model,
        vit_checkpoint=args.vit_checkpoint,
        videomae_model=args.videomae_model,
        videomae_checkpoint=args.videomae_checkpoint,
        freeze_keypoint=bool(args.freeze_keypoint),
        freeze_vit=bool(args.freeze_vit),
        freeze_videomae=bool(args.freeze_videomae),
        save_samples=bool(args.save_samples),
        preview_width=int(args.preview_width),
        preview_height=int(args.preview_height),
        preview_confidence_threshold=float(args.preview_confidence_threshold),
        preview_seed=int(args.preview_seed),
        sample_metrics=(
            tuple(args.sample_metrics)
            if args.sample_metrics is not None
            else DEFAULT_METRICS
        ),
        sample_max_new_tokens=int(args.sample_max_new_tokens),
        sample_num_beams=int(args.sample_num_beams),
        sample_metrics_lowercase=bool(args.sample_metrics_lowercase),
    )


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
