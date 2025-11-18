#!/usr/bin/env python3
"""
SONAR‑SLT: Inference script for the keypoint adapter fine‑tuned with
`tools_finetune_sonar_slt.py`.

Usage (Windows PowerShell example):

  python tools/infer_sonar_slt.py \
    --keypoints-dir data/single_signer/kp/keypoints \
    --csv meta.csv \
    --adapter-ckpt runs/sonar_slt_finetune_lsa/checkpoints/adapter_final.pt \
    --out runs/sonar_slt_infer \
    --video-dir data/single_signer/videos \
    --tgt-lang spa_Latn \
    --device auto

Notes
-----
- Expects (T, N, C) keypoints (MediaPipe Holistic by default) normalised as in training.
- Provide the checkpoint containing `adapter` and `bridge` weights from the fine‑tuning script.
- Multimodal checkpoints (those trained with ViT/VideoMAE weights) require paired RGB clips;
  pass `--video-dir` (and optionally `--clip-frames` / `--frame-size`) to mirror training.
- Uses the SONAR HF port `mtmlt/sonar-nllb-200-1.3B` by default (matching training).
- Bridge checkpoints remain compatible; `load_bridge` infers the correct shape from saved weights.
- Override the decoder with `--sonar-model-dir /path/or/repo` if you have a local copy.
- Designed to run on CPU or CUDA; use --half on capable GPUs.

Dependencies
------------
- torch, transformers >= 4.42, numpy

Output
------
- A CSV: {out}/preds.csv with columns: id,video,lang,text
- Per‑clip JSONL with extra diagnostics in {out}/logs.jsonl

"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path, WindowsPath
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from tools_keypoint_utils import extract_confidence_channel

try:  # pragma: no cover - optional dependency for multimodal checkpoints
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore[assignment]

from slt.utils.visual_adapter import (
    AdapterConfig,
    FusionAdapter,
    KEYPOINT_CHANNELS,
    VisualFusionAdapter,
)

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        PreTrainedModel,
    )
except Exception as exc:  # pragma: no cover
    print("[FATAL] transformers not available:", exc)
    raise

try:  # pragma: no cover - optional dependency
    from peft import LoraConfig, TaskType, get_peft_model, set_peft_model_state_dict
except Exception:  # pragma: no cover - best effort import when PEFT is unavailable
    LoraConfig = None  # type: ignore[assignment]
    TaskType = None  # type: ignore[assignment]
    get_peft_model = None  # type: ignore[assignment]
    set_peft_model_state_dict = None  # type: ignore[assignment]

from sonar_generation import generate_from_hidden_states

# -----------------------------
# Utilities
# -----------------------------

LOGGER = logging.getLogger("tools_infer_sonar_slt")
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO)
else:
    LOGGER.setLevel(logging.INFO)
    if not LOGGER.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setLevel(logging.INFO)
        LOGGER.addHandler(handler)
    LOGGER.propagate = False

DEFAULT_SONAR_MODEL_NAME = "mtmlt/sonar-nllb-200-1.3B"
DEFAULT_TGT_LANG = "spa_Latn"
DEFAULT_CLIP_FRAMES = 16
DEFAULT_FRAME_SIZE = 224
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".mov", ".avi", ".webm")


def _coerce_path(value: Optional[Any]) -> Optional[Path]:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        return Path(cleaned)
    return None


def _maybe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _maybe_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return None


def adapter_requires_video(state: Dict[str, Any]) -> bool:
    for key in state.keys():
        if isinstance(key, str) and (key.startswith("vit.") or key.startswith("videomae.")):
            return True
    return False


def reshape_flat_keypoints(
    kp: np.ndarray, num_landmarks: int, *, expected_channels: int = KEYPOINT_CHANNELS
) -> np.ndarray:
    """Convert flattened (T, F) MediaPipe dumps into (T, N, C) arrays.

    The heuristic mirrors the training pipeline: prefer the checkpoint's declared
    ``num_landmarks`` when the feature dimension is a multiple of that count, fall
    back to 543 (MediaPipe Holistic default), and as a last resort derive the
    landmark count from the expected channel size.

    When more than ``expected_channels`` are present per landmark (e.g., x/y/z/conf),
    the surplus channels are dropped after reshaping so that downstream code always
    observes (x, y, conf).
    """

    if kp.ndim != 2:
        return kp

    feat_dim = kp.shape[1]

    candidate_points = None
    if num_landmarks > 0 and feat_dim % num_landmarks == 0:
        candidate_points = num_landmarks
    elif feat_dim % 543 == 0:
        candidate_points = 543
    elif feat_dim % expected_channels == 0:
        candidate_points = feat_dim // expected_channels

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
        LOGGER.debug(
            "Dropping %d surplus keypoint channel(s) (keeping x/y and the last channel)",
            inferred_channels - expected_channels,
        )
        if expected_channels == 3 and inferred_channels >= 3:
            coords = reshaped[..., :2]
            confidence = reshaped[..., -1:]
            reshaped = np.concatenate([coords, confidence], axis=-1)
        else:
            reshaped = reshaped[..., :expected_channels]
    return reshaped


def normalise_keypoints(arr: np.ndarray) -> np.ndarray:
    """Center by mean landmark and scale to unit max-radius; keep confidence.
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


def pad_or_sample(
    array: np.ndarray,
    target_frames: int,
    axis: int = 0,
    *,
    stochastic: bool = False,
    random_window: bool = False,
    frame_jitter: bool = False,
    rng: Optional[random.Random] = None,
) -> np.ndarray:
    length = array.shape[axis]
    if length == target_frames:
        return array
    if length == 0:
        raise ValueError("Cannot pad/sample an empty keypoint sequence")
    if length > target_frames:
        rng_impl = rng if rng is not None else random  # type: ignore[assignment]
        if stochastic and random_window:
            max_start = max(length - target_frames, 0)
            start = rng_impl.randint(0, max_start) if max_start > 0 else 0
            idxs = np.arange(start, start + target_frames, dtype=int)
            return np.take(array, idxs, axis=axis)
        if stochastic and frame_jitter:
            fractions = np.linspace(0.0, 1.0, num=target_frames, endpoint=True, dtype=np.float64)
            phase = rng_impl.random()
            fractions = (fractions + phase) % 1.0
            fractions.sort()
            idxs = np.rint(fractions * (length - 1)).astype(int)
            return np.take(array, idxs, axis=axis)
        idxs = np.linspace(0, length - 1, num=target_frames)
        idxs = np.rint(idxs).astype(int)
        return np.take(array, idxs, axis=axis)
    pad_count = target_frames - length
    pad_idx = np.full(pad_count, length - 1, dtype=int)
    pad_values = np.take(array, pad_idx, axis=axis)
    return np.concatenate([array, pad_values], axis=axis)


def _resolve_device(name: str) -> torch.device:
    cleaned = (name or "").strip()
    if cleaned.lower() == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():  # mac
            return torch.device("mps")
        return torch.device("cpu")
    try:
        device = torch.device(cleaned)
    except (TypeError, ValueError, RuntimeError):
        lowered = cleaned.lower()
        if lowered in {"cuda", "gpu"} and torch.cuda.is_available():
            return torch.device("cuda")
        if lowered == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if device.type == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    return device


# -----------------------------
# Adapter (keypoints → SONAR space)
# -----------------------------


@dataclass
# -----------------------------
# SONAR decoder wrapper
# -----------------------------
class SonarDecoder:
    """Thin wrapper around the SONAR HF port, conditioned on a semantic vector z."""

    def __init__(
        self,
        model_dir: Optional[str],
        device: torch.device,
        half: bool = False,
        *,
        d_model: Optional[int] = None,
        lora_state: Optional[Dict[str, torch.Tensor]] = None,
        lora_config: Optional[Dict[str, Any]] = None,
    ):
        self.device = device
        # Matches TrainConfig.model_name to ensure existing bridge checkpoints load without changes.
        default_model = DEFAULT_SONAR_MODEL_NAME

        def _clean_path(path: Optional[str]) -> Optional[str]:
            if path is None:
                return None
            cleaned = path.strip()
            if not cleaned or cleaned in {os.sep, "\\"}:
                return None
            return os.path.expanduser(cleaned)

        model_id = _clean_path(model_dir) or default_model
        try:
            self.model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
        except (OSError, ValueError) as exc:
            if model_id != default_model:
                print(
                    f"[warn] Failed to load '{model_id}' ({exc}); falling back to '{default_model}'"
                )
                model_id = default_model
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
            else:
                raise
        self._use_half = bool(half and device.type == "cuda")
        if self._use_half:
            self.model.half()
        self.model.eval()
        self.tok = AutoTokenizer.from_pretrained(model_id)

        model_d_model = getattr(self.model.config, "d_model", None) or getattr(
            self.model.config,
            "hidden_size",
            1024,
        )
        self.d_model = d_model or model_d_model
        self.bridge: Optional[nn.Linear] = None
        self._lora_applied = False
        self._apply_lora_if_available(lora_state, lora_config)

    def _apply_lora_if_available(
        self,
        lora_state: Optional[Dict[str, torch.Tensor]],
        lora_config: Optional[Dict[str, Any]],
    ) -> None:
        if lora_state is None:
            if lora_config is not None:
                LOGGER.warning(
                    "LoRA config was provided without weights; skipping decoder LoRA application."
                )
            return
        if get_peft_model is None or set_peft_model_state_dict is None or LoraConfig is None:
            raise RuntimeError(
                "LoRA weights were provided but the PEFT dependency is unavailable."
            )
        if not isinstance(lora_state, dict):
            raise TypeError("LoRA state must be a state dict mapping parameter names to tensors")
        if lora_config is not None and not isinstance(lora_config, dict):
            raise TypeError("LoRA config must be a mapping when provided")
        if lora_config is not None:
            lora_cfg = LoraConfig(**lora_config)
        else:
            if TaskType is None:
                raise RuntimeError(
                    "LoRA weights were provided without a config and TaskType is unavailable."
                )
            LOGGER.info("LoRA config missing from checkpoint; using default SONAR settings.")
            lora_cfg = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM,
                target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            )
        LOGGER.info("Initialising SONAR decoder with LoRA adapters.")
        self.model = get_peft_model(self.model, lora_cfg)
        set_peft_model_state_dict(self.model, lora_state)
        self.model = self.model.to(self.device)
        if self._use_half:
            self.model.half()
        self.model.eval()
        self._lora_applied = True
        LOGGER.info("Loaded LoRA weights into SONAR decoder.")

    def load_bridge(self, bridge_state: Dict[str, torch.Tensor]) -> None:
        if not isinstance(bridge_state, dict):
            raise TypeError("Bridge state must be a state dict mapping strings to tensors")
        if "weight" not in bridge_state:
            raise KeyError("Bridge state dict must include a 'weight' tensor")
        weight = bridge_state["weight"]
        if not torch.is_tensor(weight):
            raise TypeError("Bridge 'weight' must be a tensor")
        out_features, in_features = weight.shape
        self.d_model = out_features
        self.bridge = nn.Linear(in_features, out_features, bias=False).to(self.device)
        missing, unexpected = self.bridge.load_state_dict(bridge_state, strict=False)
        if missing:
            raise RuntimeError(f"Bridge checkpoint is missing parameters: {sorted(missing)}")
        if unexpected:
            raise RuntimeError(f"Bridge checkpoint has unexpected parameters: {sorted(unexpected)}")
        if self._use_half:
            self.bridge.half()

    @torch.no_grad()
    def generate(
        self,
        z_bD: torch.Tensor,
        tgt_lang: str,
        *,
        generation_options: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        if self.bridge is None:
            raise RuntimeError("Bridge weights have not been loaded. Call load_bridge() before generation")
        # Build pseudo encoder outputs from z as a single‑token memory
        B, D = z_bD.shape
        z = z_bD.to(self.bridge.weight.device)
        z = z.to(self.bridge.weight.dtype)
        mem = self.bridge(z).unsqueeze(1)  # [B,1,d_model]
        options = dict(generation_options) if generation_options else None
        return generate_from_hidden_states(
            self.model,
            self.tok,
            mem,
            tgt_lang,
            generation_options=options,
        )


# -----------------------------
# End‑to‑end model
# -----------------------------
class SonarSLTInference(nn.Module):
    def __init__(self, fusion_adapter: nn.Module, *, expects_frames: bool) -> None:
        super().__init__()
        self.fusion_adapter = fusion_adapter
        self.expects_frames = expects_frames

    def _adapter_param_dtype(self) -> torch.dtype:
        for param in self.fusion_adapter.parameters():
            return param.dtype
        return torch.float32

    def load_adapter(self, adapter_state: Dict[str, torch.Tensor]) -> None:
        if not isinstance(adapter_state, dict):
            raise TypeError("Adapter state must be a state dict mapping strings to tensors")

        state_dict: Dict[str, torch.Tensor] = dict(adapter_state)
        expected_keys = set(self.fusion_adapter.state_dict().keys())
        expects_keypoint_bias = "keypoint_bias" in expected_keys
        expects_modality_embeddings = any(
            key == "fusion.modality_embeddings"
            or key.startswith("fusion.modality_embeddings.")
            for key in expected_keys
        )

        def _extract_bias_candidate(value: Any) -> Optional[torch.Tensor]:
            candidate = None
            if hasattr(value, "ndim") and getattr(value, "ndim") and getattr(value, "ndim") >= 1:
                try:
                    candidate = value[0]
                except Exception:  # pragma: no cover - defensive
                    candidate = None
            elif hasattr(value, "__getitem__"):
                try:
                    candidate = value[0]
                except Exception:  # pragma: no cover - defensive
                    candidate = None
            return candidate

        keypoint_bias_value: Optional[torch.Tensor] = None

        has_kp_proj_attr = hasattr(self.fusion_adapter, "kp_proj")
        has_kp_proj = has_kp_proj_attr and any(key.startswith("kp_proj.") for key in state_dict)
        has_fusion_key_proj = any(key.startswith("fusion.key_proj.") for key in state_dict)

        if has_kp_proj_attr and not has_kp_proj and has_fusion_key_proj:
            converted_state: Dict[str, torch.Tensor] = {}
            drop_prefixes = (
                "fusion.key_proj.",
                "fusion.modality_embeddings",
                "fusion.spatial_proj",
                "fusion.motion_proj",
            )
            for key, value in state_dict.items():
                if key.startswith("fusion.key_proj."):
                    suffix = key.split(".", 2)[-1]
                    converted_state[f"kp_proj.{suffix}"] = value
                    continue
                if key.startswith("fusion.modality_embeddings"):
                    if keypoint_bias_value is None:
                        keypoint_bias_value = _extract_bias_candidate(value)
                    continue
                if key.startswith(drop_prefixes):
                    continue
                converted_state[key] = value
            state_dict = converted_state

        if expects_keypoint_bias and not expects_modality_embeddings:
            modality_keys = [
                key
                for key in state_dict
                if key == "fusion.modality_embeddings"
                or key.startswith("fusion.modality_embeddings.")
            ]
            for key in modality_keys:
                value = state_dict.pop(key)
                if keypoint_bias_value is not None:
                    continue
                candidate = _extract_bias_candidate(value)
                if candidate is not None:
                    keypoint_bias_value = candidate
        filtered_state: Dict[str, torch.Tensor] = {}
        filtered_out: List[str] = []
        for key, value in state_dict.items():
            if key in expected_keys:
                filtered_state[key] = value
            else:
                filtered_out.append(key)

        if keypoint_bias_value is not None and "keypoint_bias" in expected_keys:
            filtered_state["keypoint_bias"] = keypoint_bias_value

        if filtered_out:
            filtered_out_sorted = sorted(filtered_out)
            if len(filtered_out_sorted) <= 20:
                LOGGER.info(
                    "Ignoring adapter parameters not used during inference: %s",
                    filtered_out_sorted,
                )
            else:
                preview = ", ".join(filtered_out_sorted[:5])
                LOGGER.info(
                    "Ignoring %d adapter parameter(s) not used during inference (showing first 5: %s)",
                    len(filtered_out_sorted),
                    preview,
                )
                LOGGER.debug(
                    "Full list of adapter parameters ignored during inference: %s",
                    filtered_out_sorted,
                )

        missing, unexpected = self.fusion_adapter.load_state_dict(filtered_state, strict=False)
        if missing:
            raise RuntimeError(f"Adapter checkpoint is missing parameters: {sorted(missing)}")
        if unexpected:
            unexpected_sorted = sorted(unexpected)
            LOGGER.warning(
                "Adapter checkpoint has additional parameters even after filtering: %s",
                unexpected_sorted,
            )

    @torch.no_grad()
    def forward(
        self,
        keypoints_btnc: torch.Tensor,
        frames_btnchw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if keypoints_btnc.ndim != 4:
            raise ValueError(f"Expected keypoints (B,T,N,C), got {keypoints_btnc.shape}")
        dtype = self._adapter_param_dtype()
        kp = keypoints_btnc.to(dtype)
        frames: Optional[torch.Tensor] = None
        if frames_btnchw is not None:
            if frames_btnchw.ndim != 5:
                raise ValueError(
                    f"Expected frames shaped (B,T,C,H,W), got {frames_btnchw.shape}"
                )
            frames = frames_btnchw.to(dtype)
        if self.expects_frames:
            if frames is None:
                raise ValueError("Multimodal adapter requires frames tensor during inference")
            return self.fusion_adapter(kp, frames)
        return self.fusion_adapter(kp)


# -----------------------------
# Data plumbing (meta.csv + file resolution)
# -----------------------------
@dataclass
class Clip:
    clip_id: str
    keypoints_path: Path
    video_name: Optional[str] = None
    video_path: Optional[Path] = None


def _normalise_candidate(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


class VideoClipLoader:
    def __init__(
        self,
        video_dir: Path,
        *,
        clip_frames: int,
        frame_size: int,
        random_frame_sampling: bool = False,
        frame_jitter: bool = False,
    ) -> None:
        if clip_frames <= 0:
            raise ValueError("clip_frames must be positive")
        if frame_size <= 0:
            raise ValueError("frame_size must be positive")
        if cv2 is None:
            raise RuntimeError(
                "OpenCV (cv2) is required to load videos for multimodal checkpoints."
            )
        self.video_dir = video_dir
        self.clip_frames = int(clip_frames)
        self.frame_size = int(frame_size)
        self.random_frame_sampling = bool(random_frame_sampling)
        self.frame_jitter = bool(frame_jitter)
        self._rng = random.Random()

    def resolve(self, clip: Clip) -> Optional[Path]:
        candidates: List[str] = []

        def push(raw: Optional[str]) -> None:
            value = _normalise_candidate(raw)
            if value is None:
                return
            if value not in candidates:
                candidates.append(value)

        push(clip.video_name)
        push(clip.clip_id)
        if clip.video_name:
            raw_path = Path(clip.video_name)
            if raw_path.suffix:
                push(raw_path.stem)
                push(raw_path.stem.lower())
        push(clip.clip_id.lower())

        for name in candidates:
            literal = self.video_dir / name
            if literal.exists():
                return literal
            path_name = Path(name)
            if path_name.suffix:
                continue
            for ext in VIDEO_EXTENSIONS:
                candidate = self.video_dir / f"{name}{ext}"
                if candidate.exists():
                    return candidate
        return None

    def load(self, clip: Clip) -> np.ndarray:
        path = self.resolve(clip)
        if path is None:
            raise FileNotFoundError(
                f"Video not found for id={clip.clip_id} in {self.video_dir}"
            )
        clip.video_path = path
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():  # pragma: no cover - depends on codec support
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
        stochastic_sampling = self.random_frame_sampling or self.frame_jitter
        arr = pad_or_sample(
            arr,
            self.clip_frames,
            axis=0,
            stochastic=stochastic_sampling,
            random_window=self.random_frame_sampling,
            frame_jitter=self.frame_jitter,
            rng=self._rng,
        )
        return arr


def load_meta(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig") as fh:
        rdr = csv.DictReader(fh, delimiter=";")
        rows = [ {k: (v.strip() if isinstance(v, str) else v) for k,v in row.items()} for row in rdr ]
    if not rows:
        raise ValueError(f"CSV has no rows: {csv_path}")
    return rows


def resolve_clips(rows: List[Dict[str, str]], keypoints_dir: Optional[Path]) -> List[Clip]:
    if keypoints_dir is None or not keypoints_dir.exists():
        raise FileNotFoundError("--keypoints-dir must point to existing .npz/.npy files")
    out: List[Clip] = []
    for row in rows:
        clip_id_raw = row.get("id") or row.get("video_id") or row.get("video")
        if not clip_id_raw:
            raise ValueError("meta.csv must contain an 'id' or 'video' column")

        clip_id = clip_id_raw.strip() if isinstance(clip_id_raw, str) else str(clip_id_raw)
        clip_path = Path(clip_id)
        base_path = keypoints_dir / clip_path

        kp_path = None
        candidates = [base_path]
        for ext in (".npz", ".npy"):
            cand = base_path.with_suffix(ext)
            if cand not in candidates:
                candidates.append(cand)

        for cand in candidates:
            if cand.exists():
                kp_path = cand
                break
        if kp_path is None:
            raise FileNotFoundError(
                f"Keypoints not found for id={clip_id} in {keypoints_dir}"
            )

        video_name = row.get("video") or row.get("file") or kp_path.stem
        out.append(Clip(clip_id=clip_id, keypoints_path=kp_path, video_name=video_name))
    return out


def load_keypoints_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=True) as data:
            if "keypoints" in data:
                arr = data["keypoints"]
            elif "frames" in data:
                arr = data["frames"]
            else:
                raise KeyError(f"{path} lacks 'keypoints' array")
            return np.asarray(arr)
    return np.load(path)


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="SONAR‑SLT inference: keypoints → semantic decoding")
    ap.add_argument(
        "--keypoints-dir",
        type=Path,
        required=True,
        help=(
            "Directory with MediaPipe keypoint arrays (.npy/.npz). Flattened dumps "
            "((T, N*C)) are reshaped automatically and extra channels beyond x/y/conf "
            "are dropped."
        ),
    )
    ap.add_argument(
        "--video-dir",
        type=Path,
        default=None,
        help=(
            "Directory with RGB clips when using multimodal checkpoints (ViT/VideoMAE)."
            " Frames are resized to --frame-size and sampled/padded to --clip-frames."
        ),
    )
    ap.add_argument(
        "--adapter-ckpt",
        type=Path,
        required=True,
        help="Checkpoint from tools_finetune_sonar_slt.py (required when using keypoints)",
    )
    ap.add_argument("--csv", type=Path, required=True, help="meta.csv with an 'id' or 'video' column")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--tgt-lang",
        type=str,
        default=None,
        help=(
            "NLLB language token. Defaults to the checkpoint's target language when available,"
            " otherwise 'spa_Latn'."
        ),
    )
    ap.add_argument("--T", type=int, default=128, help="Temporal length for keypoints")
    ap.add_argument(
        "--clip-frames",
        type=int,
        default=DEFAULT_CLIP_FRAMES,
        help="Number of RGB frames per clip provided to the visual backbones",
    )
    ap.add_argument(
        "--frame-size",
        type=int,
        default=DEFAULT_FRAME_SIZE,
        help="Square spatial resolution (pixels) applied when resizing video frames",
    )
    ap.add_argument(
        "--random-frame-sampling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable stochastic contiguous crops when temporal length exceeds T/clip frames."
            " Disable via --no-random-frame-sampling."
        ),
    )
    ap.add_argument(
        "--frame-jitter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply phase jitter to evenly-spaced sampling for long sequences."
            " Disable via --no-frame-jitter."
        ),
    )
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling instead of pure beam search",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for sampling",
    )
    ap.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p nucleus sampling threshold",
    )
    ap.add_argument("--top-k", type=int, default=50, help="Top-k sampling cutoff")
    ap.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Penalty applied to discourage token repetition",
    )
    ap.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=0,
        help="Prevent repeating n-grams of this size (0 disables the constraint)",
    )
    ap.add_argument(
        "--length-penalty",
        type=float,
        default=1.0,
        help="Length penalty applied during beam search",
    )
    ap.add_argument(
        "--ignore-lora",
        action="store_true",
        help="Skip loading decoder LoRA weights/config even if present in the checkpoint",
    )
    ap.add_argument(
        "--sonar-model-dir",
        type=str,
        default=None,
        help=(
            "Path or HF repo for SONAR decoder override. Defaults to the training checkpoint's"
            " model name when recorded, otherwise 'mtmlt/sonar-nllb-200-1.3B'."
        ),
    )
    ap.add_argument(
        "--adapter-device",
        type=str,
        default="cpu",
        help="Device for loading the adapter checkpoint (e.g., 'cpu', 'cuda')",
    )
    args = ap.parse_args()
    cli_args = sys.argv[1:]
    tgt_lang_from_cli = any(arg.startswith("--tgt-lang") for arg in cli_args)
    sonar_model_from_cli = any(arg.startswith("--sonar-model-dir") for arg in cli_args)
    video_dir_from_cli = any(arg.startswith("--video-dir") for arg in cli_args)
    clip_frames_from_cli = any(arg.startswith("--clip-frames") for arg in cli_args)
    T_from_cli = any(arg == "--T" or arg.startswith("--T=") for arg in cli_args)
    frame_size_from_cli = any(arg.startswith("--frame-size") for arg in cli_args)
    random_sampling_from_cli = any(
        arg.startswith("--random-frame-sampling")
        or arg.startswith("--no-random-frame-sampling")
        for arg in cli_args
    )
    frame_jitter_from_cli = any(
        arg.startswith("--frame-jitter") or arg.startswith("--no-frame-jitter")
        for arg in cli_args
    )

    device = _resolve_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    # Load data index
    rows = load_meta(args.csv)
    clips = resolve_clips(rows, args.keypoints_dir)

    if not args.adapter_ckpt.is_file():
        raise FileNotFoundError(f"Adapter checkpoint not found: {args.adapter_ckpt}")
    adapter_map_device = _resolve_device(args.adapter_device)
    if hasattr(torch.serialization, "add_safe_globals"):
        safe_globals = [WindowsPath]
        try:
            from peft.utils.peft_types import TaskType, PeftType  # type: ignore
        except Exception:  # pragma: no cover - best effort import for torch safe loading
            TaskType = PeftType = None
        else:
            safe_globals.extend([TaskType, PeftType])
        torch.serialization.add_safe_globals(safe_globals)
    try:
        ckpt = torch.load(args.adapter_ckpt, map_location=adapter_map_device)
    except (RuntimeError, OSError, ValueError, EOFError) as exc:
        raise RuntimeError(f"Failed to load adapter checkpoint '{args.adapter_ckpt}': {exc}") from exc

    config_raw = ckpt.get("config")
    if config_raw is None:
        LOGGER.warning(
            "Checkpoint does not include a 'config' section; falling back to CLI/default decoder settings."
        )
        config: Dict[str, Any] = {}
    elif isinstance(config_raw, dict):
        config = dict(config_raw)
    else:
        raise TypeError("Checkpoint key 'config' must be a mapping when present")

    config_video_dir = _coerce_path(config.get("video_dir"))
    if not video_dir_from_cli and config_video_dir is not None:
        args.video_dir = config_video_dir
        LOGGER.info("Using video directory '%s' from checkpoint config.", args.video_dir)
    elif video_dir_from_cli and args.video_dir is not None and config_video_dir is not None:
        if args.video_dir != config_video_dir:
            LOGGER.warning(
                "CLI video directory '%s' differs from training path '%s'; proceeding with user override.",
                args.video_dir,
                config_video_dir,
            )

    config_clip_frames = _maybe_int(config.get("clip_frames"))
    if config_clip_frames is not None and not clip_frames_from_cli:
        args.clip_frames = config_clip_frames
        LOGGER.info("Using clip_frames=%d from checkpoint config.", args.clip_frames)

    config_frame_size = _maybe_int(config.get("frame_size"))
    if config_frame_size is not None and not frame_size_from_cli:
        args.frame_size = config_frame_size
        LOGGER.info("Using frame_size=%d from checkpoint config.", args.frame_size)

    config_T = _maybe_int(config.get("T"))
    if config_T is not None and config_T > 0 and not T_from_cli:
        args.T = config_T
        LOGGER.info("Using T=%d from checkpoint config.", args.T)

    config_random_sampling = _maybe_bool(config.get("random_frame_sampling"))
    if config_random_sampling is not None and not random_sampling_from_cli:
        args.random_frame_sampling = bool(config_random_sampling)
        LOGGER.info(
            "Using random_frame_sampling=%s from checkpoint config.",
            args.random_frame_sampling,
        )

    config_frame_jitter = _maybe_bool(config.get("frame_jitter"))
    if config_frame_jitter is not None and not frame_jitter_from_cli:
        args.frame_jitter = bool(config_frame_jitter)
        LOGGER.info(
            "Using frame_jitter=%s from checkpoint config.",
            args.frame_jitter,
        )

    config_multimodal = _maybe_bool(config.get("multimodal_adapter"))

    if args.video_dir is not None:
        args.video_dir = Path(args.video_dir).expanduser()

    required_keys = {"adapter", "bridge", "num_landmarks"}
    missing_keys = [key for key in required_keys if key not in ckpt]
    if missing_keys:
        raise KeyError(
            f"Checkpoint '{args.adapter_ckpt}' is missing required keys: {', '.join(missing_keys)}"
        )
    adapter_state = ckpt["adapter"]
    bridge_state = ckpt["bridge"]
    if not isinstance(adapter_state, dict) or not isinstance(bridge_state, dict):
        raise TypeError("Checkpoint keys 'adapter' and 'bridge' must be state dicts")

    state_has_visual_backbones = adapter_requires_video(adapter_state)
    if config_multimodal is None:
        uses_video = state_has_visual_backbones
        if uses_video:
            LOGGER.info(
                "Checkpoint config missing 'multimodal_adapter'; detected ViT/VideoMAE weights via state dict."
            )
    else:
        uses_video = bool(config_multimodal)
        if uses_video and not state_has_visual_backbones:
            LOGGER.info(
                "Checkpoint config indicates a multimodal adapter; using VisualFusionAdapter even though the state dict lacks ViT/VideoMAE prefixes."
            )
        elif not uses_video and state_has_visual_backbones:
            LOGGER.warning(
                "Checkpoint config marks the adapter as keypoint-only but ViT/VideoMAE weights were found."
                " Visual weights will be ignored during inference."
            )

    lora_state = ckpt.get("lora")
    lora_config_dict = ckpt.get("lora_config")
    if lora_state is not None and not isinstance(lora_state, dict):
        raise TypeError("Checkpoint key 'lora' must be a state dict when present")
    if lora_config_dict is not None and not isinstance(lora_config_dict, dict):
        raise TypeError("Checkpoint key 'lora_config' must be a mapping when present")
    if args.ignore_lora and (lora_state is not None or lora_config_dict is not None):
        LOGGER.info("Ignoring decoder LoRA weights/config as requested via --ignore-lora.")
        lora_state = None
        lora_config_dict = None
    if lora_state is None:
        if lora_config_dict is not None:
            LOGGER.warning(
                "LoRA config found in checkpoint but no weights were provided; config will be ignored."
            )
            lora_config_dict = None
        else:
            LOGGER.info("No LoRA weights found in checkpoint; using base SONAR decoder.")
    else:
        LOGGER.info("LoRA weights detected in checkpoint.")
        if lora_config_dict is None:
            LOGGER.warning(
                "LoRA weights detected without a config; falling back to default SONAR LoRA settings."
            )

    config_model_name_raw = config.get("model_name")
    config_model_name: Optional[str]
    if isinstance(config_model_name_raw, str) and config_model_name_raw.strip():
        config_model_name = config_model_name_raw.strip()
    else:
        if config_model_name_raw is not None and not isinstance(config_model_name_raw, str):
            LOGGER.warning(
                "Ignoring non-string 'model_name' from checkpoint config: %r",
                config_model_name_raw,
            )
        config_model_name = None

    if not sonar_model_from_cli:
        if config_model_name:
            args.sonar_model_dir = config_model_name
            LOGGER.info("Using SONAR decoder '%s' from checkpoint config.", args.sonar_model_dir)
        else:
            args.sonar_model_dir = DEFAULT_SONAR_MODEL_NAME
            LOGGER.info(
                "Using default SONAR decoder '%s' (checkpoint config missing model name).",
                args.sonar_model_dir,
            )
    else:
        if args.sonar_model_dir:
            args.sonar_model_dir = args.sonar_model_dir.strip() or None
        if not args.sonar_model_dir:
            if config_model_name:
                args.sonar_model_dir = config_model_name
                LOGGER.info(
                    "Using SONAR decoder '%s' from checkpoint config (CLI value was empty).",
                    args.sonar_model_dir,
                )
            else:
                args.sonar_model_dir = DEFAULT_SONAR_MODEL_NAME
                LOGGER.info(
                    "Using default SONAR decoder '%s' (CLI value was empty).",
                    args.sonar_model_dir,
                )
        elif args.sonar_model_dir and config_model_name and args.sonar_model_dir != config_model_name:
            LOGGER.warning(
                "CLI SONAR decoder '%s' differs from training model '%s'; proceeding with user override.",
                args.sonar_model_dir,
                config_model_name,
            )
        elif args.sonar_model_dir:
            LOGGER.info("Using SONAR decoder '%s' from CLI.", args.sonar_model_dir)

    config_tgt_lang_raw = config.get("tgt_lang")
    config_tgt_lang: Optional[str]
    if isinstance(config_tgt_lang_raw, str) and config_tgt_lang_raw.strip():
        config_tgt_lang = config_tgt_lang_raw.strip()
    else:
        if config_tgt_lang_raw is not None and not isinstance(config_tgt_lang_raw, str):
            LOGGER.warning(
                "Ignoring non-string 'tgt_lang' from checkpoint config: %r",
                config_tgt_lang_raw,
            )
        config_tgt_lang = None

    if tgt_lang_from_cli and args.tgt_lang:
        args.tgt_lang = args.tgt_lang.strip()
        if args.tgt_lang:
            LOGGER.info("Using target language '%s' from CLI.", args.tgt_lang)
        else:
            if config_tgt_lang:
                args.tgt_lang = config_tgt_lang
                LOGGER.info(
                    "Using target language '%s' from checkpoint config (CLI value was empty).",
                    args.tgt_lang,
                )
            else:
                args.tgt_lang = DEFAULT_TGT_LANG
                LOGGER.info(
                    "Using default target language '%s' (CLI value was empty).",
                    args.tgt_lang,
                )
    else:
        if config_tgt_lang:
            args.tgt_lang = config_tgt_lang
            LOGGER.info("Using target language '%s' from checkpoint config.", args.tgt_lang)
        else:
            args.tgt_lang = DEFAULT_TGT_LANG
            LOGGER.info(
                "Using default target language '%s' (checkpoint config missing target language).",
                args.tgt_lang,
            )

    num_landmarks = int(ckpt.get("num_landmarks") or 0)
    if num_landmarks <= 0:
        raise ValueError("Checkpoint missing a valid 'num_landmarks'")
    d_model = ckpt.get("d_model")

    adapter_config = AdapterConfig()
    if uses_video:
        if VisualFusionAdapter is None:
            raise RuntimeError(
                "Checkpoint includes ViT/VideoMAE weights but VisualFusionAdapter dependencies are missing."
                " Install the project with visual extras to enable multimodal inference."
            )
        if args.video_dir is None:
            raise RuntimeError(
                "Checkpoint includes ViT/VideoMAE weights; please provide paired videos via --video-dir."
            )
        if not args.video_dir.exists():
            raise FileNotFoundError(
                f"Video directory not found: {args.video_dir}"
            )

        vit_model_name = config.get("vit_model")
        if not isinstance(vit_model_name, str) or not vit_model_name.strip():
            vit_model_name = "vit_base_patch16_224"
        else:
            vit_model_name = vit_model_name.strip()

        videomae_model_name = config.get("videomae_model")
        if not isinstance(videomae_model_name, str) or not videomae_model_name.strip():
            videomae_model_name = "MCG-NJU/videomae-base"
        else:
            videomae_model_name = videomae_model_name.strip()

        vit_checkpoint = _coerce_path(config.get("vit_checkpoint"))
        videomae_checkpoint = _coerce_path(config.get("videomae_checkpoint"))
        freeze_vit = _maybe_bool(config.get("freeze_vit"))
        freeze_videomae = _maybe_bool(config.get("freeze_videomae"))
        freeze_keypoint = _maybe_bool(config.get("freeze_keypoint"))

        adapter_module = VisualFusionAdapter(
            adapter_config,
            num_points=num_landmarks,
            vit_name=vit_model_name,
            vit_checkpoint=vit_checkpoint,
            freeze_vit=bool(freeze_vit),
            videomae_name=videomae_model_name,
            videomae_checkpoint=videomae_checkpoint,
            freeze_videomae=bool(freeze_videomae),
            freeze_keypoint=bool(freeze_keypoint),
        ).to(device)
        video_loader = VideoClipLoader(
            args.video_dir,
            clip_frames=args.clip_frames,
            frame_size=args.frame_size,
            random_frame_sampling=args.random_frame_sampling,
            frame_jitter=args.frame_jitter,
        )
    else:
        if args.video_dir is not None:
            LOGGER.info(
                "Ignoring --video-dir because checkpoint does not contain visual backbone weights."
            )
        adapter_module = FusionAdapter(adapter_config, num_landmarks).to(device)
        video_loader = None

    # Build adapter model
    model = SonarSLTInference(adapter_module, expects_frames=uses_video).to(device)
    model.load_adapter(adapter_state)
    if args.half and device.type == "cuda":
        model.half()
    model.eval()

    # SONAR decoder with bridge from checkpoint
    decoder = SonarDecoder(
        args.sonar_model_dir,
        device=device,
        half=args.half,
        d_model=d_model,
        lora_state=lora_state,
        lora_config=lora_config_dict,
    )
    decoder.load_bridge(bridge_state)

    preds_path = args.out / "preds.csv"
    logs_path = args.out / "logs.jsonl"

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "do_sample": args.do_sample,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "length_penalty": args.length_penalty,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }
    stochastic_sampling = args.random_frame_sampling or args.frame_jitter

    with preds_path.open("w", newline="", encoding="utf-8") as fh_csv, logs_path.open("w", encoding="utf-8") as fh_log:
        w = csv.writer(fh_csv)
        w.writerow(["id", "video", "lang", "text"])  # header

        for clip in clips:
            if not clip.keypoints_path.exists():
                raise FileNotFoundError(f"Missing keypoints for {clip.clip_id}: {clip.keypoints_path}")
            kp = load_keypoints_array(clip.keypoints_path)
            if kp.ndim == 2:
                kp = reshape_flat_keypoints(kp, num_landmarks)
            if kp.ndim != 3:
                raise ValueError(f"Keypoints must be (T,N,C), got {kp.shape}")
            if kp.shape[1] != num_landmarks:
                raise ValueError(
                    f"Checkpoint expects {num_landmarks} landmarks but found {kp.shape[1]} in {clip.keypoints_path}"
                )
            kp = kp.astype(np.float32, copy=False)
            kp = normalise_keypoints(kp)
            kp = pad_or_sample(
                kp,
                args.T,
                axis=0,
                stochastic=stochastic_sampling,
                random_window=args.random_frame_sampling,
                frame_jitter=args.frame_jitter,
            )
            kp_t = torch.from_numpy(kp).unsqueeze(0).to(device)
            if args.half and device.type == "cuda":
                kp_t = kp_t.half()
            else:
                kp_t = kp_t.float()

            coords = kp[..., :2]
            coords_min = float(coords.min())
            coords_max = float(coords.max())
            coords_std = float(coords.std())
            conf = kp[..., 2:]
            conf_min = float(conf.min())
            conf_max = float(conf.max())
            conf_std = float(conf.std())
            LOGGER.info(
                "[%s] keypoints coords min=%.4f max=%.4f std=%.4f | conf min=%.4f max=%.4f std=%.4f",
                clip.clip_id,
                coords_min,
                coords_max,
                coords_std,
                conf_min,
                conf_max,
                conf_std,
            )
            kp_stats = {
                "coords": {"min": coords_min, "max": coords_max, "std": coords_std},
                "conf": {"min": conf_min, "max": conf_max, "std": conf_std},
            }

            frames_tensor: Optional[torch.Tensor] = None
            frames_stats: Optional[Dict[str, float]] = None
            video_path_str: Optional[str] = None
            if video_loader is not None:
                frames_np = video_loader.load(clip)
                video_path_str = str(clip.video_path) if clip.video_path is not None else None
                frames_min = float(frames_np.min())
                frames_max = float(frames_np.max())
                frames_std = float(frames_np.std())
                LOGGER.info(
                    "[%s] video frames min=%.4f max=%.4f std=%.4f (shape=%s)",
                    clip.clip_id,
                    frames_min,
                    frames_max,
                    frames_std,
                    tuple(frames_np.shape),
                )
                frames_stats = {"min": frames_min, "max": frames_max, "std": frames_std}
                frames_tensor = torch.from_numpy(frames_np).unsqueeze(0)
                if args.half and device.type == "cuda":
                    frames_tensor = frames_tensor.half()
                else:
                    frames_tensor = frames_tensor.float()
                frames_tensor = frames_tensor.to(device)

            with torch.inference_mode():
                z = model(kp_t, frames_tensor)  # [1,1024]

            z = z.to(decoder.bridge.weight.device)
            z = z.to(decoder.bridge.weight.dtype)

            z_float = z.float()
            z_min = float(z_float.min().item())
            z_max = float(z_float.max().item())
            z_std = float(z_float.std(unbiased=False).item())
            z_norm = float(z_float.norm(dim=-1).mean().item())
            LOGGER.info(
                "[%s] z stats min=%.4f max=%.4f std=%.4f norm=%.4f",
                clip.clip_id,
                z_min,
                z_max,
                z_std,
                z_norm,
            )
            z_stats = {"min": z_min, "max": z_max, "std": z_std, "norm": z_norm}

            text = decoder.generate(
                z,
                tgt_lang=args.tgt_lang,
                generation_options=generation_kwargs,
            )[0]

            video_name = clip.video_name or clip.clip_id
            w.writerow([clip.clip_id, video_name, args.tgt_lang, text])
            fh_log.write(json.dumps({
                "id": clip.clip_id,
                "video": video_name,
                "keypoints": str(clip.keypoints_path),
                "video_path": video_path_str,
                "lang": args.tgt_lang,
                "text": text,
                "kp_stats": kp_stats,
                "z_stats": z_stats,
                "video_stats": frames_stats,
            }, ensure_ascii=False) + "\n")
            fh_log.flush()
            print(f"[ok] {clip.clip_id} → {text}")


if __name__ == "__main__":
    main()
