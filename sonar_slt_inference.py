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
  ViT image encoder together with a VideoMAE spatio-temporal backbone. Both
  modules default to publicly available pre-trained weights and can be
  overridden through checkpoints.
* Keypoints and dual visual features are processed through temporal
  Transformers and a SpaMo fusion block. The fused representation is projected
  onto the 1024-dim SONAR space and fed directly into the decoder as a single
  "pseudo token" sequence.

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
from transformers import (
    AutoTokenizer,
    BaseModelOutput,
    M2M100ForConditionalGeneration,
)

from peft import LoraConfig, TaskType, get_peft_model, set_peft_model_state_dict

try:  # pragma: no cover - optional dependency
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore[assignment]

from slt.utils.visual_adapter import (
    AdapterConfig,
    KEYPOINT_CHANNELS,
    VisualFusionAdapter,
)

from models.text_pooler import TextPooler

LOGGER = logging.getLogger("sonar_slt_inference")

VIDEO_EXTENSIONS = (".mp4", ".mkv", ".mov", ".avi", ".webm")
KEYPOINT_TOTAL_LANDMARKS = 33 + 468 + 2 * 21  # pose + face + 2 hands
KEYPOINT_CHANNELS = 3  # (x, y, confidence)
DEFAULT_FRAME_SIZE = 224
DEFAULT_NUM_FRAMES = 16
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


# Adapter configuration is imported from ``slt.utils.visual_adapter``

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


def _crop_with_offset(array: np.ndarray, target_frames: int, offset: float) -> np.ndarray:
    if array.shape[0] <= target_frames:
        return array
    offset = float(np.clip(offset, 0.0, 1.0))
    max_start = max(array.shape[0] - target_frames, 0)
    start = int(round(max_start * offset))
    end = start + target_frames
    return array[start:end]


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
    clip_frames: int,
    clip_offset: Optional[float],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    kp = _pad_or_sample(keypoints, num_kp_frames, axis=0)
    video = None
    if frames is not None:
        clipped = frames
        if clip_offset is not None:
            clipped = _crop_with_offset(clipped, clip_frames, clip_offset)
        video = _pad_or_sample(clipped, clip_frames, axis=0)
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
    adapter: VisualFusionAdapter,
    samples: Sequence[Tuple[str, Optional[Path], Path]],
    generation_config: GenerationConfig,
    *,
    forced_bos_id: int,
    device: torch.device,
    num_kp_frames: int,
    clip_frames: int,
    clip_offset: Optional[float],
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
            keypoints,
            frames_array,
            num_kp_frames=num_kp_frames,
            clip_frames=clip_frames,
            clip_offset=clip_offset,
        )

        kp_tensor = (
            torch.from_numpy(keypoints)
            .unsqueeze(0)
            .to(device=device, dtype=torch.float32)
        )
        frames_tensor: Optional[torch.Tensor] = None
        if frames_array is not None:
            frames_tensor = (
                torch.from_numpy(frames_array)
                .permute(0, 3, 1, 2)
                .unsqueeze(0)
                .to(device=device, dtype=torch.float32)
            )

        with torch.inference_mode():
            embedding = adapter(kp_tensor, frames_tensor)
            encoder_outputs = BaseModelOutput(last_hidden_state=embedding.unsqueeze(1))
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


def _load_adapter_checkpoint(
    adapter: VisualFusionAdapter, checkpoint: Optional[Path]
) -> Tuple[
    Optional[Dict[str, torch.Tensor]],
    Optional[Dict],
    Optional[Dict[str, torch.Tensor]],
    Optional[Dict],
]:
    if checkpoint is None:
        LOGGER.warning("No adapter checkpoint provided; using random initialisation")
        return None, None, None, None

    state_dict = torch.load(checkpoint, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    adapter_state = state_dict
    bridge_state: Optional[Dict[str, torch.Tensor]] = None
    lora_state: Optional[Dict[str, torch.Tensor]] = None
    lora_config: Optional[Dict] = None
    train_config: Optional[Dict] = None

    if isinstance(state_dict, dict):
        if "adapter" in state_dict:
            adapter_state = state_dict["adapter"]
        if "bridge" in state_dict:
            bridge_state = state_dict["bridge"]
        if "lora" in state_dict:
            lora_state = state_dict["lora"]
        if "lora_config" in state_dict:
            lora_config = state_dict["lora_config"]
        if "config" in state_dict:
            train_config = state_dict["config"]

    adapter.load_state_dict(adapter_state, strict=False)
    return lora_state, lora_config, bridge_state, train_config


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
        "--clip-frames",
        type=int,
        default=DEFAULT_NUM_FRAMES,
        help="Number of RGB frames per clip passed to the visual backbones.",
    )
    parser.add_argument(
        "--num-keypoint-frames",
        type=int,
        default=DEFAULT_KEYPOINT_FRAMES,
        help="Number of keypoint frames after sampling/padding.",
    )
    parser.add_argument(
        "--clip-offset",
        type=float,
        default=None,
        help="Optional relative offset [0-1] for temporal cropping before sampling.",
    )
    parser.add_argument(
        "--vit-model",
        type=str,
        default="vit_base_patch16_224",
        help="timm ViT backbone identifier.",
    )
    parser.add_argument(
        "--vit-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint with fine-tuned ViT weights.",
    )
    parser.add_argument(
        "--videomae-model",
        type=str,
        default="MCG-NJU/videomae-base",
        help="Hugging Face identifier for the VideoMAE backbone.",
    )
    parser.add_argument(
        "--videomae-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint with fine-tuned VideoMAE weights.",
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

    samples = _discover_samples(args.videos_dir, args.keypoints_dir)
    if not samples:
        raise RuntimeError(
            "No samples found. Ensure keypoints are stored as .npz/.npy files in the directory."
        )

    first_keypoints = _load_keypoints(samples[0][2])
    num_landmarks = first_keypoints.shape[1]
    adapter_config = AdapterConfig()
    adapter = VisualFusionAdapter(
        adapter_config,
        num_points=num_landmarks,
        vit_name=args.vit_model,
        vit_checkpoint=args.vit_checkpoint,
        videomae_name=args.videomae_model,
        videomae_checkpoint=args.videomae_checkpoint,
    ).to(device)
    lora_state, lora_config_dict, _bridge_state, train_config = _load_adapter_checkpoint(
        adapter, args.adapter_checkpoint
    )

    LOGGER.info("Loading SONAR decoder model: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(args.model_name)
    if lora_state is not None:
        if lora_config_dict is not None:
            lora_config = LoraConfig(**lora_config_dict)
        else:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM,
                target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            )
        model = get_peft_model(model, lora_config)
        set_peft_model_state_dict(model, lora_state)
        LOGGER.info("Loaded LoRA weights into SONAR decoder")
    model = model.to(device)
    pool_layers = 4
    if train_config is not None:
        pool_layers = int(train_config.get("text_pool_layers", pool_layers))

    text_pooler = TextPooler(model, tokenizer, args.target_lang, num_layers=pool_layers)

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

    results = run_inference(
        model,
        tokenizer,
        adapter,
        samples,
        generation_config,
        device=device,
        forced_bos_id=text_pooler.bos_id,
        num_kp_frames=args.num_keypoint_frames,
        clip_frames=args.clip_frames,
        clip_offset=args.clip_offset,
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
