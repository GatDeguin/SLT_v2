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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
def normalise_keypoints(arr: np.ndarray) -> np.ndarray:
    """Center by mean landmark and scale to unit max‑radius; keep confidence.
    Expects (T, N, C≥2). Returns same shape (T, N, 3).
    """
    coords = arr[..., :2]
    conf = arr[..., 2:3] if arr.shape[-1] >= 3 else np.ones_like(coords[..., :1])
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
        if keypoints.ndim == 2:
            # reshape (T, N*C) → (T, N, C)
            N = 543 if keypoints.shape[1] % 543 == 0 else keypoints.shape[1] // KEYPOINT_CHANNELS
            keypoints = keypoints.reshape(-1, N, KEYPOINT_CHANNELS)
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
        for ext in VIDEO_EXTENSIONS:
            candidate = self.video_dir / f"{stem}{ext}"
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


# -----------------------------
# Model definitions live in `slt.utils.visual_adapter`
# -----------------------------

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
# Training loop
# -----------------------------
@dataclass
class TrainConfig:
    keypoints_dir: Path
    csv: Path
    out_dir: Path
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
    if cfg.half and device.type == "cuda":
        decoder.half()

    # Bridge maps adapter’s 1024 to decoder d_model if needed
    d_model = getattr(decoder.config, "d_model", None) or getattr(decoder.config, "hidden_size", DEFAULT_D_MODEL)
    bridge = nn.Linear(1024, d_model, bias=False).to(device)
    if cfg.half and device.type == "cuda":
        bridge.half()

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
    if cfg.half and device.type == "cuda":
        adapter.half()

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

    step = 0
    log_path = cfg.out_dir / "train_log.jsonl"
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

            with torch.amp.autocast("cuda", enabled=cfg.half and device.type == "cuda"):
                # --- Visual → semantic vector z (Eq. 1–3)
                z = adapter(kp, frames_tensor)                # [B,1024]
                mem_z = bridge(z).unsqueeze(1) # [B,1,d_model]

                # --- Text → sentence embedding s (Eq. 4 + SONAR pooling)
                s = text_pool.encode(texts).to(mem_z.dtype)  # [B,d_model]
                # Project to adapter space for L_sem
                # bring s to 1024 dim via inverse bridge (least‑squares via pseudo‑inverse)
                with torch.no_grad():
                    # Compute W^+ (pseudo‑inverse) once per step (small d_model); cache if needed
                    W = bridge.weight  # [d_model,1024]
                    W_f32 = W.to(torch.float32)
                    W_pinv_fp32 = torch.linalg.pinv(W_f32, rcond=1e-5)
                s_fp32 = s.to(torch.float32)
                s_1024_fp32 = torch.matmul(s_fp32, W_pinv_fp32.T).contiguous()
                s_1024 = s_1024_fp32.to(z.dtype)  # [B,1024]

                # --- Losses
                # L_sem = α * MSE(z, s_1024) + β * (1 - cos(z, s_1024))
                mse = F.mse_loss(z, s_1024)
                cos = 1.0 - F.cosine_similarity(z, s_1024, dim=-1).mean()
                L_sem = cfg.mse_weight * mse + cfg.cos_weight * cos

                # L_ce — teacher forcing on target Spanish from z
                tok.src_lang = cfg.tgt_lang
                batch = tok(texts, return_tensors="pt", padding=True).to(device)
                labels = batch["input_ids"].clone()
                pad_token_id = getattr(tok, "pad_token_id", None)
                if pad_token_id is not None:
                    labels[labels == pad_token_id] = -100
                # In seq2seq HF, we pass encoder_outputs + labels
                enc_out_z = BaseModelOutput(last_hidden_state=mem_z)
                out_z = decoder(encoder_outputs=enc_out_z, labels=labels)
                L_ce = out_z.loss

                # L_ae — auto‑encoding from text embedding s
                mem_s = s.unsqueeze(1)  # [B,1,d_model]
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
                    "step": step,
                    "epoch": epoch,
                    "loss": float(loss.detach().cpu()),
                    "L_sem": float(L_sem.detach().cpu()),
                    "mse": float(mse.detach().cpu()),
                    "cos": float(cos.detach().cpu()),
                    "L_ce": float(L_ce.detach().cpu()),
                    "L_ae": float(L_ae.detach().cpu()),
                    "L_nce": float(L_nce.detach().cpu()),
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

            if step % cfg.save_every == 0:
                ckpt = {
                    "adapter": adapter.state_dict(),
                    "bridge": bridge.state_dict(),
                    "lora": get_peft_model_state_dict(decoder),
                    "lora_config": lora_config.to_dict(),
                    "config": cfg.__dict__,
                    "d_model": d_model,
                    "num_landmarks": N,
                }
                path = cfg.out_dir / "checkpoints" / f"adapter_step{step:06d}.pt"
                torch.save(ckpt, path)
                print(f"[ckpt] Saved {path}")

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
        "lora_config": lora_config.to_dict(),
        "config": cfg.__dict__,
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
    args = ap.parse_args()
    return TrainConfig(
        keypoints_dir=args.keypoints_dir,
        csv=args.csv,
        out_dir=args.out,
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
    )


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
