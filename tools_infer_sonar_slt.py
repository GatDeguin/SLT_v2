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
    --tgt-lang spa_Latn \
    --device auto

Notes
-----
- Expects (T, N, C) keypoints (MediaPipe Holistic by default) normalised as in training.
- Provide the checkpoint containing `adapter` and `bridge` weights from the fine‑tuning script.
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
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        PreTrainedModel,
        GenerationConfig,
    )
    from transformers.modeling_outputs import BaseModelOutput
except Exception as exc:  # pragma: no cover
    print("[FATAL] transformers not available:", exc)
    raise

# -----------------------------
# Utilities
# -----------------------------

KEYPOINT_CHANNELS = 3  # (x, y, conf)


def normalise_keypoints(arr: np.ndarray) -> np.ndarray:
    """Center by mean landmark and scale to unit max-radius; keep confidence.
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
    except (TypeError, ValueError):
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
        B, T, N, C = kp_btnc.shape
        x = kp_btnc.reshape(B, T, N * C)
        x = self.proj(x)
        return self.temporal(x)  # [B,T,hidden]


class FusionAdapter(nn.Module):
    """Keypoint‑only adapter that outputs a pooled vector z in SONAR space (D=1024)."""

    def __init__(self, config: AdapterConfig, num_points: int) -> None:
        super().__init__()
        self.kp_enc = KeypointEncoder(config, num_points)
        self.kp_proj = nn.Linear(config.keypoint_hidden, config.hidden_size)
        self.fusion = _TransformerEncoder(
            config.hidden_size,
            num_layers=config.fusion_layers,
            nhead=config.fusion_heads,
            dropout=config.dropout,
        )
        self.out_norm = nn.LayerNorm(config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, kp_btnc: torch.Tensor) -> torch.Tensor:
        kp = self.kp_enc(kp_btnc)                 # [B,T,H_k]
        kp = self.kp_proj(kp)                     # [B,T,D]
        fused = self.fusion(kp)                   # [B,T,D]
        z = fused.mean(dim=1)                     # mean pooling
        z = self.out_norm(z)
        return self.out_proj(z)                   # [B,D]


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
    ):
        self.device = device
        # Matches TrainConfig.model_name to ensure existing bridge checkpoints load without changes.
        default_model = "mtmlt/sonar-nllb-200-1.3B"

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
    def generate(self, z_bD: torch.Tensor, tgt_lang: str, max_new_tokens: int = 64, num_beams: int = 4) -> List[str]:
        if self.bridge is None:
            raise RuntimeError("Bridge weights have not been loaded. Call load_bridge() before generation")
        # Build pseudo encoder outputs from z as a single‑token memory
        B, D = z_bD.shape
        z = z_bD.to(self.bridge.weight.device)
        z = z.to(self.bridge.weight.dtype)
        mem = self.bridge(z).unsqueeze(1)  # [B,1,d_model]
        enc_out = BaseModelOutput(last_hidden_state=mem)

        forced_bos = self.tok.convert_tokens_to_ids([tgt_lang])
        forced_bos_id = forced_bos[0] if forced_bos and forced_bos[0] is not None else None

        gen_cfg = GenerationConfig(
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            forced_bos_token_id=forced_bos_id,
            use_cache=True,
        )
        outputs = self.model.generate(
            encoder_outputs=enc_out,
            decoder_input_ids=None,
            generation_config=gen_cfg,
        )
        return self.tok.batch_decode(outputs, skip_special_tokens=True)


# -----------------------------
# End‑to‑end model
# -----------------------------
class SonarSLTInference(nn.Module):
    def __init__(self, num_landmarks: int):
        super().__init__()
        self.fusion_adapter = FusionAdapter(AdapterConfig(), num_landmarks)

    def load_adapter(self, adapter_state: Dict[str, torch.Tensor]) -> None:
        if not isinstance(adapter_state, dict):
            raise TypeError("Adapter state must be a state dict mapping strings to tensors")
        missing, unexpected = self.fusion_adapter.load_state_dict(adapter_state, strict=False)
        if missing:
            raise RuntimeError(f"Adapter checkpoint is missing parameters: {sorted(missing)}")
        if unexpected:
            raise RuntimeError(f"Adapter checkpoint has unexpected parameters: {sorted(unexpected)}")

    @torch.no_grad()
    def forward(self, keypoints_btnc: torch.Tensor) -> torch.Tensor:
        if keypoints_btnc.ndim != 4:
            raise ValueError(f"Expected keypoints (B,T,N,C), got {keypoints_btnc.shape}")
        kp = keypoints_btnc.to(self.fusion_adapter.kp_proj.weight.dtype)
        return self.fusion_adapter(kp)


# -----------------------------
# Data plumbing (meta.csv + file resolution)
# -----------------------------
@dataclass
class Clip:
    clip_id: str
    keypoints_path: Path
    video_name: Optional[str] = None


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
        clip_id = row.get("id") or row.get("video_id") or row.get("video")
        if not clip_id:
            raise ValueError("meta.csv must contain an 'id' or 'video' column")

        kp_path = None
        for ext in (".npz", ".npy"):
            cand = keypoints_dir / f"{clip_id}{ext}"
            if cand.exists():
                kp_path = cand
                break
        if kp_path is None:
            raise FileNotFoundError(f"Keypoints not found for id={clip_id} in {keypoints_dir}")

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
    ap.add_argument("--keypoints-dir", type=Path, required=True)
    ap.add_argument(
        "--adapter-ckpt",
        type=Path,
        required=True,
        help="Checkpoint from tools_finetune_sonar_slt.py (required when using keypoints)",
    )
    ap.add_argument("--csv", type=Path, required=True, help="meta.csv with an 'id' or 'video' column")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tgt-lang", type=str, default="spa_Latn", help="NLLB language token, e.g., 'deu_Latn', 'eng_Latn', 'spa_Latn'")
    ap.add_argument("--T", type=int, default=128, help="Temporal length for keypoints")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument(
        "--sonar-model-dir",
        type=str,
        default=None,
        help="Path or HF repo for SONAR decoder override (default: 'mtmlt/sonar-nllb-200-1.3B')",
    )
    ap.add_argument(
        "--adapter-device",
        type=str,
        default="cpu",
        help="Device for loading the adapter checkpoint (e.g., 'cpu', 'cuda')",
    )
    args = ap.parse_args()

    device = _resolve_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    # Load data index
    rows = load_meta(args.csv)
    clips = resolve_clips(rows, args.keypoints_dir)

    if not args.adapter_ckpt.is_file():
        raise FileNotFoundError(f"Adapter checkpoint not found: {args.adapter_ckpt}")
    adapter_map_device = _resolve_device(args.adapter_device)
    try:
        ckpt = torch.load(args.adapter_ckpt, map_location=adapter_map_device)
    except (RuntimeError, OSError, ValueError, EOFError) as exc:
        raise RuntimeError(f"Failed to load adapter checkpoint '{args.adapter_ckpt}': {exc}") from exc

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

    num_landmarks = int(ckpt.get("num_landmarks") or 0)
    if num_landmarks <= 0:
        raise ValueError("Checkpoint missing a valid 'num_landmarks'")
    d_model = ckpt.get("d_model")

    # Build adapter model
    model = SonarSLTInference(num_landmarks=num_landmarks).to(device)
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
    )
    decoder.load_bridge(bridge_state)

    preds_path = args.out / "preds.csv"
    logs_path = args.out / "logs.jsonl"

    with preds_path.open("w", newline="", encoding="utf-8") as fh_csv, logs_path.open("w", encoding="utf-8") as fh_log:
        w = csv.writer(fh_csv)
        w.writerow(["id", "video", "lang", "text"])  # header

        for clip in clips:
            if not clip.keypoints_path.exists():
                raise FileNotFoundError(f"Missing keypoints for {clip.clip_id}: {clip.keypoints_path}")
            kp = load_keypoints_array(clip.keypoints_path)
            if kp.ndim == 2:
                feat_dim = kp.shape[1]
                if feat_dim % KEYPOINT_CHANNELS == 0:
                    num_points = feat_dim // KEYPOINT_CHANNELS
                elif feat_dim % 543 == 0:
                    num_points = 543
                else:
                    raise ValueError(
                        f"Cannot infer landmark count from shape {kp.shape} (expected multiple of {KEYPOINT_CHANNELS})"
                    )
                kp = kp.reshape(-1, num_points, KEYPOINT_CHANNELS)
            if kp.ndim != 3:
                raise ValueError(f"Keypoints must be (T,N,C), got {kp.shape}")
            if kp.shape[1] != num_landmarks:
                raise ValueError(
                    f"Checkpoint expects {num_landmarks} landmarks but found {kp.shape[1]} in {clip.keypoints_path}"
                )
            kp = kp.astype(np.float32, copy=False)
            kp = normalise_keypoints(kp)
            kp = pad_or_sample(kp, args.T, axis=0)
            kp_t = torch.from_numpy(kp).unsqueeze(0).to(device)
            if args.half and device.type == "cuda":
                kp_t = kp_t.half()
            else:
                kp_t = kp_t.float()

            with torch.inference_mode():
                z = model(kp_t)  # [1,1024]

            z = z.to(decoder.bridge.weight.device)
            z = z.to(decoder.bridge.weight.dtype)

            text = decoder.generate(
                z,
                tgt_lang=args.tgt_lang,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            )[0]

            video_name = clip.video_name or clip.clip_id
            w.writerow([clip.clip_id, video_name, args.tgt_lang, text])
            fh_log.write(json.dumps({
                "id": clip.clip_id,
                "video": video_name,
                "keypoints": str(clip.keypoints_path),
                "lang": args.tgt_lang,
                "text": text,
            }, ensure_ascii=False) + "\n")
            fh_log.flush()
            print(f"[ok] {clip.clip_id} → {text}")


if __name__ == "__main__":
    main()
