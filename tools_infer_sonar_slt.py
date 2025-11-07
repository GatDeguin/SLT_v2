#!/usr/bin/env python3
"""
SONAR‑SLT: Inference script ("paper‑perfect" core, with optional keypoint fusion)

Usage (Windows PowerShell example):

  # Minimal
  python tools/infer_sonar_slt.py \
    --videos-dir data/single_signer/videos \
    --keypoints-dir data/single_signer/kp/keypoints \
    --csv meta.csv \
    --out runs/sonar_slt_infer \
    --tgt-lang spa_Latn \
    --device auto

  # If you have local checkpoints (visual block LoRA or full .safetensors)
  python tools/infer_sonar_slt.py \
    --videos-dir data/single_signer/videos \
    --keypoints-dir data/single_signer/kp/keypoints \
    --csv meta.csv \
    --out runs/sonar_slt_infer \
    --tgt-lang spa_Latn \
    --visual-weights .\models\visual_block_lora.safetensors \
    --sonar-model-dir .\models\sonar_m200m100_port \
    --use-keypoints \
    --device auto

Notes
-----
- This script follows the SONAR‑SLT paper design: visual encoders (spatial + motion) →
  lightweight fusion (1D‑Conv + MLP) → Transformer visual encoder → pooling to a sentence‑level
  *semantic* vector z → generation with the SONAR decoder conditioned on z and a target
  language token.
- If the official HF port is installed (e.g., as a local folder), set --sonar-model-dir to it.
  The script can also attempt to fall back to a generic seq2seq model, but *true* SONAR‑style
  generation requires the ported SONAR model that accepts semantic vectors.
- Keypoint fusion is *optional* and off‑paper: enable with --use-keypoints to add an MLP over
  the (T, N, C) landmarks and late‑fuse into the visual stream (helps hands/face details).
- Designed to run on CPU or CUDA; use --half on capable GPUs.

Dependencies
------------
- torch, torchvision, timm, transformers >= 4.42, accelerate (optional), numpy, opencv-python
- If you want VideoMAE: `pip install transformers timm`

Output
------
- A CSV: {out}/preds.csv with columns: id,video,lang,text
- Per‑clip JSONL with extra diagnostics in {out}/logs.jsonl

"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional heavy deps
try:
    import timm  # ViT
except Exception:
    timm = None  # type: ignore

try:
    from transformers import (
        AutoConfig,
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

VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".webm")


def _resolve_device(name: str) -> torch.device:
    name = name.strip().lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():  # mac
            return torch.device("mps")
        return torch.device("cpu")
    if name in {"cuda", "gpu"} and torch.cuda.is_available():
        return torch.device("cuda")
    if name in {"mps"} and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -----------------------------
# Video I/O (OpenCV)
# -----------------------------
import cv2


def read_video_frames(path: Path, target_T: int, resize: Tuple[int, int]) -> np.ndarray:
    """Load `target_T` frames uniformly sampled; returns uint8 array [T,H,W,3] BGR.
    If the clip is shorter, pad by repeating the last frame.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    H, W = resize[1], resize[0]

    if total <= 0:
        # Fallback: grab sequentially until fail
        frames: List[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frames.append(cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR))
        cap.release()
        if not frames:
            raise ValueError(f"No frames read from {path}")
        total = len(frames)
        idxs = np.linspace(0, total - 1, num=target_T).round().astype(int)
        sampled = [frames[i] for i in idxs]
        arr = np.stack(sampled, axis=0)
        return arr

    idxs = np.linspace(0, max(total - 1, 0), num=target_T).round().astype(int)
    frames: List[np.ndarray] = []
    for i, idx in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            if frames:
                frames.append(frames[-1].copy())
                continue
            raise ValueError(f"Failed to read frame {idx} from {path}")
        frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)
        frames.append(frame)
    cap.release()
    return np.stack(frames, axis=0)  # [T,H,W,3]


# -----------------------------
# Visual Encoders (spatial + motion)
# -----------------------------
class ViTSpatial(nn.Module):
    """Frame‑wise spatial encoder using ViT (ImageNet pretrained).
    Emits per‑frame features of size D_s.
    """

    def __init__(self, model_name: str = "vit_base_patch16_224", out_dim: int = 768):
        super().__init__()
        if timm is None:
            raise RuntimeError("timm is required for ViTSpatial")
        vit = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.vit = vit  # timm model returns [B,D]
        self.out_dim = out_dim

    @torch.no_grad()
    def forward(self, frames_bthwc: torch.Tensor) -> torch.Tensor:
        """frames_bthwc: uint8/float [B,T,H,W,3] in BGR; converts to RGB float [0,1]
        → returns [B,T,D_s]
        """
        B, T, H, W, C = frames_bthwc.shape
        x = frames_bthwc[..., ::-1].float() / 255.0  # BGR→RGB
        x = x.view(B * T, H, W, C).permute(0, 3, 1, 2)  # [BT,3,H,W]
        feats = self.vit(x)  # [BT, D]
        return feats.view(B, T, -1)


class SimpleMotion3D(nn.Module):
    """Lightweight 3D conv as a proxy for motion features (when VideoMAE is not set).
    Intended for inference; replace with a proper VideoMAE if available.
    Outputs [B,T,D_m] (time‑aligned by tiling the clip feature).
    """

    def __init__(self, out_dim: int = 512):
        super().__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3)),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((None, 1, 1)),  # keep time, pool spatial
        )
        self.proj = nn.Linear(128, out_dim)

    @torch.no_grad()
    def forward(self, frames_bthwc: torch.Tensor) -> torch.Tensor:
        B, T, H, W, C = frames_bthwc.shape
        x = frames_bthwc[..., ::-1].float() / 255.0  # to RGB
        x = x.permute(0, 4, 1, 2, 3)  # [B,3,T,H,W]
        feats = self.conv3d(x)  # [B,128,T,1,1]
        feats = feats.squeeze(-1).squeeze(-1).permute(0, 2, 1)  # [B,T,128]
        return self.proj(feats)  # [B,T,out_dim]


class FusionBlock(nn.Module):
    """SpaMo‑style lightweight fusion: concat(spatial, motion) → 1D‑Conv → MLP.
    Produces per‑frame embedding h_t.
    """

    def __init__(self, d_spatial: int, d_motion: int, d_hidden: int = 1024):
        super().__init__()
        self.proj = nn.Conv1d(d_spatial + d_motion, d_hidden, kernel_size=1)
        self.mlp = nn.Sequential(
            nn.Linear(d_hidden, d_hidden), nn.ReLU(inplace=True), nn.Linear(d_hidden, d_hidden)
        )

    def forward(self, s_btD: torch.Tensor, m_btD: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s_btD, m_btD], dim=-1)  # [B,T,Ds+Dm]
        x = x.transpose(1, 2)  # [B,D,T]
        x = self.proj(x).transpose(1, 2)  # [B,T,Dh]
        return self.mlp(x)  # [B,T,Dh]


class TransformerVisualEncoder(nn.Module):
    """Transformer encoder over time; returns z_{1:T} and pooled z."""

    def __init__(self, d_model: int = 1024, nhead: int = 8, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout,
                                                   dim_feedforward=4 * d_model, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool_token = nn.Parameter(torch.zeros(1, 1, d_model))  # for attentive pooling
        self.attn_q = nn.Linear(d_model, d_model)

    def forward(self, h_btD: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = h_btD.shape
        z_seq = self.encoder(h_btD)  # [B,T,D]
        # Attention pooling using a learned query (paper allows mean or attention)
        q = self.attn_q(self.pool_token.expand(B, -1, -1))  # [B,1,D]
        attn = torch.softmax((q @ z_seq.transpose(1, 2)) / math.sqrt(D), dim=-1)  # [B,1,T]
        z = (attn @ z_seq).squeeze(1)  # [B,D]
        return z_seq, z


# -----------------------------
# Optional Keypoint encoder (off‑paper helper; controlled by --use-keypoints)
# -----------------------------
class KeypointEncoder(nn.Module):
    """Encodes (T, N, C) landmarks → per‑frame D_k; C in {2,3} (x,y[,conf])."""

    def __init__(self, num_points: int, in_ch: int = 3, d_k: int = 256):
        super().__init__()
        self.num_points = num_points
        self.in_ch = in_ch
        self.net = nn.Sequential(
            nn.Linear(num_points * in_ch, 512), nn.ReLU(True),
            nn.Linear(512, d_k), nn.ReLU(True),
        )

    def forward(self, kp_btnc: torch.Tensor) -> torch.Tensor:
        B, T, N, C = kp_btnc.shape
        x = kp_btnc.reshape(B, T, N * C)
        return self.net(x)  # [B,T,d_k]


class LateFusion(nn.Module):
    def __init__(self, d_visual: int, d_k: int, d_out: int = 1024):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_visual + d_k, d_out), nn.ReLU(True),
            nn.Linear(d_out, d_out)
        )

    def forward(self, visual_btD: torch.Tensor, key_btD: torch.Tensor) -> torch.Tensor:
        x = torch.cat([visual_btD, key_btD], dim=-1)
        return self.gate(x)


# -----------------------------
# SONAR decoder wrapper
# -----------------------------
class SonarDecoder:
    """Thin wrapper around the SONAR HF port, conditioned on a semantic vector z.

    If a specialized SONAR model is not available, we fall back to a generic seq2seq
    and simply prefix the target language token, but high‑quality decoding *requires*
    the SONAR port that accepts encoder_outputs.
    """

    def __init__(self, model_dir: Optional[str], device: torch.device, half: bool = False):
        self.device = device
        self.model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(
            model_dir or "facebook/nllb-200-distilled-600M"
        ).to(device)
        if half and device.type == "cuda":
            self.model.half()
        self.model.eval()
        self.tok = AutoTokenizer.from_pretrained(model_dir or "facebook/nllb-200-distilled-600M")
        # Determine model d_model
        self.d_model = getattr(self.model.config, "d_model", None) or getattr(self.model.config, "hidden_size", 1024)
        # Map visual z (1024 by default) to model d_model if needed
        self.bridge = nn.Linear(1024, self.d_model, bias=False).to(device)
        if half and device.type == "cuda":
            self.bridge.half()

    @torch.no_grad()
    def generate(self, z_bD: torch.Tensor, tgt_lang: str, max_new_tokens: int = 64, num_beams: int = 4) -> List[str]:
        # Build pseudo encoder outputs from z as a single‑token memory
        B, D = z_bD.shape
        mem = self.bridge(z_bD).unsqueeze(1)  # [B,1,d_model]
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
    def __init__(self, use_keypoints: bool = False):
        super().__init__()
        self.vit = ViTSpatial("vit_base_patch16_224", out_dim=768)
        self.motion = SimpleMotion3D(out_dim=512)
        self.fuse = FusionBlock(768, 512, d_hidden=1024)
        self.enc = TransformerVisualEncoder(d_model=1024, nhead=8, num_layers=4)
        self.use_kp = use_keypoints
        if use_keypoints:
            # default expects MediaPipe holistic: >= 543 points with (x,y,conf)
            self.kpenc = KeypointEncoder(num_points=543, in_ch=3, d_k=256)
            self.kpfuse = LateFusion(d_visual=1024, d_k=256, d_out=1024)

    @torch.no_grad()
    def forward(self, frames_bthwc: torch.Tensor, keypoints_btnc: Optional[torch.Tensor] = None) -> torch.Tensor:
        s = self.vit(frames_bthwc)     # [B,T,768]
        m = self.motion(frames_bthwc)  # [B,T,512]
        h = self.fuse(s, m)            # [B,T,1024]
        if self.use_kp and keypoints_btnc is not None:
            k = self.kpenc(keypoints_btnc)  # [B,T,256]
            h = self.kpfuse(h, k)           # [B,T,1024]
        _, z = self.enc(h)                   # [B,1024]
        return z


# -----------------------------
# Data plumbing (meta.csv + file resolution)
# -----------------------------
@dataclass
class Clip:
    clip_id: str
    video_path: Path
    keypoints_path: Optional[Path]


def load_meta(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig") as fh:
        rdr = csv.DictReader(fh, delimiter=";")
        rows = [ {k: (v.strip() if isinstance(v, str) else v) for k,v in row.items()} for row in rdr ]
    if not rows:
        raise ValueError(f"CSV has no rows: {csv_path}")
    return rows


def resolve_clips(rows: List[Dict[str, str]], videos_dir: Path, keypoints_dir: Optional[Path]) -> List[Clip]:
    out: List[Clip] = []
    for row in rows:
        clip_id = row.get("id") or row.get("video_id") or row.get("video")
        if not clip_id:
            raise ValueError("meta.csv must contain an 'id' or 'video' column")
        # video match by stem
        vid = None
        for ext in VIDEO_EXTS:
            candidate = videos_dir / f"{clip_id}{ext}"
            if candidate.exists():
                vid = candidate
                break
        if vid is None:
            # try any file starting with stem
            matches = list(videos_dir.glob(f"{clip_id}.*"))
            vid = matches[0] if matches else None
        if vid is None:
            raise FileNotFoundError(f"Video not found for id={clip_id} in {videos_dir}")

        kp_path = None
        if keypoints_dir and keypoints_dir.is_dir():
            for ext in (".npz", ".npy"):
                cand = keypoints_dir / f"{clip_id}{ext}"
                if cand.exists():
                    kp_path = cand
                    break
        out.append(Clip(clip_id, vid, kp_path))
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
    ap = argparse.ArgumentParser(description="SONAR‑SLT inference: videos + optional keypoints")
    ap.add_argument("--videos-dir", type=Path, required=True)
    ap.add_argument("--keypoints-dir", type=Path, default=None)
    ap.add_argument("--csv", type=Path, required=True, help="meta.csv with an 'id' or 'video' column")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tgt-lang", type=str, default="spa_Latn", help="NLLB language token, e.g., 'deu_Latn', 'eng_Latn', 'spa_Latn'")
    ap.add_argument("--T", type=int, default=128, help="frames per clip")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--sonar-model-dir", type=str, default=None, help="Path to the SONAR HF port (preferred)")
    ap.add_argument("--visual-weights", type=Path, default=None, help=".safetensors for visual block/LoRA (optional)")
    ap.add_argument("--use-keypoints", action="store_true", help="Fuse keypoints via MLP (off‑paper helper)")
    args = ap.parse_args()

    device = _resolve_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    # Load data index
    rows = load_meta(args.csv)
    clips = resolve_clips(rows, args.videos_dir, args.keypoints_dir)

    # Build model
    model = SonarSLTInference(use_keypoints=args.use_keypoints).to(device)
    if args.half and device.type == "cuda":
        model.half()
    model.eval()

    # Optionally load visual weights (.safetensors / state_dict subset)
    if args.visual_weights and args.visual_weights.exists():
        from safetensors.torch import load_file as safe_load
        sd = safe_load(str(args.visual_weights))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print("[warn] missing keys:", len(missing))
        if unexpected:
            print("[warn] unexpected keys:", len(unexpected))

    # SONAR decoder
    decoder = SonarDecoder(args.sonar_model_dir, device=device, half=args.half)

    preds_path = args.out / "preds.csv"
    logs_path = args.out / "logs.jsonl"

    with preds_path.open("w", newline="", encoding="utf-8") as fh_csv, logs_path.open("w", encoding="utf-8") as fh_log:
        w = csv.writer(fh_csv)
        w.writerow(["id", "video", "lang", "text"])  # header

        for clip in clips:
            # 1) Load frames
            frames = read_video_frames(clip.video_path, target_T=args.T, resize=(args.img_size, args.img_size))  # [T,H,W,3]
            frames_t = torch.from_numpy(frames).unsqueeze(0).to(device)  # [1,T,H,W,3]
            if args.half and device.type == "cuda":
                frames_t = frames_t.half()
            else:
                frames_t = frames_t.float()

            # 2) Load keypoints (optional)
            kp_t: Optional[torch.Tensor] = None
            if args.use_keypoints and clip.keypoints_path and clip.keypoints_path.exists():
                kp = load_keypoints_array(clip.keypoints_path)  # [T,N,C]
                if kp.ndim != 3:
                    raise ValueError(f"Keypoints must be (T,N,C), got {kp.shape}")
                # Normalise: fill missing conf, center & scale XY to roughly [-1,1]
                Tcur, N, C = kp.shape
                if C == 2:
                    kp = np.concatenate([kp, np.ones((Tcur, N, 1), dtype=kp.dtype)], axis=-1)
                # centre
                xy = kp[..., :2]
                xy = xy - np.nanmean(xy, axis=(1, 0), keepdims=True)
                # scale by per‑clip max abs
                scale = np.nanmax(np.abs(xy)) or 1.0
                xy = xy / float(scale)
                kp[..., :2] = xy
                # time fit
                if Tcur != args.T:
                    idxs = np.linspace(0, Tcur - 1, num=args.T).round().astype(int)
                    kp = kp[idxs]
                kp_t = torch.from_numpy(kp).unsqueeze(0).to(device)
                if args.half and device.type == "cuda":
                    kp_t = kp_t.half()
                else:
                    kp_t = kp_t.float()

            # 3) Encode → z
            with torch.inference_mode():
                z = model(frames_t, kp_t)  # [1,1024]

            # 4) Decode to target language
            text = decoder.generate(z, tgt_lang=args.tgt_lang, max_new_tokens=args.max_new_tokens, num_beams=args.num_beams)[0]

            w.writerow([clip.clip_id, clip.video_path.name, args.tgt_lang, text])
            fh_log.write(json.dumps({
                "id": clip.clip_id,
                "video": str(clip.video_path),
                "keypoints": (str(clip.keypoints_path) if clip.keypoints_path else None),
                "lang": args.tgt_lang,
                "text": text,
            }, ensure_ascii=False) + "\n")
            fh_log.flush()
            print(f"[ok] {clip.clip_id} → {text}")


if __name__ == "__main__":
    main()
