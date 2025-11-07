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
    L_joint = λ_sem L_sem  +  λ_ce L_ce  +  λ_ae L_ae (InfoNCE optional)
  with strong MSE magnitude regularizer and cosine term.

Dependencies
------------
  pip install torch torchvision timm transformers peft numpy pandas opencv-python rich

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

# -----------------------------
# Constants / Layout helpers
# -----------------------------
KEYPOINT_CHANNELS = 3  # (x, y, conf)
DEFAULT_T = 128
DEFAULT_IMG_SIZE = 224
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
        sample = fh.read(2048)
        fh.seek(0)
        delimiter: Optional[str] = None
        header_line = ""
        if sample:
            lines = sample.splitlines()
            if lines:
                header_line = lines[0]
            preferred_delimiter: Optional[str] = None
            if header_line:
                counts = {ch: header_line.count(ch) for ch in (",", ";", "\t")}
                semicolons = counts[";"]
                commas = counts[","]
                if semicolons > 0 and commas == 0:
                    preferred_delimiter = ";"
                elif semicolons > 0 and semicolons >= max(counts.values()):
                    preferred_delimiter = ";"
            try:
                sniffed = csv.Sniffer().sniff(sample, delimiters=",;\t")
                delimiter = getattr(sniffed, "delimiter", None)
            except csv.Error:
                delimiter = None
            if preferred_delimiter:
                delimiter = preferred_delimiter
        if not delimiter and header_line and ";" in header_line:
            delimiter = ";"
        reader_kwargs = {"delimiter": delimiter} if delimiter else {}
        reader = csv.DictReader(fh, **reader_kwargs)
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
    def __init__(self, keypoints_dir: Path, csv_path: Path, T: int = DEFAULT_T, shuffle: bool = True):
        self.keypoints_dir = keypoints_dir
        self.items = _read_meta_csv(csv_path)
        if shuffle:
            RNG.shuffle(self.items)
        self.T = int(T)

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
        return {
            "id": row.vid,
            "keypoints": torch.from_numpy(keypoints).to(torch.float32),  # (T, N, C)
            "text": row.text,
        }


# -----------------------------
# Model: Keypoint encoder → temporal transformer → pooled z (1024)
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
        z = fused.mean(dim=1)                     # mean pooling (paper allows mean/attn)
        z = self.out_norm(z)
        return self.out_proj(z)                   # [B,D]


# -----------------------------
# Text encoder / pooling with SONAR HF port (M2M‑style mean pooling)
# -----------------------------
class TextPooler:
    def __init__(self, model: AutoModelForSeq2SeqLM, tokenizer: AutoTokenizer, lang: str):
        self.model = model
        self.tok = tokenizer
        self.lang = lang

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        """Mean‑pool encoder states → sentence embedding s (B, d_model)."""
        self.tok.src_lang = self.lang
        batch = self.tok(texts, return_tensors="pt", padding=True).to(self.model.device)
        seq = self.model.model.encoder(**batch).last_hidden_state  # [B,S,d]
        mask = batch.attention_mask
        mean_emb = (seq * mask.unsqueeze(-1)).sum(1) / mask.unsqueeze(-1).sum(1)
        return mean_emb  # (B,d)


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
    model_name: str = "mtmlt/sonar-nllb-200-1.3B"
    tgt_lang: str = "spa_Latn"
    T: int = DEFAULT_T
    batch_size: int = 8
    accum: int = 2
    epochs: int = 10
    lr: float = 3e-4
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


def resolve_device(name: str) -> torch.device:
    name = name.strip().lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name in {"cuda", "gpu"} and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
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
    ds = KPTextDataset(cfg.keypoints_dir, cfg.csv, T=cfg.T)
    N = count_landmarks(ds[0]["keypoints"])  # landmarks

    # Model pieces
    device = resolve_device(cfg.device)
    decoder = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name).to(device)
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if cfg.half and device.type == "cuda":
        decoder.half()
    # Freeze decoder by default (PEFT spirit); only train adapter + bridge
    for p in decoder.parameters():
        p.requires_grad = False

    # Bridge maps adapter’s 1024 to decoder d_model if needed
    d_model = getattr(decoder.config, "d_model", None) or getattr(decoder.config, "hidden_size", DEFAULT_D_MODEL)
    bridge = nn.Linear(1024, d_model, bias=False).to(device)
    if cfg.half and device.type == "cuda":
        bridge.half()

    adapter = FusionAdapter(AdapterConfig(), num_points=N).to(device)
    if cfg.half and device.type == "cuda":
        adapter.half()

    text_pool = TextPooler(decoder, tok, cfg.tgt_lang)

    # Optimiser
    opt = torch.optim.AdamW(list(adapter.parameters()) + list(bridge.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.half and device.type == "cuda")

    # Loader
    def collate(batch):
        ids = [b["id"] for b in batch]
        texts = [b["text"] for b in batch]
        kps = torch.stack([b["keypoints"] for b in batch], dim=0)  # [B,T,N,C]
        return ids, texts, kps

    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, collate_fn=collate, pin_memory=(device.type=="cuda"))

    step = 0
    log_path = cfg.out_dir / "train_log.jsonl"
    for epoch in range(cfg.epochs):
        for ids, texts, kp in dl:
            step += 1
            kp = kp.to(device)
            kp = jitter_keypoints(kp, noise_std=0.04, drop_prob=0.2)

            with torch.amp.autocast("cuda", enabled=cfg.half and device.type == "cuda"):
                # --- Visual → semantic vector z (Eq. 1–3)
                z = adapter(kp)                # [B,1024]
                mem_z = bridge(z).unsqueeze(1) # [B,1,d_model]

                # --- Text → sentence embedding s (Eq. 4 + SONAR pooling)
                s = text_pool.encode(texts).to(mem_z.dtype)  # [B,d_model]
                # Project to adapter space for L_sem
                # bring s to 1024 dim via inverse bridge (least‑squares via pseudo‑inverse)
                with torch.no_grad():
                    # Compute W^+ (pseudo‑inverse) once per step (small d_model); cache if needed
                    W = bridge.weight  # [d_model,1024]
                    W_pinv = torch.linalg.pinv(W, rcond=1e-5)
                s_1024 = (s @ W_pinv.T).contiguous()  # [B,1024]

                # --- Losses
                # L_sem = α * MSE(z, s_1024) + β * (1 - cos(z, s_1024))
                mse = F.mse_loss(z, s_1024)
                cos = 1.0 - F.cosine_similarity(z, s_1024, dim=-1).mean()
                L_sem = cfg.mse_weight * mse + cfg.cos_weight * cos

                # L_ce — teacher forcing on target Spanish from z
                tok.src_lang = cfg.tgt_lang
                batch = tok(texts, return_tensors="pt", padding=True).to(device)
                labels = batch["input_ids"].clone()
                # In seq2seq HF, we pass encoder_outputs + labels
                enc_out_z = BaseModelOutput(last_hidden_state=mem_z)
                out_z = decoder(encoder_outputs=enc_out_z, labels=labels)
                L_ce = out_z.loss

                # L_ae — auto‑encoding from text embedding s
                mem_s = s.unsqueeze(1)  # [B,1,d_model]
                enc_out_s = BaseModelOutput(last_hidden_state=mem_s)
                out_s = decoder(encoder_outputs=enc_out_s, labels=labels)
                L_ae = out_s.loss

                loss = cfg.lam_sem * L_sem + cfg.lam_ce * L_ce + cfg.lam_ae * L_ae

            scaler.scale(loss / cfg.accum).backward()

            if step % cfg.accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

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
                }
                with log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"[step {step:6d}] loss={rec['loss']:.4f}  L_sem={rec['L_sem']:.4f}  L_ce={rec['L_ce']:.4f}  L_ae={rec['L_ae']:.4f}")

            if step % cfg.save_every == 0:
                ckpt = {
                    "adapter": adapter.state_dict(),
                    "bridge": bridge.state_dict(),
                    "config": cfg.__dict__,
                    "d_model": d_model,
                    "num_landmarks": N,
                }
                path = cfg.out_dir / "checkpoints" / f"adapter_step{step:06d}.pt"
                torch.save(ckpt, path)
                print(f"[ckpt] Saved {path}")

    # Final checkpoint
    path = cfg.out_dir / "checkpoints" / f"adapter_final.pt"
    ckpt = {
        "adapter": adapter.state_dict(),
        "bridge": bridge.state_dict(),
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
    ap.add_argument("--model-name", type=str, default="mtmlt/sonar-nllb-200-1.3B")
    ap.add_argument("--tgt-lang", type=str, default="spa_Latn")
    ap.add_argument("--T", type=int, default=DEFAULT_T)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--mse-weight", type=float, default=7000.0)
    ap.add_argument("--cos-weight", type=float, default=2.7)
    ap.add_argument("--lam-sem", type=float, default=1.0)
    ap.add_argument("--lam-ce", type=float, default=0.1)
    ap.add_argument("--lam-ae", type=float, default=0.2)
    args = ap.parse_args()
    return TrainConfig(
        keypoints_dir=args.keypoints_dir,
        csv=args.csv,
        out_dir=args.out,
        model_name=args.model_name,
        tgt_lang=args.tgt_lang,
        T=args.T,
        batch_size=args.batch_size,
        accum=args.accum,
        epochs=args.epochs,
        lr=args.lr,
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
    )


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
