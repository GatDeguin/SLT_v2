from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import pytest

import tools_finetune_sonar_slt as finetune


class TinyVisualAdapter(nn.Module):
    def __init__(self, config: object, num_points: int, **_: object) -> None:
        super().__init__()
        self.num_points = num_points
        self.proj = nn.Linear(num_points * finetune.KEYPOINT_CHANNELS, 1024)

    def forward(
        self,
        keypoints: torch.Tensor,
        frames: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if frames is not None:
            raise AssertionError("TinyVisualAdapter expects keypoints only in tests")
        b, t, n, c = keypoints.shape
        x = keypoints.reshape(b, t, n * c).mean(dim=1)
        return self.proj(x)


class SimpleTextPooler:
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        lang: str,
        *,
        num_layers: int,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._lang = lang
        self._num_layers = num_layers

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        if not texts:
            raise ValueError("encode expects at least one text")
        device = next(self._model.parameters()).device
        if hasattr(self._tokenizer, "src_lang"):
            self._tokenizer.src_lang = self._lang
        batch = self._tokenizer(texts, return_tensors="pt", padding=True).to(device)
        encoder = self._model.get_encoder()
        outputs = encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_dict=True,
        )
        return outputs.last_hidden_state.mean(dim=1)


@pytest.fixture()
def synthetic_training_setup(tmp_path: Path, hf_sonar_dir: Path, monkeypatch: pytest.MonkeyPatch):
    keypoints_dir = tmp_path / "kp"
    keypoints_dir.mkdir()
    meta_path = tmp_path / "meta.csv"
    rows = [
        {"id": "clip01", "text": "hello world"},
        {"id": "clip02", "text": "hola mundo"},
    ]
    with meta_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "text"], delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    rng = np.random.default_rng(0)
    for row in rows:
        kp = rng.normal(size=(4, 5, finetune.KEYPOINT_CHANNELS)).astype(np.float32)
        np.save(keypoints_dir / f"{row['id']}.npy", kp)

    monkeypatch.setattr(finetune, "VisualFusionAdapter", TinyVisualAdapter)
    monkeypatch.setattr(finetune, "TextPooler", SimpleTextPooler)
    monkeypatch.setattr(finetune.torch.amp, "GradScaler", torch.cuda.amp.GradScaler, raising=False)

    class _NoOpAutocast:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(finetune.torch.amp, "autocast", _NoOpAutocast, raising=False)

    out_dir = tmp_path / "run"
    config = finetune.TrainConfig(
        keypoints_dir=keypoints_dir,
        csv=meta_path,
        out_dir=out_dir,
        model_name=str(hf_sonar_dir),
        tgt_lang="en_XX",
        T=4,
        clip_frames=2,
        batch_size=1,
        accum=1,
        epochs=1,
        lr=5e-4,
        save_every=1,
        log_every=1,
        lam_nce=0.0,
    )
    return config


def test_train_pipeline_generates_checkpoint(synthetic_training_setup: finetune.TrainConfig):
    config = synthetic_training_setup
    finetune.train(config)

    checkpoints_dir = config.out_dir / "checkpoints"
    final_ckpt = checkpoints_dir / "adapter_final.pt"
    assert final_ckpt.exists(), "training should produce a final adapter checkpoint"

    ckpt = torch.load(final_ckpt, map_location="cpu")
    assert ckpt["num_landmarks"] == 5
    assert "adapter" in ckpt and "bridge" in ckpt
    bridge_weight = ckpt["bridge"]["weight"]
    assert bridge_weight.shape[1] == 1024
    assert ckpt["d_model"] == bridge_weight.shape[0]

    log_path = config.out_dir / "train_log.jsonl"
    with log_path.open("r", encoding="utf-8") as fh:
        entries = [json.loads(line) for line in fh if line.strip()]
    assert entries, "training log should record at least one step"


@pytest.mark.parametrize(
    "length,target",
    [
        (np.arange(5), 3),
        (np.arange(2), 4),
    ],
)
def test_pad_or_sample_behaviour(length: Iterable[int], target: int):
    arr = np.asarray(list(length), dtype=np.float32)
    arr = arr[:, None]
    adjusted = finetune.pad_or_sample(arr, target, axis=0)
    assert adjusted.shape[0] == target
