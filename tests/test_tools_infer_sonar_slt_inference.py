from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pytest

import tools_infer_sonar_slt as infer
from transformers import AutoModelForSeq2SeqLM


def _write_meta_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "text"], delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


class DummyVisualAdapter(nn.Module):
    called_with_frames: list[torch.Tensor] = []

    def __init__(self, config: object, num_points: int, **_: object) -> None:
        super().__init__()
        hidden = getattr(config, "hidden_size", 1024)
        input_dim = num_points * infer.KEYPOINT_CHANNELS
        self.kp_proj = nn.Linear(input_dim, hidden)
        self.vit = nn.Linear(1, hidden)
        self.videomae = nn.Linear(1, hidden)
        self.out = nn.Linear(hidden, hidden)

    @classmethod
    def reset_calls(cls) -> None:
        cls.called_with_frames.clear()

    def forward(self, keypoints: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
        if frames is None:
            raise AssertionError("DummyVisualAdapter expects frames")
        DummyVisualAdapter.called_with_frames.append(frames.detach().cpu())
        b, t, n, c = keypoints.shape
        kp_feat = keypoints.reshape(b, t, n * c).mean(dim=1)
        frame_scalar = frames.mean(dim=(1, 2, 3, 4), keepdim=True).view(b, 1)
        fused = self.kp_proj(kp_feat) + self.vit(frame_scalar) + self.videomae(frame_scalar)
        return self.out(torch.tanh(fused))


@pytest.fixture()
def patched_visual_adapter(monkeypatch: pytest.MonkeyPatch):
    DummyVisualAdapter.reset_calls()
    monkeypatch.setattr(infer, "VisualFusionAdapter", DummyVisualAdapter)
    return DummyVisualAdapter


@pytest.fixture()
def sonar_checkpoint(tmp_path: Path, hf_sonar_dir: Path) -> Path:
    adapter_module = infer.FusionAdapter(infer.AdapterConfig(), num_points=3)
    model = infer.SonarSLTInference(adapter_module, expects_frames=False)
    adapter_state = model.fusion_adapter.state_dict()

    decoder_model = AutoModelForSeq2SeqLM.from_pretrained(str(hf_sonar_dir))
    d_model = getattr(decoder_model.config, "d_model", None) or getattr(
        decoder_model.config, "hidden_size", 1024
    )
    bridge = torch.nn.Linear(1024, d_model, bias=False)
    bridge_state = bridge.state_dict()

    ckpt = {
        "adapter": adapter_state,
        "bridge": bridge_state,
        "num_landmarks": 3,
        "d_model": d_model,
        "config": {"model_name": str(hf_sonar_dir), "T": 42},
    }
    ckpt_path = tmp_path / "adapter.pt"
    torch.save(ckpt, ckpt_path)
    return ckpt_path


@pytest.fixture()
def multimodal_checkpoint(
    tmp_path: Path,
    hf_sonar_dir: Path,
    patched_visual_adapter: type[DummyVisualAdapter],
) -> Path:
    adapter = patched_visual_adapter(infer.AdapterConfig(), num_points=3)
    adapter_state = adapter.state_dict()

    decoder_model = AutoModelForSeq2SeqLM.from_pretrained(str(hf_sonar_dir))
    d_model = getattr(decoder_model.config, "d_model", None) or getattr(
        decoder_model.config, "hidden_size", 1024
    )
    bridge = torch.nn.Linear(1024, d_model, bias=False)
    bridge_state = bridge.state_dict()

    ckpt = {
        "adapter": adapter_state,
        "bridge": bridge_state,
        "num_landmarks": 3,
        "d_model": d_model,
        "config": {
            "model_name": str(hf_sonar_dir),
            "clip_frames": 3,
            "frame_size": 8,
            "vit_model": "dummy-vit",
            "videomae_model": "dummy-vmae",
            "freeze_vit": False,
            "freeze_videomae": False,
            "freeze_keypoint": False,
        },
    }
    ckpt_path = tmp_path / "adapter_multimodal.pt"
    torch.save(ckpt, ckpt_path)
    return ckpt_path


def test_inference_pipeline_runs(tmp_path: Path, sonar_checkpoint: Path, hf_sonar_dir: Path, monkeypatch: pytest.MonkeyPatch):
    keypoints_dir = tmp_path / "kp"
    keypoints_dir.mkdir()

    rng = np.random.default_rng(42)
    keypoints = rng.normal(size=(2, 3, infer.KEYPOINT_CHANNELS)).astype(np.float32)
    np.save(keypoints_dir / "clip01.npy", keypoints)

    meta_csv = tmp_path / "meta.csv"
    _write_meta_csv(meta_csv, [{"id": "clip01", "text": "placeholder"}])

    out_dir = tmp_path / "out"
    args = [
        "tools_infer_sonar_slt.py",
        "--keypoints-dir",
        str(keypoints_dir),
        "--adapter-ckpt",
        str(sonar_checkpoint),
        "--csv",
        str(meta_csv),
        "--out",
        str(out_dir),
        "--sonar-model-dir",
        str(hf_sonar_dir),
        "--tgt-lang",
        "en_XX",
        "--T",
        "2",
        "--num-beams",
        "1",
        "--max-new-tokens",
        "4",
    ]
    monkeypatch.setenv("TRANSFORMERS_CACHE", str((tmp_path / "hf_cache").resolve()))
    monkeypatch.setattr(sys, "argv", args)

    infer.main()

    preds_path = out_dir / "preds.csv"
    logs_path = out_dir / "logs.jsonl"

    assert preds_path.exists()
    with preds_path.open("r", encoding="utf-8") as fh:
        lines = [line.strip() for line in fh if line.strip()]
    assert len(lines) == 2

    assert logs_path.exists()
    with logs_path.open("r", encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    assert records


def test_temporal_length_inherits_from_checkpoint(
    tmp_path: Path,
    sonar_checkpoint: Path,
    hf_sonar_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keypoints_dir = tmp_path / "kp_inherit"
    keypoints_dir.mkdir()

    rng = np.random.default_rng(123)
    keypoints = rng.normal(size=(5, 3, infer.KEYPOINT_CHANNELS)).astype(np.float32)
    np.save(keypoints_dir / "clip01.npy", keypoints)

    meta_csv = tmp_path / "meta.csv"
    _write_meta_csv(meta_csv, [{"id": "clip01", "text": "placeholder"}])

    out_dir = tmp_path / "out_inherit"

    original_pad_or_sample = infer.pad_or_sample
    seen_targets: list[int] = []

    def _tracking_pad_or_sample(array: np.ndarray, target_frames: int, axis: int = 0) -> np.ndarray:
        seen_targets.append(target_frames)
        return original_pad_or_sample(array, target_frames, axis=axis)

    monkeypatch.setattr(infer, "pad_or_sample", _tracking_pad_or_sample)

    args = [
        "tools_infer_sonar_slt.py",
        "--keypoints-dir",
        str(keypoints_dir),
        "--adapter-ckpt",
        str(sonar_checkpoint),
        "--csv",
        str(meta_csv),
        "--out",
        str(out_dir),
        "--sonar-model-dir",
        str(hf_sonar_dir),
        "--tgt-lang",
        "en_XX",
        "--num-beams",
        "1",
        "--max-new-tokens",
        "4",
    ]
    monkeypatch.setenv("TRANSFORMERS_CACHE", str((tmp_path / "hf_cache").resolve()))
    monkeypatch.setattr(sys, "argv", args)

    infer.main()

    assert 42 in seen_targets

def test_resolve_clips_supports_ids_with_extension(tmp_path: Path) -> None:
    keypoints_dir = tmp_path / "kp"
    keypoints_dir.mkdir()

    keypoints = np.zeros((1, 3, infer.KEYPOINT_CHANNELS), dtype=np.float32)
    np.save(keypoints_dir / "clip.npy", keypoints)

    rows = [{"id": "clip.npy", "text": "placeholder"}]

    clips = infer.resolve_clips(rows, keypoints_dir)
    assert len(clips) == 1
    assert clips[0].keypoints_path == keypoints_dir / "clip.npy"


def test_multimodal_checkpoint_requires_video(
    tmp_path: Path,
    multimodal_checkpoint: Path,
    hf_sonar_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keypoints_dir = tmp_path / "kp"
    keypoints_dir.mkdir()
    rng = np.random.default_rng(0)
    keypoints = rng.normal(size=(2, 3, infer.KEYPOINT_CHANNELS)).astype(np.float32)
    np.save(keypoints_dir / "clip01.npy", keypoints)

    meta_csv = tmp_path / "meta.csv"
    _write_meta_csv(meta_csv, [{"id": "clip01", "text": "placeholder"}])

    out_dir = tmp_path / "out"
    args = [
        "tools_infer_sonar_slt.py",
        "--keypoints-dir",
        str(keypoints_dir),
        "--adapter-ckpt",
        str(multimodal_checkpoint),
        "--csv",
        str(meta_csv),
        "--out",
        str(out_dir),
        "--sonar-model-dir",
        str(hf_sonar_dir),
        "--tgt-lang",
        "en_XX",
        "--T",
        "2",
    ]
    monkeypatch.setenv("TRANSFORMERS_CACHE", str((tmp_path / "hf_cache").resolve()))
    monkeypatch.setattr(sys, "argv", args)

    with pytest.raises(RuntimeError, match="provide paired videos"):
        infer.main()


def test_multimodal_inference_with_video_produces_text(
    tmp_path: Path,
    multimodal_checkpoint: Path,
    hf_sonar_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    patched_visual_adapter: type[DummyVisualAdapter],
) -> None:
    keypoints_dir = tmp_path / "kp"
    keypoints_dir.mkdir()
    rng = np.random.default_rng(1)
    keypoints = rng.normal(size=(2, 3, infer.KEYPOINT_CHANNELS)).astype(np.float32)
    np.save(keypoints_dir / "clip01.npy", keypoints)

    meta_csv = tmp_path / "meta.csv"
    _write_meta_csv(meta_csv, [{"id": "clip01", "text": "placeholder"}])

    video_dir = tmp_path / "videos"
    video_dir.mkdir()

    class StubVideoLoader:
        def __init__(self, video_dir: Path, clip_frames: int, frame_size: int) -> None:
            self.video_dir = video_dir
            self.clip_frames = clip_frames
            self.frame_size = frame_size

        def load(self, clip) -> np.ndarray:
            clip.video_path = self.video_dir / f"{clip.clip_id}.mp4"
            base = np.linspace(0.1, 0.9, num=self.clip_frames, dtype=np.float32)
            frames = base[:, None, None, None]
            frames = np.tile(frames, (1, 3, self.frame_size, self.frame_size))
            return frames

    def _make_loader(video_dir: Path, clip_frames: int, frame_size: int):
        return StubVideoLoader(video_dir, clip_frames, frame_size)

    monkeypatch.setattr(infer, "VideoClipLoader", _make_loader)

    out_dir = tmp_path / "out"
    args = [
        "tools_infer_sonar_slt.py",
        "--keypoints-dir",
        str(keypoints_dir),
        "--adapter-ckpt",
        str(multimodal_checkpoint),
        "--csv",
        str(meta_csv),
        "--out",
        str(out_dir),
        "--sonar-model-dir",
        str(hf_sonar_dir),
        "--tgt-lang",
        "en_XX",
        "--T",
        "2",
        "--clip-frames",
        "3",
        "--frame-size",
        "8",
        "--video-dir",
        str(video_dir),
        "--num-beams",
        "1",
        "--max-new-tokens",
        "4",
    ]
    monkeypatch.setenv("TRANSFORMERS_CACHE", str((tmp_path / "hf_cache").resolve()))
    monkeypatch.setattr(sys, "argv", args)

    infer.main()

    preds_path = out_dir / "preds.csv"
    logs_path = out_dir / "logs.jsonl"

    assert preds_path.exists()
    with preds_path.open("r", encoding="utf-8") as fh:
        rows = [line.strip() for line in fh if line.strip()]
    assert len(rows) == 2
    record = rows[1]
    assert record.split(",", maxsplit=3)[-1].strip()

    assert logs_path.exists()
    with logs_path.open("r", encoding="utf-8") as fh:
        entries = [json.loads(line) for line in fh if line.strip()]
    assert entries and entries[0].get("video_path")
    assert entries[0].get("video_stats") is not None

    assert patched_visual_adapter.called_with_frames
    first_call = patched_visual_adapter.called_with_frames[0]
    assert first_call.shape[0] == 1
    assert first_call.shape[1] == 3
