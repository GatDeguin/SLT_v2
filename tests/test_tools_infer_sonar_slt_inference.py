from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import pytest

import tools_infer_sonar_slt as infer
from transformers import AutoModelForSeq2SeqLM


def _write_meta_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "text"], delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture()
def sonar_checkpoint(tmp_path: Path, hf_sonar_dir: Path) -> Path:
    model = infer.SonarSLTInference(num_landmarks=3)
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
        "config": {"model_name": str(hf_sonar_dir)},
    }
    ckpt_path = tmp_path / "adapter.pt"
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
