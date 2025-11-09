from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


pytest.importorskip("sacrebleu")


def write_csv(path: Path, header: str, rows: str) -> None:
    content = "\n".join([header, rows]).strip() + "\n"
    path.write_text(content, encoding="utf-8")


def test_eval_script_runs(tmp_path: Path) -> None:
    refs = tmp_path / "meta.csv"
    preds = tmp_path / "preds.csv"
    output = tmp_path / "metrics.json"

    write_csv(
        refs,
        "id;text;split",
        "\n".join(
            [
                "clip-1;Hola mundo.;test",
                "clip-2;Esto es una prueba.;test",
            ]
        ),
    )

    write_csv(
        preds,
        "id,video,lang,text",
        "\n".join(
            [
                "clip-1,video_a,spa_Latn,Hola mundo.",
                "clip-2,video_b,spa_Latn,Esto es una prueba.",
            ]
        ),
    )

    cmd = [
        sys.executable,
        str(Path.cwd() / "tools_eval_sonar_slt.py"),
        "--preds",
        str(preds),
        "--references",
        str(refs),
        "--split",
        "test",
        "--output",
        str(output),
    ]

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert "Evaluated 2 samples" in result.stdout

    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["samples"] == 2
    assert data["metrics"]["bleu"] == pytest.approx(100.0)
    assert data["metrics"]["chrf"] == pytest.approx(100.0)
    assert data["metrics"]["ter"] == pytest.approx(0.0)
    assert data["metrics"]["wer"] == pytest.approx(0.0)
    assert data["metrics"]["rougeL"] == pytest.approx(100.0)

