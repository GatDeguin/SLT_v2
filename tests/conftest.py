import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _ensure_dependency(module_name: str, package: str, extra_args: list[str]) -> None:
    if importlib.util.find_spec(module_name) is not None:
        return
    cmd = [sys.executable, "-m", "pip", "install", package, *extra_args]
    subprocess.check_call(cmd)


_REQUIREMENTS = {
    "numpy": ("numpy==1.26.4", []),
    "torch": ("torch==2.2.1", ["--index-url", "https://download.pytorch.org/whl/cpu"]),
    "transformers": ("transformers==4.40.1", []),
    "sentencepiece": ("sentencepiece==0.2.0", []),
    "peft": ("peft==0.10.0", []),
}

for module_name, (package, extra_args) in _REQUIREMENTS.items():
    _ensure_dependency(module_name, package, extra_args)


@pytest.fixture(scope="session", autouse=True)
def ensure_real_dependencies():
    return None


@pytest.fixture(scope="session")
def hf_sonar_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    cache_model_id = "hf-internal-testing/tiny-random-mbart"
    model = AutoModelForSeq2SeqLM.from_pretrained(cache_model_id)
    tokenizer = AutoTokenizer.from_pretrained(cache_model_id)

    base_dir = tmp_path_factory.mktemp("hf_cache") / "mtmlt" / "sonar-nllb-200-1.3B"
    base_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(base_dir)
    tokenizer.save_pretrained(base_dir)
    return base_dir
