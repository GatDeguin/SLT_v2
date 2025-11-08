import sys
import types
import importlib.util
from pathlib import Path


class _DummyContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False

    def __call__(self, func):
        return func


def _ensure_torch_stub():
    if "torch" in sys.modules:
        return

    torch = types.ModuleType("torch")
    torch.Tensor = object
    torch.float32 = "float32"
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))

    def _no_grad(func=None):
        if func is None:
            return _DummyContext()
        return func

    torch.no_grad = _no_grad
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        amp=types.SimpleNamespace(
            GradScaler=lambda *a, **k: object(),
            autocast=lambda *a, **k: _DummyContext(),
        ),
    )
    torch.amp = types.SimpleNamespace(
        GradScaler=lambda *a, **k: object(),
        autocast=lambda *a, **k: _DummyContext(),
    )
    torch.optim = types.SimpleNamespace(AdamW=lambda *a, **k: object())
    torch.linalg = types.SimpleNamespace(pinv=lambda *a, **k: None)
    torch.save = lambda *a, **k: None

    def _from_numpy(arr):
        class _FakeTensor:
            def __init__(self, array):
                self.array = array

            def to(self, *args, **kwargs):
                return self

        return _FakeTensor(arr)

    torch.from_numpy = _from_numpy

    class _Device:
        def __init__(self, spec):
            self._spec = spec
            if ":" in spec:
                type_part, idx = spec.split(":", 1)
                self.type = type_part.lower()
                try:
                    self.index = int(idx)
                except ValueError:
                    self.index = idx
            else:
                self.type = spec.lower()
                self.index = None

        def __repr__(self):
            return f"device(type='{self.type}', index={self.index})"

        def __str__(self):
            return self._spec

        def __eq__(self, other):
            if isinstance(other, _Device):
                return self._spec == other._spec
            if isinstance(other, str):
                return self._spec == other
            return False

    torch.device = lambda spec: _Device(spec)

    torch.stack = lambda tensors, dim=0: tensors
    torch.randn_like = lambda tensor: tensor
    torch.rand = lambda *args, **kwargs: 0
    torch.where = lambda condition, x, y: x
    torch.zeros_like = lambda tensor: tensor

    torch.utils = types.ModuleType("torch.utils")
    torch.utils.data = types.ModuleType("torch.utils.data")
    torch.utils.data.Dataset = object
    torch.utils.data.DataLoader = object

    torch_nn = types.ModuleType("torch.nn")

    class _Module:
        def __init__(self, *args, **kwargs):
            pass

        def to(self, *args, **kwargs):
            return self

    torch_nn.Module = _Module
    torch_nn.Linear = lambda *a, **k: _Module()
    torch_nn.Sequential = lambda *modules: _Module()
    torch_nn.LayerNorm = lambda *a, **k: _Module()
    torch_nn.Dropout = lambda *a, **k: _Module()
    torch_nn.GELU = lambda *a, **k: _Module()
    torch_nn.TransformerEncoderLayer = lambda *a, **k: _Module()
    torch_nn.TransformerEncoder = lambda *a, **k: _Module()

    torch_nn_functional = types.ModuleType("torch.nn.functional")

    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = torch_nn
    sys.modules["torch.nn.functional"] = torch_nn_functional
    sys.modules["torch.utils"] = torch.utils
    sys.modules["torch.utils.data"] = torch.utils.data


def _ensure_numpy_stub():
    if "numpy" in sys.modules:
        return

    class _NumpyStub(types.ModuleType):
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    numpy = _NumpyStub("numpy")
    numpy.ndarray = object
    numpy.float32 = "float32"
    numpy.float64 = "float64"
    numpy.int64 = "int64"

    sys.modules["numpy"] = numpy


def _ensure_transformers_stub():
    if "transformers" in sys.modules:
        return

    transformers = types.ModuleType("transformers")
    transformers.AutoModelForSeq2SeqLM = object
    transformers.AutoTokenizer = object

    modeling_outputs = types.ModuleType("transformers.modeling_outputs")
    modeling_outputs.BaseModelOutput = object

    transformers.modeling_outputs = modeling_outputs

    sys.modules["transformers"] = transformers
    sys.modules["transformers.modeling_outputs"] = modeling_outputs


def test_read_meta_csv_accepts_semicolon(tmp_path):
    _ensure_torch_stub()
    _ensure_numpy_stub()
    _ensure_transformers_stub()

    module_path = Path(__file__).resolve().parent.parent / "tools_finetune_sonar_slt.py"
    spec = importlib.util.spec_from_file_location("tools_finetune_sonar_slt", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["tools_finetune_sonar_slt"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)

    csv_path = tmp_path / "meta.csv"
    csv_path.write_text("video_id;text\nexample_01;hola mundo\n", encoding="utf-8")

    rows = module._read_meta_csv(csv_path)

    assert len(rows) == 1
    assert rows[0].vid == "example_01"
    assert rows[0].text == "hola mundo"


def test_read_meta_csv_prefers_semicolon_header(tmp_path):
    _ensure_torch_stub()
    _ensure_numpy_stub()
    _ensure_transformers_stub()

    module_path = Path(__file__).resolve().parent.parent / "tools_finetune_sonar_slt.py"
    spec = importlib.util.spec_from_file_location("tools_finetune_sonar_slt", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["tools_finetune_sonar_slt"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)

    csv_path = tmp_path / "meta.csv"
    csv_path.write_text(
        "id;text;video\n"
        "clip_001;estamos en noviembre, y reflexionaba.;noticias-en-lengua-de-senas-argentina-resumen-semanal-29112020\n",
        encoding="utf-8",
    )

    rows = module._read_meta_csv(csv_path)

    assert len(rows) == 1
    assert rows[0].vid == "clip_001"
    assert rows[0].text == "estamos en noviembre, y reflexionaba."


def test_resolve_device_accepts_indexed_cuda():
    _ensure_torch_stub()
    _ensure_numpy_stub()
    _ensure_transformers_stub()

    module_path = Path(__file__).resolve().parent.parent / "tools_finetune_sonar_slt.py"
    spec = importlib.util.spec_from_file_location("tools_finetune_sonar_slt", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["tools_finetune_sonar_slt"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)

    module.torch.cuda.is_available = lambda: True

    device = module.resolve_device("cuda:1")

    assert device.type == "cuda"
    assert device.index == 1
