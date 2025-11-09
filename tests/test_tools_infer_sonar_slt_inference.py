import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


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
    torch.inference_mode = _no_grad
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
    torch.from_numpy = lambda arr: arr
    torch.stack = lambda tensors, dim=0: tensors
    torch.randn_like = lambda tensor: tensor
    torch.rand = lambda *args, **kwargs: 0
    torch.where = lambda condition, x, y: x
    torch.zeros_like = lambda tensor: tensor
    torch.is_tensor = lambda obj: False
    torch.serialization = types.SimpleNamespace(add_safe_globals=lambda *a, **k: None)

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
    transformers.PreTrainedModel = object
    transformers.GenerationConfig = object

    modeling_outputs = types.ModuleType("transformers.modeling_outputs")
    modeling_outputs.BaseModelOutput = object

    transformers.modeling_outputs = modeling_outputs

    sys.modules["transformers"] = transformers
    sys.modules["transformers.modeling_outputs"] = modeling_outputs


class FakeFusionAdapter:
    def __init__(self):
        self.received_state = None
        self.missing = []
        self.unexpected = []
        self.kp_proj = types.SimpleNamespace(weight=None, bias=None)
        self.keypoint_bias = None
        self._state_dict = {
            "kp_enc.some_weight": "existing_weight",
            "kp_proj.weight": "existing_kp_weight",
            "kp_proj.bias": "existing_kp_bias",
            "keypoint_bias": "existing_bias",
        }

    def load_state_dict(self, state_dict, strict=False):
        self.received_state = dict(state_dict)
        self.kp_proj.weight = self.received_state.get("kp_proj.weight")
        self.kp_proj.bias = self.received_state.get("kp_proj.bias")
        self.keypoint_bias = self.received_state.get("keypoint_bias")
        return self.missing, self.unexpected

    def state_dict(self):
        return dict(self._state_dict)


_defused = False


def _import_module():
    global _defused
    if not _defused:
        _ensure_torch_stub()
        _ensure_numpy_stub()
        _ensure_transformers_stub()
        _defused = True

    transformers = sys.modules["transformers"]
    if not hasattr(transformers, "PreTrainedModel"):
        transformers.PreTrainedModel = object
    if not hasattr(transformers, "GenerationConfig"):
        transformers.GenerationConfig = object

    module_path = Path(__file__).resolve().parent.parent / "tools_infer_sonar_slt.py"
    spec = importlib.util.spec_from_file_location("tools_infer_sonar_slt", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["tools_infer_sonar_slt"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_load_adapter_converts_visual_fusion_state(capsys):
    module = _import_module()
    fake_adapter = FakeFusionAdapter()
    instance = module.SonarSLTInference.__new__(module.SonarSLTInference)
    instance.fusion_adapter = fake_adapter

    adapter_state = {
        "kp_enc.some_weight": object(),
        "fusion.key_proj.weight": "weight_tensor",
        "fusion.key_proj.bias": "bias_tensor",
        "fusion.modality_embeddings.weight": [[1, 2, 3], [4, 5, 6]],
        "fusion.spatial_proj.bias": "ignore_me_too",
        "fusion.motion_proj.weight": "ignore_me_three",
        "vit.encoder.layers.0.weight": "vit_tensor",
        "videomae.encoder.layers.0.bias": "videomae_tensor",
    }

    instance.load_adapter(adapter_state)
    captured = capsys.readouterr()

    assert fake_adapter.kp_proj.weight == "weight_tensor"
    assert fake_adapter.kp_proj.bias == "bias_tensor"
    assert fake_adapter.keypoint_bias == [1, 2, 3]
    assert fake_adapter.received_state is not None
    assert "fusion.key_proj.weight" not in fake_adapter.received_state
    assert "fusion.modality_embeddings.weight" not in fake_adapter.received_state
    assert "fusion.spatial_proj.bias" not in fake_adapter.received_state
    assert "fusion.motion_proj.weight" not in fake_adapter.received_state
    assert "vit.encoder.layers.0.weight" not in fake_adapter.received_state
    assert "videomae.encoder.layers.0.bias" not in fake_adapter.received_state
    log_output = captured.out + captured.err
    assert "vit.encoder.layers.0.weight" in log_output
    assert "videomae.encoder.layers.0.bias" in log_output


def test_flat_keypoints_extra_channels_trimmed():
    module = _import_module()
    num_landmarks = 543
    timesteps = 2
    flat = np.arange(timesteps * num_landmarks * 4, dtype=np.float32).reshape(
        timesteps, num_landmarks * 4
    )

    reshaped = module.reshape_flat_keypoints(flat, num_landmarks)

    assert reshaped.shape == (timesteps, num_landmarks, module.KEYPOINT_CHANNELS)
    reshaped_full = flat.reshape(timesteps, num_landmarks, 4)
    expected = np.concatenate(
        [reshaped_full[..., :2], reshaped_full[..., -1:]], axis=-1
    )
    np.testing.assert_array_equal(reshaped, expected)


def test_normalise_keypoints_retains_confidence_track():
    module = _import_module()
    raw = np.array([
        [[1.0, 2.0, 99.0, 0.9], [3.0, 4.0, 42.0, 0.8]],
    ])

    normalised = module.normalise_keypoints(raw)

    assert normalised.shape == (1, 2, module.KEYPOINT_CHANNELS)

    coords = raw[..., :2]
    center = coords.mean(axis=-2, keepdims=True)
    centered = coords - center
    norms = np.linalg.norm(centered, axis=-1)
    max_norm = np.max(norms, axis=-1, keepdims=True)
    max_norm = np.clip(max_norm, 1e-4, None)
    expected_coords = centered / max_norm[..., None]
    expected = np.concatenate([expected_coords, raw[..., -1:]], axis=-1)

    np.testing.assert_allclose(normalised, expected)


def test_fusion_adapter_forward_adds_keypoint_bias():
    module = _import_module()

    class DummyTensor:
        def __init__(self, value):
            self.value = value

        def __add__(self, other):
            other_value = getattr(other, "value", other)
            return DummyTensor(self.value + other_value)

        __radd__ = __add__

        def mean(self, dim):
            return DummyTensor(self.value)

    class DummyBias:
        def __init__(self, value):
            self.value = value

        def view(self, *shape):
            return self

    class DummyModule:
        def __init__(self, result):
            self._result = result

        def __call__(self, *args, **kwargs):
            return self._result

    class DummyFusion:
        def __init__(self):
            self.received = None

        def __call__(self, tensor):
            self.received = tensor
            return tensor

    adapter = module.FusionAdapter.__new__(module.FusionAdapter)
    encoded = DummyTensor(0)
    projected = DummyTensor(0)
    fusion = DummyFusion()

    adapter.kp_enc = DummyModule(encoded)
    adapter.kp_proj = DummyModule(projected)
    adapter.keypoint_bias = DummyBias(7)
    adapter.fusion = fusion
    adapter.out_norm = lambda x: x
    adapter.out_proj = lambda x: x

    output = adapter.forward(object())

    assert fusion.received.value == 7
    assert output.value == 7
