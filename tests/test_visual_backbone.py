"""Regression checks for visual backbones."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import torch

from models import visual_backbone


class _DummyVideoMAEOutput:
    def __init__(self, last_hidden_state: torch.Tensor) -> None:
        self.last_hidden_state = last_hidden_state


class _DummyVideoMAEConfig:
    hidden_size: int = 8
    patch_size: int = 1
    tubelet_size: int = 1
    use_cls_token: bool = False

    @classmethod
    def from_pretrained(cls, _: str) -> "_DummyVideoMAEConfig":
        return cls()


class _DummyVideoMAEModel(torch.nn.Module):
    def __init__(self, config: _DummyVideoMAEConfig) -> None:
        super().__init__()
        self.config = config
        self.last_inputs: torch.Tensor | None = None
        self.last_loaded_state: tuple[dict, bool] | None = None

    @classmethod
    def from_pretrained(cls, _: str) -> "_DummyVideoMAEModel":
        return cls(_DummyVideoMAEConfig())

    def forward(
        self,
        *,
        pixel_values: torch.Tensor,
        output_hidden_states: bool = False,  # noqa: ARG002 - parity with HF signature
    ) -> _DummyVideoMAEOutput:
        del output_hidden_states
        self.last_inputs = pixel_values
        batch, frames, channels, height, width = pixel_values.shape
        tokens_per_frame = max((height // self.config.patch_size) * (width // self.config.patch_size), 1)
        total_tokens = frames * tokens_per_frame
        hidden = pixel_values.new_zeros(batch, total_tokens, self.config.hidden_size)
        return _DummyVideoMAEOutput(hidden)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> dict:
        self.last_loaded_state = (state_dict, strict)
        return {}


def test_videomae_backbone_accepts_b_t_c_h_w(monkeypatch) -> None:
    monkeypatch.setattr(visual_backbone, "VideoMAEModel", _DummyVideoMAEModel)
    monkeypatch.setattr(visual_backbone, "VideoMAEConfig", _DummyVideoMAEConfig)

    backbone = visual_backbone.VideoMAEBackbone(model_name="dummy", pretrained=False)
    frames = torch.randn(2, 4, 3, 8, 8)

    output = backbone(frames)

    assert isinstance(backbone.model, _DummyVideoMAEModel)
    assert backbone.model.last_inputs is not None
    assert backbone.model.last_inputs.shape == frames.shape
    assert torch.allclose(backbone.model.last_inputs, frames)
    assert output.shape == (2, 4, backbone.hidden_size)


class _DummyViTModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_features = 8
        self.loaded_state: tuple[dict, bool] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - signature parity
        batch, *_ = x.shape
        return x.new_zeros(batch, self.num_features)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> dict:
        self.loaded_state = (state_dict, strict)
        return {}


def test_backbones_load_safetensors_checkpoints(monkeypatch) -> None:
    dummy_state = {"weight": torch.tensor([1.0])}
    load_calls: list[tuple[str, str]] = []

    def fake_load_file(path: str, *, device: str = "cpu") -> dict:
        load_calls.append((path, device))
        return dummy_state

    safetensors_module = types.SimpleNamespace(
        torch=types.SimpleNamespace(load_file=fake_load_file)
    )
    monkeypatch.setitem(sys.modules, "safetensors", safetensors_module)
    monkeypatch.setitem(sys.modules, "safetensors.torch", safetensors_module.torch)

    def fail_torch_load(*_: object, **__: object) -> None:
        raise AssertionError("torch.load should not be called for .safetensors")

    monkeypatch.setattr(visual_backbone.torch, "load", fail_torch_load)

    vit_model = _DummyViTModel()

    def create_model(*_: object, **__: object) -> _DummyViTModel:
        return vit_model

    monkeypatch.setattr(visual_backbone, "timm", types.SimpleNamespace(create_model=create_model))

    monkeypatch.setattr(visual_backbone, "VideoMAEModel", _DummyVideoMAEModel)
    monkeypatch.setattr(visual_backbone, "VideoMAEConfig", _DummyVideoMAEConfig)

    vit_backbone = visual_backbone.ViTBackbone(
        model_name="dummy",
        pretrained=False,
        checkpoint_path=Path("dummy.safetensors"),
    )
    assert vit_model.loaded_state == (dummy_state, False)

    videomae_backbone = visual_backbone.VideoMAEBackbone(
        model_name="dummy",
        pretrained=False,
        checkpoint_path=Path("dummy.safetensors"),
    )
    assert isinstance(videomae_backbone.model, _DummyVideoMAEModel)
    assert videomae_backbone.model.last_loaded_state == (dummy_state, False)

    expected_path = str(Path("dummy.safetensors"))
    assert load_calls == [(expected_path, "cpu"), (expected_path, "cpu")]
