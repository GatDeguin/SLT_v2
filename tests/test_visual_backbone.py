"""Regression checks for visual backbones."""

from __future__ import annotations

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
