from __future__ import annotations

import numpy as np


def extract_confidence_channel(
    arr: np.ndarray, *, expected_channels: int = 3
) -> np.ndarray:
    """Return a confidence channel compatible with SONAR-SLT keypoint layout.

    The helpers expect (T, N, C>=2) arrays where the last channel can encode
    confidence. When more than ``expected_channels`` channels are present we keep
    the final one to mirror inference preprocessing, ensuring reshaped MediaPipe
    dumps preserve their confidence scores.
    """

    if arr.shape[-1] > expected_channels:
        return arr[..., -1:]
    if arr.shape[-1] >= expected_channels:
        return arr[..., expected_channels - 1 : expected_channels]
    return np.ones_like(arr[..., :1])
