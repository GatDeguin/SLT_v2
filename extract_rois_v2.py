"""Extracción de regiones de interés (cara y manos) con MediaPipe.

Este script procesa videos y guarda crops cuadrados de cara y manos junto
con la pose superior. Incluye utilidades de línea de comandos para reanudar
procesamientos, limitar FPS de muestreo y generar un registro de metadata
por video, facilitando auditorías posteriores.
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import cv2
import numpy as np

try:  # pragma: no cover - dependencias opcionales
    import mediapipe as mp
except Exception:  # pragma: no cover - dependencias opcionales
    mp = None  # type: ignore[assignment]

try:  # pragma: no cover - dependencias opcionales
    from mediapipe.tasks import python as mp_tasks_python
    from mediapipe.tasks.python import vision as mp_vision
except Exception:  # pragma: no cover - dependencias opcionales
    mp_tasks_python = None  # type: ignore[assignment]
    mp_vision = None  # type: ignore[assignment]


if hasattr(cv2, "__spec__") and getattr(cv2.__spec__, "name", None) is None:  # type: ignore[attr-defined]
    cv2.__spec__ = None  # type: ignore[assignment]


if not hasattr(cv2, "VideoCapture"):
    class _UnavailableVideoCapture:  # pragma: no cover - fallback mínimo
        def __init__(self, *_: object, **__: object) -> None:
            self._opened = False

        def isOpened(self) -> bool:
            return False

        def read(self):  # type: ignore[override]
            return False, np.empty((0, 0, 3), dtype=np.uint8)

        def release(self) -> None:
            return None

        def get(self, *_: object) -> float:
            return 0.0

    cv2.VideoCapture = _UnavailableVideoCapture  # type: ignore[assignment]

if not hasattr(cv2, "COLOR_BGR2GRAY"):
    cv2.COLOR_BGR2GRAY = 0  # type: ignore[attr-defined]
if not hasattr(cv2, "COLOR_GRAY2BGR"):
    cv2.COLOR_GRAY2BGR = 1  # type: ignore[attr-defined]
if not hasattr(cv2, "COLOR_BGR2RGB"):
    cv2.COLOR_BGR2RGB = 2  # type: ignore[attr-defined]
if not hasattr(cv2, "INTER_LINEAR"):
    cv2.INTER_LINEAR = 1  # type: ignore[attr-defined]
if not hasattr(cv2, "CAP_PROP_FPS"):
    cv2.CAP_PROP_FPS = 5  # type: ignore[attr-defined]


def _ensure_uint8(array: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(array), 0, 255).astype(np.uint8)


if not hasattr(cv2, "cvtColor"):

    def _cvt_color(image: np.ndarray, code: int) -> np.ndarray:
        if code == cv2.COLOR_BGR2GRAY:
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("Se requiere una imagen BGR")
            weights = np.array([0.114, 0.587, 0.299], dtype=np.float32)
            gray = np.tensordot(image.astype(np.float32), weights, axes=([2], [0]))
            return _ensure_uint8(gray)
        if code == cv2.COLOR_GRAY2BGR:
            if image.ndim != 2:
                raise ValueError("Se requiere una imagen en escala de grises")
            return np.stack([image, image, image], axis=-1)
        if code == cv2.COLOR_BGR2RGB:
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("Se requiere una imagen BGR")
            return image[..., ::-1]
        raise NotImplementedError(f"Conversión no soportada: {code}")

    cv2.cvtColor = _cvt_color  # type: ignore[assignment]


if not hasattr(cv2, "GaussianBlur"):

    def _gaussian_blur(image: np.ndarray, ksize: Tuple[int, int], sigma: float) -> np.ndarray:
        kernel_x, kernel_y = ksize
        pad_x = kernel_x // 2
        pad_y = kernel_y // 2
        kernel = np.ones((kernel_y, kernel_x), dtype=np.float32)
        kernel /= kernel.sum()

        if image.ndim == 2:
            padded = np.pad(image.astype(np.float32), ((pad_y, pad_y), (pad_x, pad_x)), mode="edge")
            out = np.zeros_like(image, dtype=np.float32)
            for row in range(image.shape[0]):
                for col in range(image.shape[1]):
                    region = padded[row : row + kernel_y, col : col + kernel_x]
                    out[row, col] = float(np.sum(region * kernel))
            return _ensure_uint8(out)

        if image.ndim == 3:
            channels = [_gaussian_blur(image[:, :, idx], ksize, sigma) for idx in range(image.shape[2])]
            return np.stack(channels, axis=-1)

        raise ValueError("Imagen no soportada para blur")

    cv2.GaussianBlur = _gaussian_blur  # type: ignore[assignment]


if not hasattr(cv2, "merge"):

    def _merge_channels(channels: list[np.ndarray]) -> np.ndarray:
        return np.stack(channels, axis=-1)

    cv2.merge = _merge_channels  # type: ignore[assignment]


if not hasattr(cv2, "bitwise_not"):
    cv2.bitwise_not = np.bitwise_not  # type: ignore[assignment]


if not hasattr(cv2, "bitwise_and"):
    cv2.bitwise_and = np.bitwise_and  # type: ignore[assignment]


if not hasattr(cv2, "add"):

    def _add_images(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        return _ensure_uint8(lhs.astype(np.int32) + rhs.astype(np.int32))

    cv2.add = _add_images  # type: ignore[assignment]


if not hasattr(cv2, "circle"):

    def _draw_circle(image: np.ndarray, center: Tuple[int, int], radius: int, color: int, thickness: int) -> None:
        if thickness != -1:
            raise NotImplementedError("Solo se admite thickness=-1 en el modo de compatibilidad")
        cx, cy = center
        yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
        image[mask] = color

    cv2.circle = _draw_circle  # type: ignore[assignment]


if not hasattr(cv2, "resize"):

    def _resize(image: np.ndarray, size: Tuple[int, int], interpolation: int | None = None) -> np.ndarray:
        out_w, out_h = size
        if image.ndim == 2:
            channels = [image]
        else:
            channels = [image[:, :, idx] for idx in range(image.shape[2])]
        resized_channels = []
        for channel in channels:
            ys = np.linspace(0, channel.shape[0] - 1, out_h)
            xs = np.linspace(0, channel.shape[1] - 1, out_w)
            grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
            coords_y = np.clip(np.round(grid_y).astype(int), 0, channel.shape[0] - 1)
            coords_x = np.clip(np.round(grid_x).astype(int), 0, channel.shape[1] - 1)
            resized_channels.append(channel[coords_y, coords_x])
        stacked = np.stack(resized_channels, axis=-1)
        if image.ndim == 2:
            return stacked[:, :, 0]
        return stacked.astype(image.dtype)

    cv2.resize = _resize  # type: ignore[assignment]



_MP_WARNING = (
    "MediaPipe no está disponible. Instala el paquete `mediapipe` para poder "
    "extraer regiones de interés."
)


FACE_KEEP_MOUTH_IDXS = list(range(0, 18)) + list(range(61, 91))
FACE_KEEP_LEFT_EYE_IDXS = list(range(33, 134))
FACE_KEEP_RIGHT_EYE_IDXS = list(range(263, 363))
FACE_KEEP_INDICES = set(
    FACE_KEEP_MOUTH_IDXS + FACE_KEEP_LEFT_EYE_IDXS + FACE_KEEP_RIGHT_EYE_IDXS
)
FACE_POSE_INDICES = tuple(range(11))


_HAS_CV2_CIRCLE = hasattr(cv2, "circle")
_HAS_CV2_CVTCOLOR = hasattr(cv2, "cvtColor")
_HAS_CV2_GAUSSIAN = hasattr(cv2, "GaussianBlur")
_HAS_CV2_MERGE = hasattr(cv2, "merge")
_HAS_CV2_BITWISE_AND = hasattr(cv2, "bitwise_and")
_HAS_CV2_BITWISE_NOT = hasattr(cv2, "bitwise_not")
_HAS_CV2_ADD = hasattr(cv2, "add")


_DELEGATE_CPU = "cpu"
_DELEGATE_GPU = "gpu"


_TASK_DEFAULTS = {
    "--face-model": Path("modules/face_landmarker/face_landmarker.task"),
    "--hand-model": Path("modules/hand_landmarker/hand_landmarker.task"),
    "--pose-model": Path("modules/pose_landmarker/pose_landmarker_full.task"),
}


# Valores admitidos por GLOG/ABSL y sus equivalentes en MediaPipe.
_MP_LOG_LEVELS = {
    "info": {"env": "0", "absl": "info"},
    "warning": {"env": "1", "absl": "warning"},
    "error": {"env": "2", "absl": "error"},
    "fatal": {"env": "3", "absl": "fatal"},
}
_DEFAULT_MP_LOG_LEVEL = "error"
_configured_mp_log_level: Optional[str] = None


KEYPOINT_BODY_LANDMARKS = 33
KEYPOINT_FACE_LANDMARKS = 468
KEYPOINT_HAND_LANDMARKS = 21
KEYPOINT_TOTAL_LANDMARKS = (
    KEYPOINT_BODY_LANDMARKS + KEYPOINT_FACE_LANDMARKS + 2 * KEYPOINT_HAND_LANDMARKS
)
KEYPOINT_LAYOUT_NAME = "mediapipe_holistic_v1"
def _fill_circle(mask: np.ndarray, center: tuple[int, int], radius: int, value: int) -> None:
    """Rellena un disco en la máscara sin depender de cv2.circle."""

    if _HAS_CV2_CIRCLE:
        cv2.circle(mask, center, radius, value, -1)
        return

    cx, cy = center
    yy, xx = np.ogrid[: mask.shape[0], : mask.shape[1]]
    disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    mask[disk] = value


def _bgr_to_gray(image: np.ndarray) -> np.ndarray:
    if _HAS_CV2_CVTCOLOR and hasattr(cv2, "COLOR_BGR2GRAY"):
        try:
            converted = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if isinstance(converted, np.ndarray) and converted.ndim == 2:
                return converted
        except Exception:
            pass
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Se requiere una imagen BGR")
    weights = np.array([0.114, 0.587, 0.299], dtype=np.float32)
    gray = np.tensordot(image.astype(np.float32), weights, axes=([2], [0]))
    return _ensure_uint8(gray)


def _gray_to_bgr(image: np.ndarray) -> np.ndarray:
    if _HAS_CV2_CVTCOLOR and hasattr(cv2, "COLOR_GRAY2BGR"):
        try:
            converted = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            if isinstance(converted, np.ndarray) and converted.ndim == 3:
                return converted
        except Exception:
            pass
    if image.ndim != 2:
        raise ValueError("Se requiere una imagen en escala de grises")
    return np.stack([image, image, image], axis=-1)


def _merge_channels(channels: list[np.ndarray]) -> np.ndarray:
    if _HAS_CV2_MERGE:
        return cv2.merge(channels)
    return np.stack(channels, axis=-1)


def _bitwise_not(image: np.ndarray) -> np.ndarray:
    if _HAS_CV2_BITWISE_NOT:
        return cv2.bitwise_not(image)
    return np.bitwise_not(image)


def _bitwise_and(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    if _HAS_CV2_BITWISE_AND:
        return cv2.bitwise_and(lhs, rhs)
    return np.bitwise_and(lhs, rhs)


def _add_images(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    if _HAS_CV2_ADD:
        return cv2.add(lhs, rhs)
    return _ensure_uint8(lhs.astype(np.int32) + rhs.astype(np.int32))


def _gaussian_blur(image: np.ndarray, kernel: tuple[int, int], sigma: float) -> np.ndarray:
    if _HAS_CV2_GAUSSIAN:
        return cv2.GaussianBlur(image, kernel, sigma)

    kx, ky = kernel
    if kx % 2 == 0 or ky % 2 == 0:
        raise ValueError("Los kernels deben ser impares")

    pad_x = kx // 2
    pad_y = ky // 2
    if image.ndim == 2:
        padded = np.pad(image.astype(np.float32), ((pad_y, pad_y), (pad_x, pad_x)), mode="edge")
        out = np.empty_like(image, dtype=np.float32)
        window = kx * ky
        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                region = padded[row : row + ky, col : col + kx]
                out[row, col] = float(np.sum(region) / window)
        return _ensure_uint8(out)

    channels = [_gaussian_blur(image[:, :, idx], kernel, sigma) for idx in range(image.shape[2])]
    return np.stack(channels, axis=-1)


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """Crea el directorio indicado (y padres) si no existe."""

    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _resolve_task_asset(flag: str, provided: Optional[str]) -> str:
    if provided:
        model_path = Path(provided)
        if model_path.exists():
            return str(model_path)
        raise ValueError(f"No se encontró el modelo {flag}: {model_path}")

    if mp is None:
        raise ValueError(
            "MediaPipe no está disponible. Proporciona la ruta del modelo con "
            f"{flag} o instala los assets oficiales."
        )

    module_file = getattr(mp, "__file__", None)
    if not module_file:
        raise ValueError(
            "No fue posible localizar los modelos predeterminados de MediaPipe. "
            f"Proporciona la ruta manualmente con {flag}."
        )

    default_path = Path(module_file).resolve().parent / _TASK_DEFAULTS[flag]
    if default_path.exists():
        return str(default_path)

    raise ValueError(
        "No se encontró un modelo predeterminado para "
        f"{flag}: {default_path}. Proporciona la ruta manualmente."
    )


def _resolve_delegate_models(
    delegate: str,
    face_model: Optional[str],
    hand_model: Optional[str],
    pose_model: Optional[str],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if delegate != _DELEGATE_GPU:
        return face_model, hand_model, pose_model

    resolved_face = _resolve_task_asset("--face-model", face_model)
    resolved_hand = _resolve_task_asset("--hand-model", hand_model)
    resolved_pose = _resolve_task_asset("--pose-model", pose_model)
    return resolved_face, resolved_hand, resolved_pose


def expand_clamp_bbox(
    x: int,
    y: int,
    w: int,
    h: int,
    scale: float,
    frame_width: int,
    frame_height: int,
) -> Tuple[int, int, int, int]:
    """Amplía un *bounding box* y lo clampa al tamaño del frame."""

    if w <= 0 or h <= 0:
        return x, y, 0, 0

    cx = x + w / 2.0
    cy = y + h / 2.0
    new_w = w * scale
    new_h = h * scale

    x1 = max(0, int(round(cx - new_w / 2.0)))
    y1 = max(0, int(round(cy - new_h / 2.0)))
    x2 = min(frame_width, int(round(cx + new_w / 2.0)))
    y2 = min(frame_height, int(round(cy + new_h / 2.0)))

    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)


def crop_square(
    image: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    out_size: int = 224,
) -> np.ndarray:
    """Extrae un recorte cuadrado centrado en la región indicada."""

    if image.size == 0 or w <= 0 or h <= 0:
        return np.zeros((out_size, out_size, 3), dtype=image.dtype if image.size else np.uint8)

    height, width = image.shape[:2]
    side = int(round(max(w, h)))
    cx, cy = x + w // 2, y + h // 2

    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(width, x1 + side)
    y2 = min(height, y1 + side)

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((out_size, out_size, 3), dtype=image.dtype)

    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)


def hand_bbox_from_pose(
    pose_landmarks: Optional[object],
    indices: Tuple[int, int, int],
    frame_width: int,
    frame_height: int,
    scale: float = 1.2,
) -> Optional[Tuple[int, int, int, int]]:
    """Construye un *bounding box* cuadrado a partir de la pose."""

    if pose_landmarks is None:
        return None

    try:
        points = [pose_landmarks.landmark[idx] for idx in indices]
    except (AttributeError, IndexError):
        return None

    xs = [int(round(pt.x * frame_width)) for pt in points]
    ys = [int(round(pt.y * frame_height)) for pt in points]

    x_min = max(0, min(xs, default=0))
    y_min = max(0, min(ys, default=0))
    x_max = min(frame_width, max(xs, default=0))
    y_max = min(frame_height, max(ys, default=0))

    side = max(x_max - x_min, y_max - y_min)
    if side <= 0:
        return None

    x, y, w, h = expand_clamp_bbox(x_min, y_min, side, side, scale, frame_width, frame_height)
    if w <= 0 or h <= 0:
        return None
    return x, y, w, h


def resolve_hand_bbox(
    detected_bbox: Optional[Tuple[int, int, int, int]],
    pose_landmarks: Optional[object],
    indices: Tuple[int, int, int],
    prev_bbox: Optional[Tuple[int, int, int, int]],
    frame_width: int,
    frame_height: int,
    scale: float = 1.2,
) -> Tuple[Optional[Tuple[int, int, int, int]], str]:
    """Selecciona el *bounding box* más apropiado para una mano."""

    if detected_bbox and detected_bbox[2] > 0 and detected_bbox[3] > 0:
        return detected_bbox, "detected"

    pose_bbox = hand_bbox_from_pose(pose_landmarks, indices, frame_width, frame_height, scale)
    if pose_bbox is not None:
        return pose_bbox, "pose"

    if prev_bbox is not None and prev_bbox[2] > 0 and prev_bbox[3] > 0:
        return prev_bbox, "previous"

    return None, "black"


def face_bbox_from_pose(
    pose_landmarks: Optional[object],
    frame_width: int,
    frame_height: int,
    scale: float = 1.4,
) -> Optional[Tuple[int, int, int, int]]:
    """Construye un *bounding box* facial aproximado a partir de la pose."""

    if pose_landmarks is None:
        return None

    try:
        iterator = pose_landmarks.landmark  # type: ignore[attr-defined]
    except AttributeError:
        return None

    xs: List[int] = []
    ys: List[int] = []
    for idx in FACE_POSE_INDICES:
        try:
            landmark = iterator[idx]
        except (IndexError, TypeError):
            continue
        lx = getattr(landmark, "x", None)
        ly = getattr(landmark, "y", None)
        if lx is None or ly is None:
            continue
        xs.append(int(round(float(lx) * frame_width)))
        ys.append(int(round(float(ly) * frame_height)))

    if not xs or not ys:
        return None

    x_min = max(0, min(xs))
    y_min = max(0, min(ys))
    x_max = min(frame_width, max(xs))
    y_max = min(frame_height, max(ys))

    width = x_max - x_min
    height = y_max - y_min
    if width <= 0 or height <= 0:
        return None

    x, y, w, h = expand_clamp_bbox(
        x_min,
        y_min,
        width,
        height,
        scale,
        frame_width,
        frame_height,
    )
    if w <= 0 or h <= 0:
        return None
    return x, y, w, h


def resolve_face_bbox(
    detected_bbox: Optional[Tuple[int, int, int, int]],
    pose_landmarks: Optional[object],
    prev_bbox: Optional[Tuple[int, int, int, int]],
    frame_width: int,
    frame_height: int,
    scale: float = 1.4,
) -> Tuple[Optional[Tuple[int, int, int, int]], str]:
    """Selecciona el *bounding box* más apropiado para la cara."""

    if detected_bbox and detected_bbox[2] > 0 and detected_bbox[3] > 0:
        return detected_bbox, "detected"

    if prev_bbox is not None and prev_bbox[2] > 0 and prev_bbox[3] > 0:
        return prev_bbox, "previous"

    pose_bbox = face_bbox_from_pose(pose_landmarks, frame_width, frame_height, scale)
    if pose_bbox is not None:
        return pose_bbox, "pose"

    return None, "black"


def build_face_keep_mask(
    face_landmarks: Optional[object],
    bbox: Tuple[int, int, int, int],
    frame_shape: Tuple[int, int],
    keep_radius: int = 6,
) -> np.ndarray:
    """Genera una máscara binaria para preservar ojos y boca."""

    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return np.zeros((max(h, 0), max(w, 0)), dtype=np.uint8)

    mask = np.zeros((h, w), dtype=np.uint8)
    if face_landmarks is None:
        return mask

    try:
        iterator = face_landmarks.landmark  # type: ignore[attr-defined]
    except AttributeError:
        return mask

    frame_height, frame_width = frame_shape
    for idx in FACE_KEEP_INDICES:
        try:
            landmark = iterator[idx]
        except (IndexError, TypeError):
            continue
        lx = getattr(landmark, "x", None)
        ly = getattr(landmark, "y", None)
        if lx is None or ly is None:
            continue
        px = int(round(float(lx) * frame_width)) - x
        py = int(round(float(ly) * frame_height)) - y
        if 0 <= px < w and 0 <= py < h:
            _fill_circle(mask, (px, py), keep_radius, 255)

    return mask


def apply_face_partial_grayscale(patch: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Convierte a escala de grises fuera de las zonas preservadas."""

    if patch.size == 0:
        return patch

    gray = _bgr_to_gray(patch)
    gray_bgr = _gray_to_bgr(gray)
    if mask.size == 0 or mask.max() == 0:
        return gray_bgr

    mask3 = _merge_channels([mask, mask, mask])
    inv_mask = _bitwise_not(mask3)
    color_part = _bitwise_and(patch, mask3)
    gray_part = _bitwise_and(gray_bgr, inv_mask)
    return _add_images(color_part, gray_part)


def blur_face_preserve_eyes_mouth(patch: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Aplica desenfoque manteniendo intactas las zonas preservadas."""

    if patch.size == 0:
        return patch

    gray_full = _bgr_to_gray(patch)
    gray_full_bgr = _gray_to_bgr(gray_full)

    if mask.size == 0 or mask.max() == 0:
        return _gaussian_blur(gray_full_bgr, (31, 31), 0)

    mask3 = _merge_channels([mask, mask, mask])
    inv_mask = _bitwise_not(mask3)
    blurred = _gaussian_blur(gray_full_bgr, (31, 31), 0)
    kept = _bitwise_and(patch, mask3)
    blurred_rest = _bitwise_and(blurred, inv_mask)
    return _add_images(kept, blurred_rest)


def _configure_mediapipe_logging(level: str) -> None:
    global _configured_mp_log_level

    resolved = level.strip().lower()
    if resolved not in _MP_LOG_LEVELS:
        raise ValueError(
            "Nivel de log de MediaPipe no soportado. Usa info, warning, error o fatal."
        )

    if _configured_mp_log_level == resolved:
        return

    level_values = _MP_LOG_LEVELS[resolved]
    os.environ["GLOG_minloglevel"] = level_values["env"]
    os.environ["ABSL_MIN_LOG_LEVEL"] = level_values["env"]

    try:  # pragma: no cover - dependencias opcionales
        from absl import logging as absl_logging
    except Exception:  # pragma: no cover - dependencias opcionales
        pass
    else:
        absl_verbosity = level_values["absl"]
        absl_logging.set_stderrthreshold(absl_verbosity)
        absl_logging.set_verbosity(int(level_values["env"]))

    _configured_mp_log_level = resolved


def _ensure_mediapipe_available() -> bool:
    if mp is None:  # pragma: no cover - dependencias opcionales
        warnings.warn(_MP_WARNING)
        return False
    return True


def _normalise_format(value: str) -> str:
    fmt = value.lower().strip()
    if fmt in {"jpg", "jpeg"}:
        return "jpg"
    if fmt == "png":
        return "png"
    if fmt == "npz":
        return "npz"
    raise ValueError("Formato no soportado. Usa 'jpg', 'png' o 'npz'.")


def _normalise_keypoints_format(value: str) -> str:
    fmt = value.lower().strip()
    if fmt in {"npz", "npy"}:
        return fmt
    raise ValueError("Formato de keypoints no soportado. Usa 'npz' o 'npy'.")


def _normalise_streams(selection: Optional[Iterable[str]]) -> Set[str]:
    mapping = {
        "face": {"face"},
        "hands": {"hand_l", "hand_r"},
        "hand_l": {"hand_l"},
        "hand-left": {"hand_l"},
        "left": {"hand_l"},
        "hand_r": {"hand_r"},
        "hand-right": {"hand_r"},
        "right": {"hand_r"},
        "pose": {"pose"},
        "all": {"face", "hand_l", "hand_r", "pose"},
    }

    if selection is None:
        return {"face", "hand_l", "hand_r", "pose"}

    resolved: Set[str] = set()
    for item in selection:
        key = item.lower().strip()
        if key not in mapping:
            raise ValueError(f"Stream desconocido: {item}")
        resolved.update(mapping[key])

    if not resolved:
        raise ValueError("Debes seleccionar al menos un stream válido")

    return resolved


def _normalise_delegate(value: Optional[str]) -> str:
    if value is None:
        return _DELEGATE_CPU

    delegate = value.strip().lower()
    if delegate in {_DELEGATE_CPU, _DELEGATE_GPU}:
        return delegate

    raise ValueError("Delegate no soportado. Usa 'cpu' o 'gpu'.")


@dataclass
class _Landmark:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 0.0


@dataclass
class _LandmarkList:
    landmark: List[_Landmark]


@dataclass
class _Classification:
    label: str


@dataclass
class _Handedness:
    classification: List[_Classification]


@dataclass
class _FaceResult:
    multi_face_landmarks: List[_LandmarkList]


@dataclass
class _HandsResult:
    multi_hand_landmarks: List[_LandmarkList]
    multi_handedness: List[_Handedness]


@dataclass
class _PoseResult:
    pose_landmarks: Optional[_LandmarkList]


class _PipelineBase:
    def process(self, rgb_frame: np.ndarray) -> Tuple[object, object, object]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def __enter__(self) -> "_PipelineBase":
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self.close()


class _CpuMediaPipePipeline(_PipelineBase):
    def __init__(self) -> None:
        if mp is None:
            raise RuntimeError("MediaPipe no está disponible")

        self._face = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
        )
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
        )
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
        )

    def process(self, rgb_frame: np.ndarray) -> Tuple[object, object, object]:
        face_result = self._face.process(rgb_frame)
        hands_result = self._hands.process(rgb_frame)
        pose_result = self._pose.process(rgb_frame)
        return face_result, hands_result, pose_result

    def close(self) -> None:
        self._face.close()
        self._hands.close()
        self._pose.close()


def _wrap_landmarks(sequence: Iterable[object]) -> List[_Landmark]:
    wrapped: List[_Landmark] = []
    for landmark in sequence:
        wrapped.append(
            _Landmark(
                float(getattr(landmark, "x", 0.0)),
                float(getattr(landmark, "y", 0.0)),
                float(getattr(landmark, "z", 0.0)),
                float(getattr(landmark, "visibility", 0.0)),
            )
        )
    return wrapped


def _wrap_landmark_lists(raw: Optional[Iterable[Iterable[object]]]) -> List[_LandmarkList]:
    if raw is None:
        return []

    wrapped: List[_LandmarkList] = []
    for group in raw:
        wrapped.append(_LandmarkList(_wrap_landmarks(group)))
    return wrapped


def _wrap_pose_landmarks(raw: Optional[Iterable[Iterable[object]]]) -> Optional[_LandmarkList]:
    lists = _wrap_landmark_lists(raw)
    if not lists:
        return None
    return lists[0]


def _wrap_handedness(raw: Optional[Iterable[Iterable[object]]]) -> List[_Handedness]:
    if raw is None:
        return []

    handedness: List[_Handedness] = []
    for group in raw:
        classifications: List[_Classification] = []
        for item in group:
            label = getattr(item, "category_name", None) or getattr(item, "display_name", None)
            if not label:
                label = getattr(item, "label", "")
            classifications.append(_Classification(str(label)))
        handedness.append(_Handedness(classifications))
    return handedness


class _GpuMediaPipePipeline(_PipelineBase):
    def __init__(
        self,
        face_model: str,
        hand_model: str,
        pose_model: str,
    ) -> None:
        if mp is None or mp_tasks_python is None or mp_vision is None:
            raise RuntimeError("MediaPipe Tasks no está disponible para GPU")

        delegate = mp_tasks_python.BaseOptions.Delegate.GPU
        running_mode = mp_vision.RunningMode.IMAGE

        face_options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_tasks_python.BaseOptions(
                model_asset_path=face_model,
                delegate=delegate,
            ),
            running_mode=running_mode,
            num_faces=1,
        )
        hand_options = mp_vision.HandLandmarkerOptions(
            base_options=mp_tasks_python.BaseOptions(
                model_asset_path=hand_model,
                delegate=delegate,
            ),
            running_mode=running_mode,
            num_hands=2,
        )
        pose_options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_tasks_python.BaseOptions(
                model_asset_path=pose_model,
                delegate=delegate,
            ),
            running_mode=running_mode,
            output_segmentation_masks=False,
        )

        self._face = mp_vision.FaceLandmarker.create_from_options(face_options)
        self._hands = mp_vision.HandLandmarker.create_from_options(hand_options)
        self._pose = mp_vision.PoseLandmarker.create_from_options(pose_options)

    def process(self, rgb_frame: np.ndarray) -> Tuple[_FaceResult, _HandsResult, _PoseResult]:
        if mp is None:
            raise RuntimeError("MediaPipe no está disponible")

        rgb_contiguous = np.ascontiguousarray(rgb_frame)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_contiguous)

        face_raw = self._face.detect(mp_image)
        hands_raw = self._hands.detect(mp_image)
        pose_raw = self._pose.detect(mp_image)

        face_result = _FaceResult(
            multi_face_landmarks=_wrap_landmark_lists(getattr(face_raw, "face_landmarks", None))
        )
        hands_result = _HandsResult(
            multi_hand_landmarks=_wrap_landmark_lists(getattr(hands_raw, "hand_landmarks", None)),
            multi_handedness=_wrap_handedness(getattr(hands_raw, "handedness", None)),
        )
        pose_result = _PoseResult(
            pose_landmarks=_wrap_pose_landmarks(getattr(pose_raw, "pose_landmarks", None))
        )

        return face_result, hands_result, pose_result

    def close(self) -> None:
        self._face.close()
        self._hands.close()
        self._pose.close()


def _create_pipeline(
    delegate: str,
    face_model: Optional[str],
    hand_model: Optional[str],
    pose_model: Optional[str],
) -> _PipelineBase:
    if delegate == _DELEGATE_GPU:
        if not face_model or not hand_model or not pose_model:
            raise ValueError(
                "No se pudieron resolver los modelos .task necesarios para ejecutar en GPU."
            )
        return _GpuMediaPipePipeline(face_model, hand_model, pose_model)

    return _CpuMediaPipePipeline()


def process_video(
    video_path: str,
    out_dirs: Dict[str, Path],
    pose_dir: Optional[Path],
    keypoints_dir: Optional[Path] = None,
    fps_target: int = 25,
    face_blur: bool = False,
    fps_limit: Optional[float] = None,
    streams: Optional[Iterable[str]] = None,
    image_format: str = "jpg",
    delegate: str = _DELEGATE_CPU,
    face_model: Optional[str] = None,
    hand_model: Optional[str] = None,
    pose_model: Optional[str] = None,
    mp_log_level: str = _DEFAULT_MP_LOG_LEVEL,
    export_keypoints: bool = False,
    keypoints_format: str = "npz",
) -> Dict[str, object]:
    """Procesa un único video y guarda los ROIs correspondientes.

    Devuelve un diccionario de metadata con métricas de procesamiento.
    """

    resolved_streams = _normalise_streams(streams)
    resolved_format = _normalise_format(image_format)
    resolved_keypoints_format = _normalise_keypoints_format(keypoints_format)
    resolved_delegate = _normalise_delegate(delegate)
    face_model, hand_model, pose_model = _resolve_delegate_models(
        resolved_delegate,
        face_model,
        hand_model,
        pose_model,
    )
    export_face = "face" in resolved_streams
    export_hand_l = "hand_l" in resolved_streams
    export_hand_r = "hand_r" in resolved_streams
    export_pose = "pose" in resolved_streams

    _configure_mediapipe_logging(mp_log_level)

    metadata = {
        "video": Path(video_path).name,
        "video_path": video_path,
        "success": False,
        "error": None,
        "fps_source": None,
        "fps_target": fps_target,
        "fps_limit": fps_limit,
        "frames_written": 0,
        "pose_frames": 0,
        "keypoints_frames": 0,
        "stride": None,
        "face_blur": face_blur,
        "streams": sorted(resolved_streams),
        "delegate": resolved_delegate,
        "mp_log_level": mp_log_level,
        "keypoints_format": resolved_keypoints_format if export_keypoints else None,
        "keypoints_layout": KEYPOINT_LAYOUT_NAME if export_keypoints else None,
    }

    fallback_template = {"pose": 0, "previous": 0, "black": 0}
    metadata["fallbacks"] = {}
    if export_face:
        metadata["fallbacks"]["face"] = dict(fallback_template)
    if export_hand_l:
        metadata["fallbacks"]["hand_left"] = dict(fallback_template)
    if export_hand_r:
        metadata["fallbacks"]["hand_right"] = dict(fallback_template)

    if not _ensure_mediapipe_available():  # pragma: no cover - dependencias opcionales
        metadata["error"] = "mediapipe-no-disponible"
        return metadata

    try:
        pipeline = _create_pipeline(
            resolved_delegate,
            face_model=face_model,
            hand_model=hand_model,
            pose_model=pose_model,
        )
    except Exception as exc:
        metadata["error"] = str(exc)
        warnings.warn(str(exc))
        return metadata

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        msg = f"No se pudo abrir el video: {video_path}"
        warnings.warn(msg)
        metadata["error"] = "video-no-abre"
        pipeline.close()
        return metadata

    try:
        basename = Path(video_path).stem
        face_out = ensure_dir(out_dirs["face"]) if export_face and "face" in out_dirs else None
        hand_l_out = (
            ensure_dir(out_dirs["hand_l"]) if export_hand_l and "hand_l" in out_dirs else None
        )
        hand_r_out = (
            ensure_dir(out_dirs["hand_r"]) if export_hand_r and "hand_r" in out_dirs else None
        )
        pose_out = ensure_dir(pose_dir) if export_pose and pose_dir is not None else None
        keypoints_out = (
            ensure_dir(keypoints_dir) if export_keypoints and keypoints_dir is not None else None
        )

        fps = cap.get(cv2.CAP_PROP_FPS) or fps_target
        metadata["fps_source"] = fps
        if fps_limit and fps > fps_limit:
            fps = fps_limit
        stride = max(1, int(round(fps / fps_target)))
        metadata["stride"] = stride

        pose_frames: List[np.ndarray] = []
        prev_pose_norm: Optional[np.ndarray] = None
        prev_face_bbox: Optional[Tuple[int, int, int, int]] = None
        prev_left_bbox: Optional[Tuple[int, int, int, int]] = None
        prev_right_bbox: Optional[Tuple[int, int, int, int]] = None

        face_buffer: List[np.ndarray] = []
        hand_l_buffer: List[np.ndarray] = []
        hand_r_buffer: List[np.ndarray] = []
        keypoints_buffer: List[np.ndarray] = []
        image_ext = f".{resolved_format}" if resolved_format in {"jpg", "png"} else ""

        with pipeline as detectors:
            frame_index = 0
            out_index = 0

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if frame_index % stride != 0:
                    frame_index += 1
                    continue

                height, width = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                face_result, hands_result, pose_result = detectors.process(rgb)
                pose_landmarks = getattr(pose_result, "pose_landmarks", None)

                # Cara
                face_crop = np.zeros((224, 224, 3), dtype=frame.dtype)
                face_landmarks = (
                    face_result.multi_face_landmarks[0]
                    if face_result.multi_face_landmarks
                    else None
                )
                detected_face_bbox: Optional[Tuple[int, int, int, int]] = None
                if face_landmarks is not None:
                    xs = [int(landmark.x * width) for landmark in face_landmarks.landmark]
                    ys = [int(landmark.y * height) for landmark in face_landmarks.landmark]
                    x1 = max(0, min(xs))
                    y1 = max(0, min(ys))
                    x2 = min(width, max(xs))
                    y2 = min(height, max(ys))
                    detected_face_bbox = expand_clamp_bbox(
                        x1,
                        y1,
                        x2 - x1,
                        y2 - y1,
                        1.2,
                        width,
                        height,
                    )

                face_bbox, face_source = resolve_face_bbox(
                    detected_face_bbox,
                    pose_landmarks,
                    prev_face_bbox,
                    width,
                    height,
                )
                if export_face and face_source != "detected":
                    metadata["fallbacks"]["face"][face_source] += 1

                if face_bbox is not None and face_bbox[2] > 0 and face_bbox[3] > 0:
                    x, y, w, h = face_bbox
                    face_patch = frame[y : y + h, x : x + w]
                    mask = build_face_keep_mask(face_landmarks, face_bbox, (height, width))
                    gray_patch = apply_face_partial_grayscale(face_patch, mask)
                    processed_patch = (
                        blur_face_preserve_eyes_mouth(gray_patch, mask)
                        if face_blur
                        else gray_patch
                    )
                    face_crop = crop_square(
                        processed_patch,
                        0,
                        0,
                        processed_patch.shape[1],
                        processed_patch.shape[0],
                        224,
                    )
                    prev_face_bbox = face_bbox
                else:
                    prev_face_bbox = None

                if export_face and face_out is not None:
                    if resolved_format == "npz":
                        face_buffer.append(face_crop)
                    else:
                        cv2.imwrite(
                            str(face_out / f"{basename}_f{out_index:06d}{image_ext}"),
                            face_crop,
                        )

                # Manos
                left_detected: Optional[Tuple[int, int, int, int]] = None
                right_detected: Optional[Tuple[int, int, int, int]] = None
                if hands_result.multi_hand_landmarks and hands_result.multi_handedness:
                    for landmarks, handedness in zip(
                        hands_result.multi_hand_landmarks, hands_result.multi_handedness
                    ):
                        xs = [int(l.x * width) for l in landmarks.landmark]
                        ys = [int(l.y * height) for l in landmarks.landmark]
                        x1 = max(0, min(xs))
                        y1 = max(0, min(ys))
                        x2 = min(width, max(xs))
                        y2 = min(height, max(ys))
                        bbox = expand_clamp_bbox(x1, y1, x2 - x1, y2 - y1, 1.2, width, height)
                        label = handedness.classification[0].label.lower()
                        if label.startswith("left"):
                            left_detected = bbox
                        else:
                            right_detected = bbox

                left_bbox, left_source = resolve_hand_bbox(
                    left_detected,
                    pose_landmarks,
                    (17, 19, 21),
                    prev_left_bbox,
                    width,
                    height,
                )
                right_bbox, right_source = resolve_hand_bbox(
                    right_detected,
                    pose_landmarks,
                    (18, 20, 22),
                    prev_right_bbox,
                    width,
                    height,
                )

                if export_hand_l and left_source != "detected":
                    metadata["fallbacks"]["hand_left"][left_source] += 1
                if export_hand_r and right_source != "detected":
                    metadata["fallbacks"]["hand_right"][right_source] += 1

                if export_hand_l:
                    if left_bbox is not None:
                        left_crop = crop_square(frame, *left_bbox, 224)
                        prev_left_bbox = left_bbox if left_bbox[2] > 0 and left_bbox[3] > 0 else None
                    else:
                        left_crop = np.zeros((224, 224, 3), dtype=frame.dtype)
                        prev_left_bbox = None
                    if hand_l_out is not None:
                        if resolved_format == "npz":
                            hand_l_buffer.append(left_crop)
                        else:
                            cv2.imwrite(
                                str(hand_l_out / f"{basename}_f{out_index:06d}{image_ext}"),
                                left_crop,
                            )
                else:
                    prev_left_bbox = None

                if export_hand_r:
                    if right_bbox is not None:
                        right_crop = crop_square(frame, *right_bbox, 224)
                        prev_right_bbox = right_bbox if right_bbox[2] > 0 and right_bbox[3] > 0 else None
                    else:
                        right_crop = np.zeros((224, 224, 3), dtype=frame.dtype)
                        prev_right_bbox = None
                    if hand_r_out is not None:
                        if resolved_format == "npz":
                            hand_r_buffer.append(right_crop)
                        else:
                            cv2.imwrite(
                                str(hand_r_out / f"{basename}_f{out_index:06d}{image_ext}"),
                                right_crop,
                            )
                else:
                    prev_right_bbox = None

                if export_keypoints:
                    left_landmarks_obj, right_landmarks_obj = _resolve_hands_landmarks(
                        hands_result
                    )
                    body_arr = _landmark_list_to_array(
                        pose_landmarks,
                        KEYPOINT_BODY_LANDMARKS,
                        default_confidence=0.0,
                    )
                    face_arr = _landmark_list_to_array(
                        face_landmarks,
                        KEYPOINT_FACE_LANDMARKS,
                        default_confidence=1.0,
                    )
                    left_arr = _landmark_list_to_array(
                        left_landmarks_obj,
                        KEYPOINT_HAND_LANDMARKS,
                        default_confidence=1.0,
                    )
                    right_arr = _landmark_list_to_array(
                        right_landmarks_obj,
                        KEYPOINT_HAND_LANDMARKS,
                        default_confidence=1.0,
                    )
                    frame_keypoints = np.concatenate(
                        [body_arr, face_arr, left_arr, right_arr], axis=0
                    )
                    keypoints_buffer.append(frame_keypoints)

                # Pose
                pose_landmarks = getattr(pose_result, "pose_landmarks", None)
                pose_vec: Optional[np.ndarray] = None
                if pose_landmarks and getattr(pose_landmarks, "landmark", None):
                    pose_vec = normalize_pose_landmarks(
                        pose_landmarks,
                        landmark_count=POSE_LANDMARK_COUNT,
                    )
                    prev_pose_norm = pose_vec.copy()
                elif prev_pose_norm is not None:
                    pose_vec = prev_pose_norm.copy()
                else:
                    pose_vec = sentinel_pose(POSE_LANDMARK_COUNT)
                if export_pose:
                    pose_frames.append(pose_vec.reshape(-1))

                out_index += 1
                frame_index += 1

        pipeline = None
        metadata["frames_written"] = out_index
        metadata["pose_frames"] = len(pose_frames) if export_pose else 0

        cap.release()

        if export_pose and pose_out is not None:
            pose_array = np.asarray(pose_frames, dtype=np.float32)
            np.savez_compressed(
                pose_out / f"{basename}.npz",
                pose=pose_array,
                pose_norm=np.asarray("signing_space_v1", dtype=np.str_),
            )

        if export_face and resolved_format == "npz" and face_out is not None:
            face_array = (
                np.stack(face_buffer, axis=0)
                if face_buffer
                else np.zeros((0, 224, 224, 3), dtype=np.uint8)
            )
            np.savez_compressed(face_out / f"{basename}.npz", frames=face_array)
        if export_hand_l and resolved_format == "npz" and hand_l_out is not None:
            hand_l_array = (
                np.stack(hand_l_buffer, axis=0)
                if hand_l_buffer
                else np.zeros((0, 224, 224, 3), dtype=np.uint8)
            )
            np.savez_compressed(hand_l_out / f"{basename}.npz", frames=hand_l_array)
        if export_hand_r and resolved_format == "npz" and hand_r_out is not None:
            hand_r_array = (
                np.stack(hand_r_buffer, axis=0)
                if hand_r_buffer
                else np.zeros((0, 224, 224, 3), dtype=np.uint8)
            )
            np.savez_compressed(hand_r_out / f"{basename}.npz", frames=hand_r_array)
        if export_keypoints and keypoints_out is not None:
            keypoints_array = (
                np.stack(keypoints_buffer, axis=0)
                if keypoints_buffer
                else np.zeros((0, KEYPOINT_TOTAL_LANDMARKS, 3), dtype=np.float32)
            )
            keypoints_path = keypoints_out / f"{basename}.{resolved_keypoints_format}"
            if resolved_keypoints_format == "npz":
                np.savez_compressed(
                    keypoints_path,
                    keypoints=keypoints_array,
                    layout=np.asarray(KEYPOINT_LAYOUT_NAME, dtype=np.str_),
                )
            else:
                np.save(keypoints_path, keypoints_array)
            metadata["keypoints_frames"] = keypoints_array.shape[0]
        metadata["success"] = True
        return metadata

    except Exception as exc:  # pragma: no cover - flujo inesperado
        metadata["error"] = str(exc)
        return metadata
    finally:
        cap.release()
        try:
            if "pipeline" in locals() and pipeline is not None:
                pipeline.close()
        except Exception:
            pass


def _metadata_path(out_root: str, metadata_path: Optional[str]) -> Path:
    if metadata_path:
        return ensure_dir(Path(metadata_path).parent) / Path(metadata_path).name
    return ensure_dir(out_root) / "metadata.jsonl"


def _read_metadata_index(path: Path) -> Dict[str, Dict[str, object]]:
    index: Dict[str, Dict[str, object]] = {}
    if not path.exists():
        return index
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            video_name = entry.get("video")
            if isinstance(video_name, str):
                index[video_name] = entry
    return index


def _append_metadata(path: Path, entry: Dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_bulk(
    videos_dir: str,
    out_root: str,
    fps_target: int = 25,
    face_blur: bool = False,
    resume: bool = False,
    metadata_path: Optional[str] = None,
    fps_limit: Optional[float] = None,
    streams: Optional[Iterable[str]] = None,
    image_format: str = "jpg",
    delegate: str = _DELEGATE_CPU,
    face_model: Optional[str] = None,
    hand_model: Optional[str] = None,
    pose_model: Optional[str] = None,
    mp_log_level: str = _DEFAULT_MP_LOG_LEVEL,
    export_keypoints: bool = False,
    keypoints_output: Optional[str] = None,
    keypoints_format: str = "npz",
) -> None:
    """Procesa todos los videos *.mp4 en ``videos_dir``."""

    _configure_mediapipe_logging(mp_log_level)

    if not _ensure_mediapipe_available():  # pragma: no cover - dependencias opcionales
        return

    resolved_streams = _normalise_streams(streams)
    resolved_format = _normalise_format(image_format)
    resolved_delegate = _normalise_delegate(delegate)
    resolved_keypoints_format = _normalise_keypoints_format(keypoints_format)
    face_model, hand_model, pose_model = _resolve_delegate_models(
        resolved_delegate,
        face_model,
        hand_model,
        pose_model,
    )

    videos_path = Path(videos_dir)
    if not videos_path.exists():
        warnings.warn(f"Directorio no encontrado: {videos_dir}")
        return

    out_root_path = Path(out_root)
    out_dirs: Dict[str, Path] = {}
    if "face" in resolved_streams:
        out_dirs["face"] = ensure_dir(out_root_path / "face")
    if "hand_l" in resolved_streams:
        out_dirs["hand_l"] = ensure_dir(out_root_path / "hand_l")
    if "hand_r" in resolved_streams:
        out_dirs["hand_r"] = ensure_dir(out_root_path / "hand_r")

    pose_dir = ensure_dir(out_root_path / "pose") if "pose" in resolved_streams else None
    keypoints_dir: Optional[Path] = None
    if export_keypoints:
        base_dir = Path(keypoints_output) if keypoints_output else out_root_path / "keypoints"
        keypoints_dir = ensure_dir(base_dir)

    meta_file = _metadata_path(out_root, metadata_path)
    index = _read_metadata_index(meta_file)

    processed = []
    errors = []

    for video_path in sorted(videos_path.glob("*.mp4")):
        video_name = video_path.name
        if resume and video_name in index and index[video_name].get("success"):
            print(f"Omitiendo {video_name} (ya procesado)")
            continue

        print(f"Procesando {video_name}")
        entry = process_video(
            str(video_path),
            out_dirs,
            pose_dir,
            keypoints_dir,
            fps_target=fps_target,
            face_blur=face_blur,
            fps_limit=fps_limit,
            streams=resolved_streams,
            image_format=resolved_format,
            delegate=resolved_delegate,
            face_model=face_model,
            hand_model=hand_model,
            pose_model=pose_model,
            mp_log_level=mp_log_level,
            export_keypoints=export_keypoints,
            keypoints_format=resolved_keypoints_format,
        )
        entry["video"] = video_name
        entry["video_path"] = str(video_path)
        _append_metadata(meta_file, entry)
        if entry.get("success"):
            processed.append(entry)
        else:
            errors.append(entry)

    if errors:
        print("\nErrores detectados:")
        for item in errors:
            print(f"- {item['video']}: {item.get('error')}")

    print(
        f"\nProcesamiento finalizado. OK: {len(processed)}, "
        f"Errores: {len(errors)}. Metadata: {meta_file}"
    )


POSE_LANDMARK_COUNT = 17
POSE_SIGNING_WIDTH = 6.0
POSE_SIGNING_HEIGHT = 7.0


def _as_float32(value: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _build_pose_array(
    pose_landmarks: Optional[object],
    landmark_count: int = POSE_LANDMARK_COUNT,
) -> np.ndarray:
    """Convierte los landmarks de MediaPipe en un arreglo ``(N, 3)``."""

    arr = np.zeros((landmark_count, 3), dtype=np.float32)
    if pose_landmarks is None:
        return arr

    try:
        iterator: Iterable[object] = pose_landmarks.landmark  # type: ignore[attr-defined]
    except AttributeError:
        return arr

    for idx, landmark in enumerate(iterator):
        if idx >= landmark_count:
            break
        arr[idx, 0] = _as_float32(getattr(landmark, "x", 0.0))
        arr[idx, 1] = _as_float32(getattr(landmark, "y", 0.0))
        arr[idx, 2] = _as_float32(getattr(landmark, "visibility", 0.0))
    return arr


def _landmark_list_to_array(
    landmarks: Optional[object],
    expected_count: int,
    *,
    default_confidence: float = 0.0,
) -> np.ndarray:
    """Convierte una lista de landmarks en ``(N, 3)`` con ``(x, y, conf)``."""

    arr = np.zeros((expected_count, 3), dtype=np.float32)
    if landmarks is None:
        return arr

    iterator = getattr(landmarks, "landmark", None)
    if iterator is None:
        return arr

    for idx, landmark in enumerate(iterator):
        if idx >= expected_count:
            break
        arr[idx, 0] = _as_float32(getattr(landmark, "x", 0.0))
        arr[idx, 1] = _as_float32(getattr(landmark, "y", 0.0))
        raw_conf = getattr(landmark, "visibility", None)
        if raw_conf is None:
            conf = default_confidence
        else:
            try:
                conf = float(raw_conf)
            except (TypeError, ValueError):
                conf = default_confidence
        if not np.isfinite(conf) or conf <= 0.0:
            conf = default_confidence
        arr[idx, 2] = _as_float32(conf)
    return arr


def _resolve_hands_landmarks(
    hands_result: object,
) -> Tuple[Optional[object], Optional[object]]:
    """Devuelve los landmarks detectados para la mano izquierda y derecha."""

    left_landmarks: Optional[object] = None
    right_landmarks: Optional[object] = None

    multi_landmarks = getattr(hands_result, "multi_hand_landmarks", None)
    multi_handedness = getattr(hands_result, "multi_handedness", None)
    if not multi_landmarks or not multi_handedness:
        return None, None

    for landmarks, handedness in zip(multi_landmarks, multi_handedness):
        classifications = getattr(handedness, "classification", None)
        if not classifications:
            continue
        label = str(getattr(classifications[0], "label", "")).lower()
        if label.startswith("left"):
            left_landmarks = landmarks
        else:
            right_landmarks = landmarks

    return left_landmarks, right_landmarks


def _head_unit_and_center(coords: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Calcula centro y unidad de cabeza para normalizar la pose."""

    landmark_count = coords.shape[0]
    if landmark_count == 0:
        center = np.array([0.5, 0.5], dtype=np.float32)
        return center, 1.0, 1.0

    nose = coords[0]
    shoulders = []
    if landmark_count > 11:
        shoulders.append(coords[11])
    if landmark_count > 12:
        shoulders.append(coords[12])

    if shoulders:
        center = np.mean(np.stack(shoulders, axis=0), axis=0)
    else:
        center = nose

    head_unit = float(np.linalg.norm(nose - center))
    if not np.isfinite(head_unit) or head_unit <= 1e-6:
        # Fallback: distancia interocular
        if landmark_count > 5:
            left_eye = coords[min(2, landmark_count - 1)]
            right_eye = coords[min(5, landmark_count - 1)]
            head_unit = float(np.linalg.norm(left_eye - right_eye))
    if not np.isfinite(head_unit) or head_unit <= 1e-6:
        head_unit = 1.0 / POSE_SIGNING_HEIGHT

    width = max(head_unit * POSE_SIGNING_WIDTH, 1e-6)
    height = max(head_unit * POSE_SIGNING_HEIGHT, 1e-6)
    return center.astype(np.float32), float(width), float(height)


def normalize_pose_landmarks(
    pose_landmarks: Optional[object],
    landmark_count: int = POSE_LANDMARK_COUNT,
) -> np.ndarray:
    """Normaliza los landmarks de pose al rango ``[0, 1]`` dentro del signing space."""

    pose_arr = _build_pose_array(pose_landmarks, landmark_count)
    coords = pose_arr[:, :2]
    center, width, height = _head_unit_and_center(coords)

    if width <= 0 or height <= 0:
        width, height = 1.0, 1.0

    normalized = (coords - center[None, :])
    normalized[:, 0] /= width
    normalized[:, 1] /= height
    normalized += 0.5
    np.clip(normalized, 0.0, 1.0, out=normalized)
    pose_arr[:, :2] = normalized
    return pose_arr


def sentinel_pose(landmark_count: int = POSE_LANDMARK_COUNT) -> np.ndarray:
    """Devuelve un vector sentinel para frames sin landmarks."""

    sentinel = np.full((landmark_count, 3), -1.0, dtype=np.float32)
    sentinel[:, 2] = 0.0
    return sentinel


if __name__ == "__main__":  # pragma: no cover - ejecución manual
    parser = argparse.ArgumentParser(description="Extracción de ROIs (cara/manos/pose)")
    parser.add_argument(
        "videos_dir",
        nargs="?",
        help="Directorio con videos .mp4 (posicional o --videos)",
    )
    parser.add_argument(
        "out_root",
        nargs="?",
        help=(
            "Directorio destino para los recortes (por ejemplo "
            "data/single_signer/processed)"
        ),
    )
    parser.add_argument(
        "--videos",
        dest="videos_opt",
        help="Directorio con videos .mp4 (equivalente al posicional videos_dir)",
    )
    parser.add_argument(
        "--output",
        "--out",
        dest="out_opt",
        help=(
            "Directorio destino para los recortes MediaPipe; equivale al posicional "
            "out_root"
        ),
    )
    parser.add_argument("--fps", type=int, default=25, help="FPS de muestreo")
    parser.add_argument(
        "--face-blur",
        action="store_true",
        help="Aplica desenfoque de cara conservando ojos y boca",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Omite videos que ya tengan metadata de éxito registrada",
    )
    parser.add_argument(
        "--metadata",
        help="Ruta del archivo JSONL para registrar metadata (por defecto en out_root)",
    )
    parser.add_argument(
        "--fps-limit",
        type=float,
        help="FPS máximo leído desde el video original antes de aplicar el muestreo",
    )
    parser.add_argument(
        "--format",
        default="jpg",
        help="Formato para cara/manos: jpg, png o npz",
    )
    parser.add_argument(
        "--streams",
        nargs="+",
        help=(
            "Streams a exportar (face, hand_l, hand_r, hands, pose, all)."
        ),
    )
    parser.add_argument(
        "--delegate",
        choices=[_DELEGATE_CPU, _DELEGATE_GPU],
        default=_DELEGATE_CPU,
        help="Delegate de MediaPipe a utilizar (cpu o gpu).",
    )
    parser.add_argument(
        "--mp-log-level",
        choices=sorted(_MP_LOG_LEVELS.keys()),
        default=_DEFAULT_MP_LOG_LEVEL,
        help="Nivel mínimo para los logs de MediaPipe (info, warning, error o fatal).",
    )
    parser.add_argument(
        "--face-model",
        help=(
            "Ruta al modelo .task para FaceLandmarker. "
            "Por defecto se usa el asset incluido en MediaPipe."
        ),
    )
    parser.add_argument(
        "--hand-model",
        help=(
            "Ruta al modelo .task para HandLandmarker. "
            "Por defecto se usa el asset incluido en MediaPipe."
        ),
    )
    parser.add_argument(
        "--pose-model",
        help=(
            "Ruta al modelo .task para PoseLandmarker. "
            "Por defecto se usa el asset incluido en MediaPipe."
        ),
    )
    parser.add_argument(
        "--export-keypoints",
        action="store_true",
        help=(
            "Genera keypoints MediaPipe (pose+cara+manos) en processed/keypoints/ "
            "o en la ruta indicada."
        ),
    )
    parser.add_argument(
        "--keypoints-output",
        help=(
            "Directorio destino para los keypoints (.npz/.npy). "
            "Por defecto se usa <out_root>/keypoints."
        ),
    )
    parser.add_argument(
        "--keypoints-format",
        choices=["npz", "npy"],
        default="npz",
        help="Formato de serialización para los keypoints (npz o npy).",
    )

    args = parser.parse_args()

    videos_dir = args.videos_opt or args.videos_dir
    out_root = args.out_opt or args.out_root

    if args.videos_dir and args.videos_opt and args.videos_dir != args.videos_opt:
        parser.error("Usa solo uno de los parámetros videos_dir o --videos")
    if args.out_root and args.out_opt and args.out_root != args.out_opt:
        parser.error("Usa solo uno de los parámetros out_root o --output/--out")
    if not videos_dir:
        parser.error("Debes especificar el directorio de videos (posicional o --videos)")
    if not out_root:
        parser.error("Debes especificar el directorio de salida (posicional o --output)")

    try:
        run_bulk(
            videos_dir,
            out_root,
            fps_target=args.fps,
            face_blur=args.face_blur,
            resume=args.resume,
            metadata_path=args.metadata,
            fps_limit=args.fps_limit,
            streams=args.streams,
            image_format=args.format,
            delegate=args.delegate,
            face_model=args.face_model,
            hand_model=args.hand_model,
            pose_model=args.pose_model,
            mp_log_level=args.mp_log_level,
            export_keypoints=args.export_keypoints,
            keypoints_output=args.keypoints_output,
            keypoints_format=args.keypoints_format,
        )
    except ValueError as exc:
        parser.error(str(exc))
