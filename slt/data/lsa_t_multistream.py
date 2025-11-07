"""Dataset multi-stream para LSA-T."""
from __future__ import annotations

import csv
import importlib
import math
import os
import random
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from slt.utils.metadata import (
    SplitSegment,
    parse_split_column,
    sanitize_time_value,
)

_POSE_SENTINEL = -1.0
_POSE_SIGNING_WIDTH = 6.0
_POSE_SIGNING_HEIGHT = 7.0

_CSV_SNIFFER_SAMPLE_SIZE = 4096
_CSV_SNIFFER_DELIMITERS = [";", ",", "\t", "|"]

_MAIN_CSV_ALIASES: Dict[str, Sequence[str]] = {
    "video_id": ("video_id", "id", "video"),
    "texto": ("texto", "text", "transcript", "sentence"),
}
_INDEX_CSV_ALIASES: Dict[str, Sequence[str]] = {
    "video_id": ("video_id", "id"),
}


def _load_csv_with_auto_delimiter(
    pd_module,
    path: str,
    *,
    fallback: str = ";",
    **kwargs: Any,
):
    """Lee un CSV detectando delimitadores y priorizando cabeceras con más columnas.

    La heurística complementa a ``csv.Sniffer`` cuando los archivos contienen listas
    separadas por comas dentro de campos (por ejemplo, ``split=train,val``). En esos
    casos el detector puede favorecer ``,``, por lo que se revisa la primera línea no
    vacía y se escoge el delimitador que produzca más columnas, desempatando con el
    valor ``fallback`` (``;`` por defecto).
    """

    with open(path, "r", encoding="utf-8-sig") as fh:
        sample = fh.read(_CSV_SNIFFER_SAMPLE_SIZE)
        fh.seek(0)
        delimiter = fallback
        header_line = ""
        column_counts: Dict[str, int] = {}
        if sample.strip():
            for raw_line in sample.splitlines():
                if raw_line.strip():
                    header_line = raw_line
                    break

            if header_line:
                for candidate in _CSV_SNIFFER_DELIMITERS:
                    reader = csv.reader([header_line], delimiter=candidate)
                    try:
                        row = next(reader)
                    except StopIteration:
                        count = 0
                    else:
                        count = len(row)
                    column_counts[candidate] = count
                if fallback not in column_counts:
                    reader = csv.reader([header_line], delimiter=fallback)
                    try:
                        row = next(reader)
                    except StopIteration:
                        column_counts[fallback] = 0
                    else:
                        column_counts[fallback] = len(row)

            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=_CSV_SNIFFER_DELIMITERS)
                delimiter = dialect.delimiter
            except csv.Error:
                pass

        if header_line and column_counts:
            sniffed_count = column_counts.get(delimiter, 0)
            fallback_count = column_counts.get(fallback, 0)
            if sniffed_count <= 1 or sniffed_count < fallback_count:
                max_columns = max(column_counts.values()) if column_counts else 0
                best_candidates = [
                    candidate
                    for candidate, count in column_counts.items()
                    if count == max_columns
                ]
                if fallback in best_candidates:
                    delimiter = fallback
                else:
                    for candidate in _CSV_SNIFFER_DELIMITERS:
                        if candidate in best_candidates:
                            delimiter = candidate
                            break
        return pd_module.read_csv(fh, sep=delimiter, **kwargs)


def _apply_column_aliases(
    df,
    aliases: Dict[str, Sequence[str]],
    *,
    context: str,
) -> None:
    """Normaliza las columnas de ``df`` según ``aliases`` y valida duplicados."""

    rename_map: Dict[str, str] = {}
    for canonical, candidates in aliases.items():
        present = [column for column in df.columns if column in candidates]
        if not present:
            accepted = ", ".join(sorted(candidates))
            raise ValueError(
                f"El {context} debe incluir una de las columnas "
                f"equivalentes a '{canonical}': {accepted}."
            )

        chosen = present[0]
        if canonical not in df.columns and chosen != canonical:
            rename_map[chosen] = canonical

        for alias in present[1:]:
            if alias == canonical:
                continue
            warnings.warn(
                "Se ignorará la columna '%s' al normalizar el %s; utiliza '%s' si "
                "necesitas acceder a la información adicional." % (
                    alias,
                    context,
                    canonical,
                ),
                stacklevel=2,
            )

    if rename_map:
        df.rename(columns=rename_map, inplace=True)


@lru_cache(maxsize=None)
def _lazy_import(name: str):
    """Importación perezosa con memoización."""
    return importlib.import_module(name)


def _get_pandas():  # pragma: no cover - función sencilla
    return _lazy_import("pandas")


def _get_numpy():  # pragma: no cover - función sencilla
    return _lazy_import("numpy")


def _get_pil_image():  # pragma: no cover - función sencilla
    return _lazy_import("PIL.Image")


@dataclass
class SampleItem:
    """Estructura con los tensores normalizados por clip."""

    face: torch.Tensor
    hand_l: torch.Tensor
    hand_r: torch.Tensor
    pose: torch.Tensor
    pose_conf_mask: torch.Tensor
    pad_mask: torch.Tensor
    length: torch.Tensor
    miss_mask_hl: torch.Tensor
    miss_mask_hr: torch.Tensor
    keypoints: torch.Tensor
    keypoints_mask: torch.Tensor
    keypoints_frame_mask: torch.Tensor
    keypoints_body: torch.Tensor
    keypoints_body_mask: torch.Tensor
    keypoints_body_frame_mask: torch.Tensor
    keypoints_hand_l: torch.Tensor
    keypoints_hand_l_mask: torch.Tensor
    keypoints_hand_l_frame_mask: torch.Tensor
    keypoints_hand_r: torch.Tensor
    keypoints_hand_r_mask: torch.Tensor
    keypoints_hand_r_frame_mask: torch.Tensor
    keypoints_face: torch.Tensor
    keypoints_face_mask: torch.Tensor
    keypoints_face_frame_mask: torch.Tensor
    keypoints_lengths: torch.Tensor
    ctc_labels: Optional[torch.Tensor]
    ctc_mask: Optional[torch.Tensor]
    gloss_sequence: Optional[List[str]]
    quality: Dict[str, Any]
    text: str
    gloss_text: Optional[str]
    video_id: str


_MSKA_FACE_SUBSET = [1, 133, 362, 13]


def _resolve_mediapipe_layout(
    num_landmarks: int,
    *,
    face_subset: Optional[Sequence[int]] = None,
) -> Dict[str, List[int]]:
    """Devuelve el layout de índices esperado para los keypoints de MediaPipe."""

    if num_landmarks >= 543:
        face_offset = 33
        hand_l_offset = face_offset + 468
        hand_r_offset = hand_l_offset + 21
        face_indices = [face_offset + idx for idx in range(468)]
        if face_subset is not None:
            try:
                subset = [int(value) for value in face_subset]
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "face_subset debe ser una secuencia de enteros"
                ) from exc
            filtered: List[int] = []
            for rel_idx in subset:
                if rel_idx < 0 or rel_idx >= len(face_indices):
                    warnings.warn(
                        "Índice facial fuera de rango en face_subset (%d); se ignorará."
                        % rel_idx,
                        stacklevel=2,
                    )
                    continue
                abs_idx = face_offset + rel_idx
                if abs_idx not in filtered:
                    filtered.append(abs_idx)
            if filtered:
                face_indices = filtered
            elif not subset:
                face_indices = []
            else:
                warnings.warn(
                    "face_subset vacío o inválido; se usará el rango completo de 468 puntos.",
                    stacklevel=2,
                )
        layout = {
            "body": list(range(0, 33)),
            "hand_l": list(range(hand_l_offset, hand_l_offset + 21)),
            "hand_r": list(range(hand_r_offset, hand_r_offset + 21)),
            "face": face_indices,
        }
        return {key: list(value) for key, value in layout.items()}
    if num_landmarks == 79:
        layout = {
            "body": list(range(0, 33)),
            "hand_l": list(range(33, 54)),
            "hand_r": list(range(54, 75)),
            "face": list(range(75, 79)),
        }
        return {key: list(value) for key, value in layout.items()}
    raise ValueError(
        "No se puede inferir la estructura de keypoints: se esperaban 79 u "
        ">=543 landmarks.")


_MEDIAPIPE_CONNECTIONS_TEMPLATE: Dict[str, List[Tuple[int, int]]] = {
    "body": [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 7),
        (0, 4),
        (4, 5),
        (5, 6),
        (6, 8),
        (9, 10),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
        (15, 17),
        (17, 19),
        (15, 19),
        (16, 18),
        (18, 20),
        (16, 20),
        (11, 23),
        (12, 24),
        (23, 24),
        (23, 25),
        (24, 26),
        (25, 27),
        (26, 28),
        (27, 29),
        (28, 30),
        (29, 31),
        (30, 32),
        (27, 31),
        (28, 32),
    ],
    "hand": [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),
    ],
    "face": [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 3),
        (2, 3),
    ],
}


def _resolve_mediapipe_connections(
    layout: Dict[str, List[int]]
) -> Dict[str, List[Tuple[int, int]]]:
    """Mapea conexiones estándar de MediaPipe según el layout recibido."""

    connections: Dict[str, List[Tuple[int, int]]] = {}
    for segment, indices in layout.items():
        template_key = "hand" if segment in {"hand_l", "hand_r"} else segment
        template = _MEDIAPIPE_CONNECTIONS_TEMPLATE.get(template_key)
        if not template:
            continue
        mapped: List[Tuple[int, int]] = []
        for start_rel, end_rel in template:
            if start_rel >= len(indices) or end_rel >= len(indices):
                continue
            mapped.append((indices[start_rel], indices[end_rel]))
        if mapped:
            connections[segment] = mapped
    return connections


def _face_swap_pairs(face_count: int) -> List[Tuple[int, int]]:
    """Pares de puntos faciales a intercambiar al reflejar."""

    if face_count >= len(_MSKA_FACE_SUBSET):
        # orden canónico: [nariz, ojo izq, ojo der, boca]
        return [(1, 2)]
    if face_count >= 2:
        return [(1, face_count - 1)]
    return []


class LsaTMultiStream(Dataset):
    """Dataset multi-stream para clips del corpus LSA-T.

    Espera la siguiente estructura de carpetas:

    - ``face/<video_id>_fXXXXXX.jpg`` o ``face/<video_id>.npz`` con ``frames``
    - ``hand_l/<video_id>_fXXXXXX.jpg`` o ``hand_l/<video_id>.npz``
    - ``hand_r/<video_id>_fXXXXXX.jpg`` o ``hand_r/<video_id>.npz``
    - ``pose/<video_id>.npz`` con clave ``pose``

    Además necesita un CSV con columnas ``video_id``/``id`` y ``texto``/``text``
    (separadas por ``;`` por defecto) y un CSV adicional con la lista de IDs
    pertenecientes al split a utilizar.
    """

    def __init__(
        self,
        face_dir: str,
        hand_l_dir: str,
        hand_r_dir: str,
        pose_dir: str,
        csv_path: str,
        index_csv: str,
        keypoints_dir: Optional[str] = None,
        gloss_csv: Optional[str] = None,
        T: int = 128,
        img_size: int = 224,
        lkp_count: int = 13,
        min_conf: float = 0.25,
        flip_prob: float = 0.2,
        enable_flip: bool = True,
        quality_checks: bool = True,
        quality_strict: bool = False,
        fps_tolerance: float = 1.0,
        keypoint_normalize_center: bool = True,
        keypoint_scale_range: Optional[Sequence[float]] = None,
        keypoint_translate_range: Optional[Sequence[float]] = None,
        keypoint_rotate_range: Optional[Sequence[float]] = None,
        keypoint_resample_range: Optional[Sequence[float]] = None,
        face_landmark_subset: Optional[Sequence[int]] = None,
    ) -> None:
        pd = _get_pandas()
        np = _get_numpy()

        self.face_dir = face_dir
        self.hand_l_dir = hand_l_dir
        self.hand_r_dir = hand_r_dir
        self.pose_dir = pose_dir
        self.keypoints_dir = keypoints_dir
        self.img_size = img_size
        self.T = T
        self.lkp_count = lkp_count
        self.min_conf = min_conf
        self.flip_prob = flip_prob
        self.enable_flip = enable_flip
        self.quality_checks = quality_checks
        self.quality_strict = quality_strict
        self.fps_tolerance = fps_tolerance
        self.keypoint_normalize_center = bool(keypoint_normalize_center)
        self.keypoint_scale_range = self._normalise_pair_range(
            keypoint_scale_range,
            field_name="keypoint_scale_range",
            positive=True,
            symmetric_on_scalar=False,
        )
        self.keypoint_translate_range = self._normalise_translation_range(
            keypoint_translate_range,
            field_name="keypoint_translate_range",
        )
        self.keypoint_rotate_range = self._normalise_pair_range(
            keypoint_rotate_range,
            field_name="keypoint_rotate_range",
            positive=False,
            symmetric_on_scalar=True,
        )
        self.keypoint_resample_range = self._normalise_pair_range(
            keypoint_resample_range,
            field_name="keypoint_resample_range",
            positive=True,
            symmetric_on_scalar=False,
        )
        self._np = np
        self._roi_npz_cache: Dict[str, Any] = {}
        self._keypoint_layout: Optional[Dict[str, List[int]]] = None
        self._keypoint_total: int = 0
        self._face_landmark_subset = self._normalise_face_subset(
            face_landmark_subset
        )
        self._gloss_sequences: Dict[str, List[str]] = {}
        self._gloss_texts: Dict[str, str] = {}
        self._ctc_labels: Dict[str, torch.Tensor] = {}
        self._ctc_masks: Dict[str, torch.Tensor] = {}
        self._pose_norm_flags: Dict[str, bool] = {}
        self._legacy_pose_warning_emitted = False
        self._keypoint_source_total: int = 0
        self._split_segments_cache: Dict[str, List[SplitSegment]] = {}
        self._has_split_column = False

        df = _load_csv_with_auto_delimiter(pd, csv_path)
        df.columns = [c.strip().lower() for c in df.columns]
        _apply_column_aliases(
            df,
            _MAIN_CSV_ALIASES,
            context="CSV principal",
        )

        idx = _load_csv_with_auto_delimiter(pd, index_csv)
        idx.columns = [c.strip().lower() for c in idx.columns]
        _apply_column_aliases(
            idx,
            _INDEX_CSV_ALIASES,
            context="CSV de índices",
        )

        def _normalise_text(value: Any) -> str:
            if value is None:
                return ""
            text = str(value).strip()
            if not text or text.lower() == "nan":
                return ""
            return text

        merged = df.merge(idx[["video_id"]], on="video_id", how="inner").copy()
        merged["_texto_norm"] = merged["texto"].map(_normalise_text)

        has_duration = "duration" in merged.columns
        if has_duration:
            merged["_duration_norm"] = merged["duration"].map(sanitize_time_value)
        else:
            merged["_duration_norm"] = pd.Series([None] * len(merged), index=merged.index)
            warnings.warn(
                "El CSV principal no contiene la columna 'duration'; se omitirá el "
                "filtrado por duración.",
                stacklevel=2,
            )

        drop_mask = merged["_texto_norm"] == ""
        if has_duration:
            drop_mask |= merged["_duration_norm"].isna()
        dropped_ids = merged.loc[drop_mask, "video_id"].astype(str).tolist()
        if dropped_ids:
            sample = ", ".join(dropped_ids[:5])
            suffix = f" (ejemplos: {sample})" if sample else ""
            warnings.warn(
                "Se descartaron %d video(s) sin texto o duración tras la normalización%s."
                % (len(dropped_ids), suffix),
                stacklevel=2,
            )

        self.df = merged.loc[~drop_mask].copy()
        self.df["texto"] = self.df["_texto_norm"]
        self.df.drop(columns=["_texto_norm", "_duration_norm"], inplace=True)
        self.df["video_id"] = self.df["video_id"].astype(str)
        self.df.reset_index(drop=True, inplace=True)
        self.ids = self.df["video_id"].tolist()
        self.texts = dict(zip(self.df["video_id"], self.df["texto"]))
        self._has_split_column = "split" in self.df.columns

        if gloss_csv:
            gloss_df = pd.read_csv(gloss_csv, sep=";")
            gloss_df.columns = [c.strip().lower() for c in gloss_df.columns]
            if "video_id" not in gloss_df.columns:
                raise ValueError(
                    "El CSV de glosas debe contener la columna 'video_id'.")
            if "gloss" in gloss_df.columns:
                self._gloss_texts = {
                    str(row["video_id"]): str(row["gloss"])
                    for _, row in gloss_df.iterrows()
                }
                self._gloss_sequences = {
                    vid: [token for token in text.split() if token]
                    for vid, text in self._gloss_texts.items()
                }
            if "ctc_labels" in gloss_df.columns:
                for _, row in gloss_df.iterrows():
                    vid = str(row["video_id"])
                    labels_raw = str(row["ctc_labels"]).strip()
                    if not labels_raw:
                        continue
                    try:
                        labels = torch.tensor(
                            [int(tok) for tok in labels_raw.split()],
                            dtype=torch.long,
                        )
                    except ValueError as exc:
                        raise ValueError(
                            f"ctc_labels inválidos para {vid}: {labels_raw}") from exc
                    mask = torch.ones(labels.shape[0], dtype=torch.bool)
                    self._ctc_labels[vid] = labels
                    self._ctc_masks[vid] = mask

        def _coerce(value: Any) -> Optional[float]:
            if value is None:
                return None
            return sanitize_time_value(value)

        self.meta = {}
        for vid in self.ids:
            vid_meta: Dict[str, Any] = {}
            rows = self.df.loc[self.df["video_id"] == vid]
            if rows.empty:
                continue
            row = rows.iloc[0]
            if "fps" in row:
                vid_meta["fps"] = _coerce(row["fps"])
            if "duration" in row:
                vid_meta["duration"] = _coerce(row["duration"])
            if "frame_count" in row:
                vid_meta["frame_count"] = _coerce(row["frame_count"])
            self.meta[vid] = vid_meta

    def __len__(self) -> int:  # pragma: no cover - simple
        return len(self.ids)

    def get_split_segments(self, video_id: str) -> List[SplitSegment]:
        """Return cached split segments parsed from ``meta.csv`` for ``video_id``."""

        vid = str(video_id)
        if vid in self._split_segments_cache:
            return self._split_segments_cache[vid]

        if not self._has_split_column:
            segments: List[SplitSegment] = []
        else:
            rows = self.df.loc[self.df["video_id"] == vid]
            if rows.empty:
                segments = []
            else:
                raw_value = rows.iloc[0].get("split")
                segments = parse_split_column(raw_value)

        self._split_segments_cache[vid] = segments
        return segments

    # ------------------------------------------------------------------
    # Utilidades internas
    # ------------------------------------------------------------------
    def _read_image(self, path: str) -> torch.Tensor:
        """Lee y normaliza una imagen RGB en rango ``[0, 1]``."""

        Image = _get_pil_image()
        np = self._np

        if "::npz::" in path:
            base, _, index_str = path.partition("::npz::")
            cache_key = base
            frames = self._roi_npz_cache.get(cache_key)
            if frames is None:
                with np.load(base) as data:
                    frames = data.get("frames")
                    if frames is None:
                        raise ValueError(f"{base} no contiene la clave 'frames'")
                    frames = np.asarray(frames)
                if frames.ndim != 4:
                    raise ValueError(
                        f"{base} no contiene un tensor (N, H, W, C) válido"
                    )
                self._roi_npz_cache[cache_key] = frames
            idx = int(index_str)
            if idx < 0 or idx >= frames.shape[0]:
                raise IndexError(f"Índice {idx} fuera de rango para {base}")
            frame = frames[idx]
            arr = frame.astype("float32") / 255.0
        else:
            with Image.open(path) as img:
                img = img.convert("RGB").resize((self.img_size, self.img_size))
                arr = np.asarray(img, dtype="float32") / 255.0
        arr = arr.transpose(2, 0, 1)
        return torch.from_numpy(arr)

    def _list_frames(self, base_dir: str, vid: str) -> List[str]:
        if not os.path.isdir(base_dir):
            return []
        entries = sorted(os.listdir(base_dir))
        prefix = f"{vid}_f"
        files = [
            os.path.join(base_dir, name)
            for name in entries
            if name.startswith(prefix)
            and name.lower().endswith((".jpg", ".png"))
        ]
        if files:
            return files

        npz_path = os.path.join(base_dir, f"{vid}.npz")
        if os.path.isfile(npz_path):
            cache_key = npz_path
            frames = self._roi_npz_cache.get(cache_key)
            if frames is None:
                try:
                    with self._np.load(npz_path) as data:
                        frames = data.get("frames")
                        if frames is None:
                            raise KeyError("frames")
                        frames = self._np.asarray(frames)
                except Exception as exc:  # pragma: no cover - lectura inesperada
                    warnings.warn(
                        f"No se pudo leer {npz_path}: {exc}",
                        stacklevel=2,
                    )
                    self._roi_npz_cache.pop(cache_key, None)
                    return []
                if frames.ndim != 4:
                    warnings.warn(
                        f"{npz_path} no contiene un tensor (N, H, W, C) válido.",
                        stacklevel=2,
                    )
                    self._roi_npz_cache.pop(cache_key, None)
                    return []
                self._roi_npz_cache[cache_key] = frames
            count = int(frames.shape[0]) if hasattr(frames, "shape") else 0
            return [f"{npz_path}::npz::{idx}" for idx in range(max(count, 0))]

        return []

    def _sample_indices(self, T0: int) -> List[int]:
        """Devuelve índices equiespaciados para muestrear ``T0`` frames."""

        if self.T <= 0:
            return []
        if T0 <= 0:
            return [0] * self.T

        np = self._np
        last_index = max(T0 - 1, 0)

        if T0 == 1:
            return [0] * self.T

        # ``linspace`` garantiza alineación consistente sin depender de ``random``
        # para mantener reproducibilidad.
        positions = np.linspace(0.0, float(last_index), num=self.T)
        idxs = np.rint(positions).astype("int64").tolist()
        return [max(0, min(last_index, int(i))) for i in idxs]

    def _normalise_pair_range(
        self,
        value: Optional[Sequence[float]],
        *,
        field_name: str,
        positive: bool,
        symmetric_on_scalar: bool,
    ) -> Optional[Tuple[float, float]]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            scalar = float(value)
            if symmetric_on_scalar:
                low, high = -scalar, scalar
            else:
                low = high = scalar
        else:
            try:
                items = [float(v) for v in value]
            except TypeError as exc:
                raise TypeError(
                    f"{field_name} debe ser numérico o una secuencia de floats"
                ) from exc
            if len(items) != 2:
                raise ValueError(
                    f"{field_name} espera exactamente 2 valores; se recibieron {len(items)}"
                )
            low, high = items
        if low > high:
            raise ValueError(
                f"{field_name}: el mínimo ({low}) no puede ser mayor que el máximo ({high})."
            )
        if positive and (low <= 0 or high <= 0):
            raise ValueError(
                f"{field_name}: los valores deben ser estrictamente positivos."
            )
        return float(low), float(high)

    def _normalise_translation_range(
        self,
        value: Optional[Sequence[float]],
        *,
        field_name: str,
    ) -> Optional[Tuple[float, float, float, float]]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            delta = float(value)
            return (-delta, delta, -delta, delta)
        try:
            items = [float(v) for v in value]
        except TypeError as exc:
            raise TypeError(
                f"{field_name} debe ser numérico o una secuencia de floats"
            ) from exc
        if len(items) == 2:
            low, high = items
            if low > high:
                raise ValueError(
                    f"{field_name}: el mínimo ({low}) no puede ser mayor que el máximo ({high})."
                )
            return float(low), float(high), float(low), float(high)
        if len(items) == 4:
            min_x, max_x, min_y, max_y = items
            if min_x > max_x or min_y > max_y:
                raise ValueError(
                    f"{field_name}: los límites deben cumplir min <= max por eje."
                )
            return float(min_x), float(max_x), float(min_y), float(max_y)
        raise ValueError(
            f"{field_name} espera 1, 2 o 4 valores; se recibieron {len(items)}"
        )

    def _normalise_face_subset(
        self,
        value: Optional[Sequence[int]],
    ) -> Optional[List[int]]:
        if value is None:
            return None
        try:
            subset = [int(v) for v in value]
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "face_landmark_subset debe ser una secuencia de enteros"
            ) from exc
        cleaned: List[int] = []
        for idx in subset:
            if idx < 0:
                raise ValueError(
                    "face_landmark_subset no admite índices negativos."
                )
            if idx not in cleaned:
                cleaned.append(idx)
        return cleaned

    def _ensure_keypoint_layout(self, num_landmarks: int) -> None:
        if num_landmarks <= 0:
            return
        if self._keypoint_layout is None:
            layout = _resolve_mediapipe_layout(
                num_landmarks,
                face_subset=self._face_landmark_subset,
            )
            self._keypoint_layout = layout
            self._keypoint_total = sum(len(v) for v in layout.values())
            self._keypoint_source_total = num_landmarks
            return
        if num_landmarks in (self._keypoint_source_total, self._keypoint_total):
            return
        raise ValueError(
            "Formato de keypoints inconsistente entre videos: se detectaron "
            f"{self._keypoint_source_total} y {num_landmarks} landmarks.")

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------
    def __getitem__(self, index: int) -> SampleItem:
        vid = self.ids[index]
        text = str(self.texts.get(vid, "")).strip()

        face_frames = self._list_frames(self.face_dir, vid)
        hl_frames = self._list_frames(self.hand_l_dir, vid)
        hr_frames = self._list_frames(self.hand_r_dir, vid)

        stream_lengths = {
            "face": len(face_frames),
            "hand_l": len(hl_frames),
            "hand_r": len(hr_frames),
        }

        pose_path = os.path.join(self.pose_dir, f"{vid}.npz")
        pose = self._load_pose(pose_path)
        pose_norm = self._pose_norm_flags.get(pose_path)
        pose_length = pose.shape[0] if hasattr(pose, "shape") else 0
        stream_lengths["pose"] = pose_length

        kp_selected = self._np.zeros((0, 0, 3), dtype="float32")
        if self.keypoints_dir:
            base = os.path.join(self.keypoints_dir, vid)
            kp_path = None
            for ext in (".npz", ".npy"):
                candidate = base + ext
                if os.path.exists(candidate):
                    kp_path = candidate
                    break
            kp_raw = self._read_keypoints_array(kp_path)
            kp_selected = self._select_keypoints(kp_raw)
        kp_processed = self._prepare_keypoints(kp_selected)
        keypoints_length = kp_processed.shape[0] if kp_processed.size > 0 else 0
        stream_lengths["keypoints"] = keypoints_length

        T0 = max(stream_lengths.values()) if stream_lengths else 0
        idxs = self._sample_indices(T0)

        def safe_get(frames: List[str], j: int) -> Optional[str]:
            if not frames:
                return None
            return frames[min(j, len(frames) - 1)]

        face_list: List[torch.Tensor] = []
        hl_list: List[torch.Tensor] = []
        hr_list: List[torch.Tensor] = []
        miss_hl: List[int] = []
        miss_hr: List[int] = []

        zero_img = torch.zeros(3, self.img_size, self.img_size, dtype=torch.float32)

        for j in idxs:
            fp = safe_get(face_frames, j)
            lp = safe_get(hl_frames, j)
            rp = safe_get(hr_frames, j)

            face_list.append(self._read_image(fp) if fp else zero_img)

            if lp:
                hl_list.append(self._read_image(lp))
                miss_hl.append(0)
            else:
                hl_list.append(zero_img)
                miss_hl.append(1)

            if rp:
                hr_list.append(self._read_image(rp))
                miss_hr.append(0)
            else:
                hr_list.append(zero_img)
                miss_hr.append(1)

        face = torch.stack(face_list, dim=0)
        hand_l = torch.stack(hl_list, dim=0)
        hand_r = torch.stack(hr_list, dim=0)

        pose_t, pose_mask = self._sample_pose(pose, normalized=pose_norm)
        keypoints_data = self._sample_keypoints(kp_processed, idxs)
        effective_length = self._effective_length(stream_lengths)

        pad_mask = torch.zeros(self.T, dtype=torch.bool)
        if effective_length > 0:
            pad_mask[:effective_length] = True
        length = torch.tensor(effective_length, dtype=torch.long)
        miss_mask_hl = torch.tensor(miss_hl, dtype=torch.bool)
        miss_mask_hr = torch.tensor(miss_hr, dtype=torch.bool)

        keypoints = torch.from_numpy(keypoints_data["keypoints"]).to(torch.float32)
        keypoints_mask = torch.from_numpy(keypoints_data["mask"])
        keypoints_frame_mask = torch.from_numpy(keypoints_data["frame_mask"])

        views_tensors = {
            key: torch.from_numpy(value).to(torch.float32)
            for key, value in keypoints_data["views"].items()
        }
        view_masks = {
            key: torch.from_numpy(value)
            for key, value in keypoints_data["view_masks"].items()
        }
        view_frame_masks = {
            key: torch.from_numpy(value)
            for key, value in keypoints_data["view_frame_masks"].items()
        }

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

        face = (face - mean) / std
        hand_l = (hand_l - mean) / std
        hand_r = (hand_r - mean) / std

        if self.enable_flip and self.flip_prob > 0 and random.random() < self.flip_prob:
            face = torch.flip(face, dims=[3])
            new_hand_l = torch.flip(hand_r, dims=[3])
            new_hand_r = torch.flip(hand_l, dims=[3])
            hand_l, hand_r = new_hand_l, new_hand_r
            miss_mask_hl, miss_mask_hr = miss_mask_hr, miss_mask_hl
            pose_t = self._flip_pose_tensor(pose_t)
            pose_mask = self._flip_pose_mask(pose_mask)
            keypoints, keypoints_mask, keypoints_frame_mask = self._flip_keypoints(
                keypoints, keypoints_mask, keypoints_frame_mask
            )
            views_tensors, view_masks, view_frame_masks = self._flip_keypoint_views(
                views_tensors, view_masks, view_frame_masks
            )

        keypoints_lengths = torch.tensor(
            [
                int(keypoints_frame_mask.sum().item()),
                int(view_frame_masks.get("body", torch.zeros(self.T, dtype=torch.bool)).sum().item()),
                int(view_frame_masks.get("hand_l", torch.zeros(self.T, dtype=torch.bool)).sum().item()),
                int(view_frame_masks.get("hand_r", torch.zeros(self.T, dtype=torch.bool)).sum().item()),
                int(view_frame_masks.get("face", torch.zeros(self.T, dtype=torch.bool)).sum().item()),
            ],
            dtype=torch.long,
        )

        quality = self._build_quality_report(
            vid,
            face_frames=face_frames,
            hand_l_frames=hl_frames,
            hand_r_frames=hr_frames,
            pose_frames=pose,
            keypoint_frames=kp_processed,
            effective_length=effective_length,
        )

        gloss_text = self._gloss_texts.get(vid)
        gloss_seq = self._gloss_sequences.get(vid)
        ctc_labels = self._ctc_labels.get(vid)
        ctc_mask = self._ctc_masks.get(vid)

        zeros_body = torch.zeros(self.T, 0, 3)
        zeros_mask = torch.zeros(self.T, 0, dtype=torch.bool)
        zeros_frame = torch.zeros(self.T, dtype=torch.bool)

        return SampleItem(
            face=face,
            hand_l=hand_l,
            hand_r=hand_r,
            pose=pose_t,
            pose_conf_mask=pose_mask,
            pad_mask=pad_mask,
            length=length,
            miss_mask_hl=miss_mask_hl,
            miss_mask_hr=miss_mask_hr,
            keypoints=keypoints,
            keypoints_mask=keypoints_mask,
            keypoints_frame_mask=keypoints_frame_mask,
            keypoints_body=views_tensors.get("body", torch.zeros_like(zeros_body)),
            keypoints_body_mask=view_masks.get("body", torch.zeros_like(zeros_mask)),
            keypoints_body_frame_mask=view_frame_masks.get(
                "body", torch.zeros_like(zeros_frame)
            ),
            keypoints_hand_l=views_tensors.get("hand_l", torch.zeros_like(zeros_body)),
            keypoints_hand_l_mask=view_masks.get("hand_l", torch.zeros_like(zeros_mask)),
            keypoints_hand_l_frame_mask=view_frame_masks.get(
                "hand_l", torch.zeros_like(zeros_frame)
            ),
            keypoints_hand_r=views_tensors.get("hand_r", torch.zeros_like(zeros_body)),
            keypoints_hand_r_mask=view_masks.get("hand_r", torch.zeros_like(zeros_mask)),
            keypoints_hand_r_frame_mask=view_frame_masks.get(
                "hand_r", torch.zeros_like(zeros_frame)
            ),
            keypoints_face=views_tensors.get("face", torch.zeros_like(zeros_body)),
            keypoints_face_mask=view_masks.get("face", torch.zeros_like(zeros_mask)),
            keypoints_face_frame_mask=view_frame_masks.get(
                "face", torch.zeros_like(zeros_frame)
            ),
            keypoints_lengths=keypoints_lengths,
            ctc_labels=ctc_labels,
            ctc_mask=ctc_mask,
            gloss_sequence=gloss_seq,
            quality=quality,
            text=text,
            gloss_text=gloss_text,
            video_id=vid,
        )

    # ------------------------------------------------------------------
    # Pose helpers
    # ------------------------------------------------------------------
    def _load_pose(self, pose_path: str) -> Any:
        np = self._np
        if not os.path.exists(pose_path):
            self._pose_norm_flags[pose_path] = False
            return np.zeros((1, self.lkp_count * 3), dtype="float32")

        try:
            with np.load(pose_path) as pose_npz:
                normalized = False
                for key in ("pose_norm", "pose_normalization", "pose_norm_tag"):
                    if key in pose_npz.files:
                        raw = pose_npz[key]
                        try:
                            value = raw.tolist()  # type: ignore[assignment]
                        except AttributeError:
                            value = raw
                        if isinstance(value, (list, tuple)):
                            value = value[0]
                        if isinstance(value, bytes):
                            value = value.decode("utf-8", "ignore")
                        normalized = bool(str(value).strip())
                        break
                self._pose_norm_flags[pose_path] = normalized

                pose = pose_npz.get("pose")
                if pose is None:
                    return np.zeros((1, self.lkp_count * 3), dtype="float32")

                pose_arr = np.asarray(pose, dtype="float32")
                if pose_arr.ndim == 3:
                    frames, landmarks, dims = pose_arr.shape
                    if dims < 2:
                        return np.zeros((1, self.lkp_count * 3), dtype="float32")
                    if dims == 2:
                        conf = pose_npz.get("confidence") or pose_npz.get("pose_confidence")
                        if conf is None:
                            conf = np.ones((frames, landmarks), dtype="float32")
                        conf = np.asarray(conf, dtype="float32").reshape(frames, landmarks)
                        conf = np.clip(conf, 0.0, 1.0)
                        pose_arr = np.concatenate([pose_arr, conf[..., None]], axis=2)
                    elif dims > 3:
                        pose_arr = pose_arr[..., :3]
                elif pose_arr.ndim == 2:
                    frames = pose_arr.shape[0]
                    features = pose_arr.shape[1]
                    if features % 3 == 0:
                        landmarks = features // 3
                        pose_arr = pose_arr.reshape(frames, landmarks, 3)
                    elif features % 2 == 0:
                        landmarks = features // 2
                        coords = pose_arr.reshape(frames, landmarks, 2)
                        conf = pose_npz.get("confidence") or pose_npz.get("pose_confidence")
                        if conf is None:
                            conf = np.ones((frames, landmarks), dtype="float32")
                        conf = np.asarray(conf, dtype="float32").reshape(frames, landmarks)
                        conf = np.clip(conf, 0.0, 1.0)
                        pose_arr = np.concatenate([coords, conf[..., None]], axis=2)
                    else:
                        return np.zeros((1, self.lkp_count * 3), dtype="float32")
                else:
                    return np.zeros((1, self.lkp_count * 3), dtype="float32")

                frames = pose_arr.shape[0]
                landmarks = pose_arr.shape[1]
                out = np.zeros((frames, self.lkp_count, 3), dtype="float32")
                copy_landmarks = min(self.lkp_count, landmarks)
                out[:, :copy_landmarks, : pose_arr.shape[2]] = pose_arr[:, :copy_landmarks, : pose_arr.shape[2]]
                return out.reshape(frames, self.lkp_count * 3)
        except (OSError, ValueError):
            self._pose_norm_flags[pose_path] = False
            return np.zeros((1, self.lkp_count * 3), dtype="float32")

    def _sample_pose(self, pose: Any, normalized: Optional[bool] = None) -> tuple[torch.Tensor, torch.Tensor]:
        np = self._np
        pose_arr = np.asarray(pose, dtype="float32")
        T0p = pose_arr.shape[0] if pose_arr.size > 0 else 0

        if T0p <= 0:
            pose_s = np.zeros((self.T, self.lkp_count, 3), dtype="float32")
        else:
            idxs_p = self._sample_indices(T0p)
            pose_s = pose_arr[idxs_p]
            if pose_s.ndim == 2:
                expected = self.lkp_count * 3
                if pose_s.shape[1] < expected:
                    padded = np.zeros((self.T, expected), dtype="float32")
                    padded[:, : pose_s.shape[1]] = pose_s
                    pose_s = padded
                pose_s = pose_s.reshape(self.T, self.lkp_count, 3)
            elif pose_s.ndim == 3:
                if pose_s.shape[1] != self.lkp_count:
                    trimmed = np.zeros((self.T, self.lkp_count, pose_s.shape[2]), dtype="float32")
                    copy_landmarks = min(self.lkp_count, pose_s.shape[1])
                    trimmed[:, :copy_landmarks, : pose_s.shape[2]] = pose_s[:, :copy_landmarks]
                    pose_s = trimmed
                if pose_s.shape[2] < 3:
                    pad = np.zeros((self.T, self.lkp_count, 3 - pose_s.shape[2]), dtype="float32")
                    pose_s = np.concatenate([pose_s, pad], axis=2)
                elif pose_s.shape[2] > 3:
                    pose_s = pose_s[:, :, :3]
            else:
                pose_s = np.zeros((self.T, self.lkp_count, 3), dtype="float32")

        pose_s = self._prepare_pose_array(pose_s, normalized)
        pose_tensor = torch.from_numpy(pose_s.reshape(self.T, self.lkp_count * 3))
        mask_tensor = torch.from_numpy(self._pose_mask_from_array(pose_s))
        return pose_tensor, mask_tensor

    def _prepare_pose_array(
        self,
        pose_arr: Any,
        normalized: Optional[bool] = None,
    ) -> Any:
        np = self._np
        if pose_arr.ndim != 3 or pose_arr.shape[-1] < 3:
            return np.zeros((self.T, self.lkp_count, 3), dtype="float32")

        processed = pose_arr.astype("float32", copy=True)
        normalized_flag = bool(normalized)

        if normalized is None:
            normalized_flag = self._infer_pose_normalization(processed)

        if not normalized_flag:
            processed = self._normalize_legacy_pose(processed)
            if not self._legacy_pose_warning_emitted:
                warnings.warn(
                    (
                        "Se detectó un stream de pose sin normalizar. Aplicando "
                        "reescalado de compatibilidad en LsaTMultiStream."
                    ),
                    RuntimeWarning,
                )
                self._legacy_pose_warning_emitted = True

        coords = processed[:, :, :2]
        conf = processed[:, :, 2]
        sentinel_mask = (coords < 0).any(axis=2)
        valid_mask = (conf >= self.min_conf) & (~sentinel_mask)

        coords = coords * valid_mask[..., None].astype("float32")
        coords[sentinel_mask] = _POSE_SENTINEL
        processed[:, :, :2] = coords
        processed[sentinel_mask, 2] = 0.0
        return processed

    def _pose_mask_from_array(self, pose_arr: Any) -> Any:
        np = self._np
        if pose_arr.ndim != 3 or pose_arr.shape[-1] < 3:
            return np.zeros((self.T, self.lkp_count), dtype="bool")

        conf = pose_arr[:, :, 2]
        coords = pose_arr[:, :, :2]
        sentinel_mask = (coords < 0).any(axis=2)
        mask = (conf >= self.min_conf) & (~sentinel_mask)
        return mask.astype("bool")

    def _infer_pose_normalization(self, pose_arr: Any) -> bool:
        coords = pose_arr[:, :, :2]
        if coords.size == 0:
            return False
        # Datos nuevos se esperan en [0, 1] y con posibles sentinels negativos.
        if (coords <= _POSE_SENTINEL).any():
            return True
        min_val = coords.min()
        max_val = coords.max()
        if min_val < -0.05 or max_val > 1.05:
            return False
        return True

    def _normalize_legacy_pose(self, pose_arr: Any) -> Any:
        np = self._np
        normalized = pose_arr.astype("float32", copy=True)
        if normalized.ndim != 3 or normalized.shape[-1] < 2:
            return normalized

        frames, landmarks, _ = normalized.shape
        for frame_idx in range(frames):
            coords = normalized[frame_idx, :, :2]
            center, width, height = self._legacy_signing_space(coords)
            translated = coords - center[None, :]
            if width <= 0 or height <= 0:
                continue
            translated[:, 0] /= width
            translated[:, 1] /= height
            translated += 0.5
            np.clip(translated, 0.0, 1.0, out=translated)
            normalized[frame_idx, :, :2] = translated
        return normalized

    def _legacy_signing_space(self, coords: Any) -> tuple[Any, float, float]:
        np = self._np
        if coords.size == 0:
            return np.array([0.5, 0.5], dtype="float32"), 1.0, 1.0

        nose = coords[0]
        shoulders: List[np.ndarray] = []
        if coords.shape[0] > 11:
            shoulders.append(coords[11])
        if coords.shape[0] > 12:
            shoulders.append(coords[12])

        if shoulders:
            center = np.mean(np.stack(shoulders, axis=0), axis=0)
        else:
            center = nose

        head_unit = float(np.linalg.norm(nose - center))
        if not math.isfinite(head_unit) or head_unit <= 1e-6:
            if coords.shape[0] > 5:
                left_eye = coords[min(2, coords.shape[0] - 1)]
                right_eye = coords[min(5, coords.shape[0] - 1)]
                head_unit = float(np.linalg.norm(left_eye - right_eye))
        if not math.isfinite(head_unit) or head_unit <= 1e-6:
            head_unit = 1.0 / _POSE_SIGNING_HEIGHT

        width = max(head_unit * _POSE_SIGNING_WIDTH, 1e-6)
        height = max(head_unit * _POSE_SIGNING_HEIGHT, 1e-6)
        return center.astype("float32"), float(width), float(height)

    # ------------------------------------------------------------------
    # Keypoint helpers
    # ------------------------------------------------------------------
    def _read_keypoints_array(self, kp_path: Optional[str]) -> Any:
        np = self._np
        if not kp_path or not os.path.exists(kp_path):
            total = self._keypoint_total if self._keypoint_total > 0 else 0
            return np.zeros((0, total, 3), dtype="float32")

        try:
            if kp_path.endswith(".npz"):
                with np.load(kp_path) as kp_npz:
                    arr = None
                    for key in ("keypoints", "kp", "points", "landmarks"):
                        if key in kp_npz:
                            arr = kp_npz[key]
                            break
                    if arr is None and kp_npz.files:
                        arr = kp_npz[kp_npz.files[0]]
                    conf = None
                    for key in ("confidence", "conf", "scores"):
                        if key in kp_npz:
                            conf = kp_npz[key]
                            break
            else:
                arr = np.load(kp_path)
                conf = None
        except (OSError, ValueError):
            return np.zeros((0, 0, 3), dtype="float32")

        arr = np.asarray(arr, dtype="float32")
        if arr.ndim == 3:
            frames, landmarks, dims = arr.shape
            if dims < 2:
                return np.zeros((0, 0, 3), dtype="float32")
            if dims == 2:
                ones = np.ones((frames, landmarks, 1), dtype="float32")
                arr = np.concatenate([arr, ones], axis=2)
            elif dims > 3:
                arr = arr[:, :, :3]
        elif arr.ndim == 2:
            frames, feats = arr.shape
            if feats % 3 == 0:
                arr = arr.reshape(frames, feats // 3, 3)
            elif feats % 2 == 0:
                coords = arr.reshape(frames, feats // 2, 2)
                ones = np.ones((frames, feats // 2, 1), dtype="float32")
                arr = np.concatenate([coords, ones], axis=2)
            else:
                return np.zeros((0, 0, 3), dtype="float32")
        else:
            return np.zeros((0, 0, 3), dtype="float32")

        frames = arr.shape[0]
        landmarks = arr.shape[1]
        if conf is not None:
            conf_arr = np.asarray(conf, dtype="float32")
            try:
                conf_arr = conf_arr.reshape(frames, landmarks)
            except ValueError:
                conf_arr = np.ones((frames, landmarks), dtype="float32")
            arr[:, :, 2] = conf_arr
        self._ensure_keypoint_layout(landmarks)
        return arr

    def _select_keypoints(self, kp_arr: Any) -> Any:
        np = self._np
        arr = np.asarray(kp_arr, dtype="float32")
        if arr.size == 0:
            total = self._keypoint_total if self._keypoint_total > 0 else 0
            return np.zeros((0, total, 3), dtype="float32")
        if self._keypoint_layout is None:
            self._ensure_keypoint_layout(arr.shape[1])
        if not self._keypoint_layout:
            return np.zeros((arr.shape[0], 0, 3), dtype="float32")

        order: List[int] = []
        for key in ("body", "hand_l", "hand_r", "face"):
            order.extend(self._keypoint_layout.get(key, []))

        selected = np.zeros((arr.shape[0], len(order), 3), dtype="float32")
        for out_idx, src_idx in enumerate(order):
            if src_idx < arr.shape[1]:
                selected[:, out_idx, :] = arr[:, src_idx, :]
        return selected

    def _split_keypoint_views(
        self,
        kp_arr: Any,
    ) -> Dict[str, Any]:
        np = self._np
        if kp_arr.size == 0 or not self._keypoint_layout:
            result: Dict[str, Any] = {}
            for key in ("body", "hand_l", "hand_r", "face"):
                count = len(self._keypoint_layout[key]) if self._keypoint_layout else 0
                result[key] = np.zeros((kp_arr.shape[0], count, 3), dtype="float32")
            return result

        result = {}
        start = 0
        for key in ("body", "hand_l", "hand_r", "face"):
            count = len(self._keypoint_layout.get(key, []))
            result[key] = kp_arr[:, start : start + count, :]
            start += count
        return result

    def _temporal_resample(
        self,
        kp_arr: Any,
        ratio_range: Tuple[float, float],
    ) -> Any:
        np = self._np
        arr = np.asarray(kp_arr, dtype="float32")
        frames = arr.shape[0]
        if frames <= 1:
            return arr
        low, high = ratio_range
        if low <= 0 or high <= 0:
            return arr
        scale = random.uniform(low, high)
        new_frames = max(1, int(round(frames * scale)))
        if new_frames == frames:
            return arr
        positions = np.linspace(0.0, float(frames - 1), num=new_frames, dtype="float32")
        lower = np.floor(positions).astype("int64")
        upper = np.clip(lower + 1, 0, frames - 1)
        weights = (positions - lower).astype("float32").reshape(new_frames, 1, 1)
        base = arr[lower]
        next_frames = arr[upper]
        resampled = base * (1.0 - weights) + next_frames * weights
        return resampled.astype("float32")

    def _prepare_keypoints(self, kp_arr: Any) -> Any:
        np = self._np
        arr = np.asarray(kp_arr, dtype="float32")
        if arr.ndim != 3:
            total = self._keypoint_total if self._keypoint_total > 0 else 0
            return np.zeros((0, total, 3), dtype="float32")
        frames, landmarks, _ = arr.shape
        if frames == 0 or landmarks == 0:
            return arr.astype("float32")

        processed = arr.astype("float32", copy=True)
        coords = processed[:, :, :2].copy()

        if self.keypoint_normalize_center:
            coords -= 0.5
        if self.keypoint_scale_range is not None:
            scale = random.uniform(*self.keypoint_scale_range)
            coords *= scale
        if self.keypoint_rotate_range is not None:
            angle = random.uniform(*self.keypoint_rotate_range)
            if abs(angle) > 1e-6:
                radians = math.radians(angle)
                cos_val = math.cos(radians)
                sin_val = math.sin(radians)
                rot = np.array([[cos_val, -sin_val], [sin_val, cos_val]], dtype="float32")
                coords = coords @ rot.T
        if self.keypoint_translate_range is not None:
            min_x, max_x, min_y, max_y = self.keypoint_translate_range
            shift_x = random.uniform(min_x, max_x)
            shift_y = random.uniform(min_y, max_y)
            coords += np.array([shift_x, shift_y], dtype="float32")
        if self.keypoint_normalize_center:
            coords += 0.5

        processed[:, :, :2] = coords

        if self.keypoint_resample_range is not None:
            processed = self._temporal_resample(processed, self.keypoint_resample_range)

        return processed.astype("float32")

    def _sample_keypoints(
        self,
        kp_arr: Any,
        idxs: Sequence[int],
    ) -> Dict[str, Any]:
        np = self._np
        arr = np.asarray(kp_arr, dtype="float32")
        if arr.ndim != 3:
            arr = np.zeros((0, 0, 3), dtype="float32")
        frames = arr.shape[0]
        if frames <= 0:
            total = self._keypoint_total if self._keypoint_total > 0 else 0
            base = np.zeros((self.T, total, 3), dtype="float32")
            mask = np.zeros((self.T, total), dtype="bool")
            frame_mask = np.zeros(self.T, dtype="bool")
            views = self._split_keypoint_views(base)
            view_masks = {k: np.zeros(v.shape[:2], dtype="bool") for k, v in views.items()}
            view_frame_masks = {k: np.zeros(self.T, dtype="bool") for k in views}
            lengths = {k: 0 for k in views}
            return {
                "keypoints": base,
                "mask": mask,
                "frame_mask": frame_mask,
                "views": views,
                "view_masks": view_masks,
                "view_frame_masks": view_frame_masks,
                "view_lengths": lengths,
            }

        clamped = [min(max(0, i), frames - 1) for i in idxs]
        sampled = arr[clamped]
        conf = sampled[:, :, 2]
        mask = conf >= self.min_conf
        coords = sampled[:, :, :2]
        in_bounds = (
            (coords[:, :, 0] >= 0.0)
            & (coords[:, :, 0] <= 1.0)
            & (coords[:, :, 1] >= 0.0)
            & (coords[:, :, 1] <= 1.0)
        )
        mask = mask & in_bounds
        clipped = self._np.clip(coords, 0.0, 1.0)
        sampled[:, :, :2] = clipped * mask[..., None].astype("float32")
        frame_mask = mask.any(axis=1)

        views_np = self._split_keypoint_views(sampled)
        view_masks = {}
        view_frame_masks = {}
        view_lengths = {}
        start = 0
        for key, view_arr in views_np.items():
            count = view_arr.shape[1]
            if count == 0:
                view_mask = np.zeros((self.T, 0), dtype="bool")
                view_frame = np.zeros(self.T, dtype="bool")
            else:
                view_mask = mask[:, start : start + count]
                view_frame = view_mask.any(axis=1)
            view_masks[key] = view_mask
            view_frame_masks[key] = view_frame
            view_lengths[key] = int(view_frame.sum())
            start += count

        return {
            "keypoints": sampled,
            "mask": mask,
            "frame_mask": frame_mask,
            "views": views_np,
            "view_masks": view_masks,
            "view_frame_masks": view_frame_masks,
            "view_lengths": view_lengths,
        }

    def _flip_keypoints(
        self,
        keypoints: torch.Tensor,
        mask: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if keypoints.numel() == 0:
            return keypoints, mask, frame_mask

        flipped = keypoints.clone()
        flipped[:, :, 0] = 1.0 - flipped[:, :, 0]
        mask_flipped = mask.clone()

        body_count = len(self._keypoint_layout.get("body", [])) if self._keypoint_layout else 0
        hand_l_count = len(self._keypoint_layout.get("hand_l", [])) if self._keypoint_layout else 0
        hand_r_count = len(self._keypoint_layout.get("hand_r", [])) if self._keypoint_layout else 0
        face_count = len(self._keypoint_layout.get("face", [])) if self._keypoint_layout else 0

        if body_count:
            for left_idx, right_idx in self._pose_swap_pairs(body_count):
                if left_idx < body_count and right_idx < body_count:
                    li = left_idx
                    ri = right_idx
                    flipped[:, [li, ri], :] = flipped[:, [ri, li], :]
                    mask_flipped[:, [li, ri]] = mask_flipped[:, [ri, li]]

        # Manos: intercambiar segmentos completos
        if hand_l_count and hand_r_count:
            left_start = body_count
            right_start = body_count + hand_l_count
            left_slice = slice(left_start, left_start + hand_l_count)
            right_slice = slice(right_start, right_start + hand_r_count)
            flipped_left = flipped[:, left_slice].clone()
            flipped_right = flipped[:, right_slice].clone()
            mask_left = mask_flipped[:, left_slice].clone()
            mask_right = mask_flipped[:, right_slice].clone()
            flipped[:, left_slice] = flipped_right
            flipped[:, right_slice] = flipped_left
            mask_flipped[:, left_slice] = mask_right
            mask_flipped[:, right_slice] = mask_left

        if face_count:
            face_start = body_count + hand_l_count + hand_r_count
            for left_idx, right_idx in _face_swap_pairs(face_count):
                if left_idx < face_count and right_idx < face_count:
                    li = face_start + left_idx
                    ri = face_start + right_idx
                    flipped[:, [li, ri], :] = flipped[:, [ri, li], :]
                    mask_flipped[:, [li, ri]] = mask_flipped[:, [ri, li]]

        frame_mask_flipped = mask_flipped.any(dim=1)
        return flipped, mask_flipped, frame_mask_flipped

    def _flip_keypoint_views(
        self,
        views: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
        frame_masks: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        flipped_views: Dict[str, torch.Tensor] = {}
        flipped_masks: Dict[str, torch.Tensor] = {}
        flipped_frame_masks: Dict[str, torch.Tensor] = {}

        body = views.get("body")
        body_mask = masks.get("body")
        body_frame = frame_masks.get("body")
        if body is not None:
            body_flipped = body.clone()
            body_flipped[:, :, 0] = 1.0 - body_flipped[:, :, 0]
            mask_body = body_mask.clone() if body_mask is not None else torch.zeros(
                body_flipped.shape[:2], dtype=torch.bool
            )
            for left_idx, right_idx in self._pose_swap_pairs(body_flipped.shape[1]):
                if right_idx < body_flipped.shape[1] and left_idx < body_flipped.shape[1]:
                    body_flipped[:, [left_idx, right_idx]] = body_flipped[:, [right_idx, left_idx]]
                    mask_body[:, [left_idx, right_idx]] = mask_body[:, [right_idx, left_idx]]
            flipped_views["body"] = body_flipped
            flipped_masks["body"] = mask_body
            flipped_frame_masks["body"] = (
                body_frame.clone() if body_frame is not None else mask_body.any(dim=1)
            )

        left = views.get("hand_l")
        right = views.get("hand_r")
        left_mask = masks.get("hand_l")
        right_mask = masks.get("hand_r")
        left_frame = frame_masks.get("hand_l")
        right_frame = frame_masks.get("hand_r")
        if left is not None and right is not None:
            new_left = right.clone()
            new_right = left.clone()
            new_left[:, :, 0] = 1.0 - new_left[:, :, 0]
            new_right[:, :, 0] = 1.0 - new_right[:, :, 0]
            mask_left = right_mask.clone() if right_mask is not None else torch.zeros(
                right.shape[:2], dtype=torch.bool
            )
            mask_right = left_mask.clone() if left_mask is not None else torch.zeros(
                left.shape[:2], dtype=torch.bool
            )
            flipped_views["hand_l"] = new_left
            flipped_views["hand_r"] = new_right
            flipped_masks["hand_l"] = mask_left
            flipped_masks["hand_r"] = mask_right
            flipped_frame_masks["hand_l"] = (
                right_frame.clone() if right_frame is not None else mask_left.any(dim=1)
            )
            flipped_frame_masks["hand_r"] = (
                left_frame.clone() if left_frame is not None else mask_right.any(dim=1)
            )
        else:
            if left is not None:
                flipped_views["hand_l"] = left
                flipped_masks["hand_l"] = left_mask if left_mask is not None else torch.zeros(
                    left.shape[:2], dtype=torch.bool
                )
                flipped_frame_masks["hand_l"] = (
                    left_frame.clone() if left_frame is not None else flipped_masks["hand_l"].any(dim=1)
                )
            if right is not None:
                flipped_views["hand_r"] = right
                flipped_masks["hand_r"] = right_mask if right_mask is not None else torch.zeros(
                    right.shape[:2], dtype=torch.bool
                )
                flipped_frame_masks["hand_r"] = (
                    right_frame.clone() if right_frame is not None else flipped_masks["hand_r"].any(dim=1)
                )

        face = views.get("face")
        face_mask = masks.get("face")
        face_frame = frame_masks.get("face")
        if face is not None:
            face_flipped = face.clone()
            face_flipped[:, :, 0] = 1.0 - face_flipped[:, :, 0]
            mask_face = face_mask.clone() if face_mask is not None else torch.zeros(
                face.shape[:2], dtype=torch.bool
            )
            for left_idx, right_idx in _face_swap_pairs(face_flipped.shape[1]):
                if right_idx < face_flipped.shape[1] and left_idx < face_flipped.shape[1]:
                    face_flipped[:, [left_idx, right_idx]] = face_flipped[:, [right_idx, left_idx]]
                    mask_face[:, [left_idx, right_idx]] = mask_face[:, [right_idx, left_idx]]
            flipped_views["face"] = face_flipped
            flipped_masks["face"] = mask_face
            flipped_frame_masks["face"] = (
                face_frame.clone() if face_frame is not None else mask_face.any(dim=1)
            )

        return flipped_views, flipped_masks, flipped_frame_masks

    def _flip_pose_tensor(self, pose: torch.Tensor) -> torch.Tensor:
        """Devuelve ``pose`` reflejada horizontalmente, intercambiando lados."""

        if pose.numel() == 0:
            return pose

        flipped = pose.clone()
        T, pose_dim = flipped.shape
        reshaped = flipped.view(T, -1, 3)
        lkp_count = reshaped.shape[1]

        reshaped[:, :, 0] = 1.0 - reshaped[:, :, 0]

        for left_idx, right_idx in self._pose_swap_pairs(lkp_count):
            if left_idx < lkp_count and right_idx < lkp_count:
                reshaped[:, [left_idx, right_idx]] = reshaped[:, [right_idx, left_idx]]

        return reshaped.view(T, pose_dim)

    def _flip_pose_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Refleja la máscara de confianza de la pose."""

        if mask.numel() == 0:
            return mask

        flipped = mask.clone()
        T, lkp_count = flipped.shape

        for left_idx, right_idx in self._pose_swap_pairs(lkp_count):
            if left_idx < lkp_count and right_idx < lkp_count:
                flipped[:, [left_idx, right_idx]] = flipped[:, [right_idx, left_idx]]

        return flipped

    @staticmethod
    def _pose_swap_pairs(lkp_count: int) -> List[tuple[int, int]]:
        """Pares de landmarks que deben intercambiarse al reflejar."""

        del lkp_count  # solo para mantener la firma uniforme
        return [
            (1, 4),
            (2, 5),
            (3, 6),
            (7, 8),
            (9, 10),
            (11, 12),
            (13, 14),
            (15, 16),
        ]

    def _effective_length(self, lengths: Dict[str, int]) -> int:
        positives = [v for v in lengths.values() if v > 0]
        if not positives:
            return 0
        if any(v == 0 for v in lengths.values()):
            face_length = lengths.get("face", 0)
            if face_length > 0:
                return min(face_length, self.T)
            return min(max(positives), self.T)
        return min(min(positives), self.T)

    def _frame_indices(self, frames: List[str]) -> List[int]:
        indices: List[int] = []
        for path in frames:
            name = os.path.basename(path)
            stem, _ = os.path.splitext(name)
            if "_f" in stem:
                try:
                    idx = int(stem.split("_f", 1)[1])
                except ValueError:
                    continue
                indices.append(idx)
        return sorted(indices)

    def _detect_missing_indices(self, indices: List[int]) -> List[int]:
        if not indices:
            return []
        expected = range(indices[0], indices[-1] + 1)
        idx_set = set(indices)
        return [i for i in expected if i not in idx_set]

    def _build_quality_report(
        self,
        vid: str,
        *,
        face_frames: List[str],
        hand_l_frames: List[str],
        hand_r_frames: List[str],
        pose_frames: Any,
        keypoint_frames: Any,
        effective_length: int,
    ) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "video_id": vid,
            "effective_length": effective_length,
            "missing_frames": {},
            "fps": None,
            "issues": [],
        }

        streams = {
            "face": face_frames,
            "hand_l": hand_l_frames,
            "hand_r": hand_r_frames,
        }

        meta = self.meta.get(vid, {})
        expected_total = None
        if meta:
            expected_total = meta.get("frame_count")
        if expected_total is None:
            expected_total = max(len(face_frames), len(hand_l_frames), len(hand_r_frames), effective_length)
        expected_total = int(expected_total) if expected_total is not None else None

        for name, frames in streams.items():
            indices = self._frame_indices(frames)
            missing = self._detect_missing_indices(indices)
            if missing:
                report["missing_frames"][name] = {
                    "count": len(missing),
                    "indices": missing,
                    "available": len(indices),
                    "expected": len(indices) + len(missing),
                }
            if expected_total is not None and len(indices) < expected_total:
                deficit = expected_total - len(indices)
                entry = report["missing_frames"].setdefault(
                    name,
                    {
                        "count": 0,
                        "indices": [],
                        "available": len(indices),
                        "expected": expected_total,
                    },
                )
                entry["count"] += deficit
                entry["expected"] = expected_total

        pose_len = pose_frames.shape[0] if hasattr(pose_frames, "shape") else 0
        if pose_len <= 0:
            report["missing_frames"]["pose"] = {
                "count": 1,
                "indices": "all",
                "available": 0,
                "expected": effective_length,
            }

        kp_len = keypoint_frames.shape[0] if hasattr(keypoint_frames, "shape") else 0
        if kp_len <= 0:
            report["missing_frames"]["keypoints"] = {
                "count": 1,
                "indices": "all",
                "available": 0,
                "expected": effective_length,
            }

        expected_fps = meta.get("fps") if meta else None
        duration = meta.get("duration") if meta else None
        frame_count = meta.get("frame_count") if meta else None

        actual_frames = max([len(face_frames), len(hand_l_frames), len(hand_r_frames), pose_len, effective_length])
        actual_fps = None
        if duration and duration > 0:
            actual_fps = actual_frames / float(duration)

        if expected_fps is None and frame_count and duration:
            expected_fps = frame_count / float(duration)

        fps_info: Optional[Dict[str, Any]] = None
        if expected_fps is not None or actual_fps is not None:
            diff = None
            ok = True
            if expected_fps is not None and actual_fps is not None:
                diff = abs(actual_fps - expected_fps)
                ok = diff <= self.fps_tolerance
            fps_info = {
                "expected": expected_fps,
                "actual": actual_fps,
                "diff": diff,
                "ok": ok,
            }
        report["fps"] = fps_info

        if self.quality_checks:
            issues: List[str] = []
            if report["missing_frames"]:
                issues.append(
                    f"Frames faltantes detectados en {vid}: "
                    + ", ".join(f"{k} ({v['count']})" for k, v in report["missing_frames"].items())
                )
            if fps_info and fps_info["expected"] and fps_info["actual"] and not fps_info["ok"]:
                issues.append(
                    f"FPS fuera de tolerancia para {vid}: esperado {fps_info['expected']}, "
                    f"observado {fps_info['actual']:.2f}"
                )
            if issues:
                report["issues"].extend(issues)
                message = "; ".join(issues)
                if self.quality_strict:
                    raise ValueError(message)
                warnings.warn(message)

        return report


def collate_fn(batch: Iterable[SampleItem]) -> Dict[str, Any]:
    batch_list = list(batch)
    if not batch_list:
        raise ValueError("El batch no puede estar vacío.")

    def stack_attr(attr: str) -> torch.Tensor:
        return torch.stack([getattr(sample, attr) for sample in batch_list], dim=0)

    batch_size = len(batch_list)

    data: Dict[str, Any] = {
        "face": stack_attr("face"),
        "hand_l": stack_attr("hand_l"),
        "hand_r": stack_attr("hand_r"),
        "pose": stack_attr("pose"),
        "pose_conf_mask": stack_attr("pose_conf_mask"),
        "pad_mask": stack_attr("pad_mask"),
        "lengths": stack_attr("length"),
        "miss_mask_hl": stack_attr("miss_mask_hl"),
        "miss_mask_hr": stack_attr("miss_mask_hr"),
        "keypoints": stack_attr("keypoints"),
        "keypoints_mask": stack_attr("keypoints_mask"),
        "keypoints_frame_mask": stack_attr("keypoints_frame_mask"),
        "keypoints_body": stack_attr("keypoints_body"),
        "keypoints_body_mask": stack_attr("keypoints_body_mask"),
        "keypoints_body_frame_mask": stack_attr("keypoints_body_frame_mask"),
        "keypoints_hand_l": stack_attr("keypoints_hand_l"),
        "keypoints_hand_l_mask": stack_attr("keypoints_hand_l_mask"),
        "keypoints_hand_l_frame_mask": stack_attr("keypoints_hand_l_frame_mask"),
        "keypoints_hand_r": stack_attr("keypoints_hand_r"),
        "keypoints_hand_r_mask": stack_attr("keypoints_hand_r_mask"),
        "keypoints_hand_r_frame_mask": stack_attr("keypoints_hand_r_frame_mask"),
        "keypoints_face": stack_attr("keypoints_face"),
        "keypoints_face_mask": stack_attr("keypoints_face_mask"),
        "keypoints_face_frame_mask": stack_attr("keypoints_face_frame_mask"),
        "keypoints_lengths": stack_attr("keypoints_lengths"),
        "quality": [sample.quality for sample in batch_list],
        "texts": [sample.text for sample in batch_list],
        "gloss_texts": [sample.gloss_text for sample in batch_list],
        "gloss_sequences": [sample.gloss_sequence or [] for sample in batch_list],
        "video_ids": [sample.video_id for sample in batch_list],
    }

    ctc_lengths = [
        sample.ctc_labels.shape[0] if sample.ctc_labels is not None else 0
        for sample in batch_list
    ]
    max_ctc_len = max(ctc_lengths) if ctc_lengths else 0
    if max_ctc_len > 0:
        labels_pad = torch.full((batch_size, max_ctc_len), -100, dtype=torch.long)
        mask_pad = torch.zeros((batch_size, max_ctc_len), dtype=torch.bool)
        for idx, sample in enumerate(batch_list):
            if sample.ctc_labels is None:
                continue
            length = sample.ctc_labels.shape[0]
            labels_pad[idx, :length] = sample.ctc_labels
            if sample.ctc_mask is not None:
                mask_pad[idx, :length] = sample.ctc_mask
            else:
                mask_pad[idx, :length] = True
        data["ctc_labels"] = labels_pad
        data["ctc_mask"] = mask_pad
        data["ctc_lengths"] = torch.tensor(ctc_lengths, dtype=torch.long)
    else:
        data["ctc_labels"] = torch.zeros(batch_size, 0, dtype=torch.long)
        data["ctc_mask"] = torch.zeros(batch_size, 0, dtype=torch.bool)
        data["ctc_lengths"] = torch.zeros(batch_size, dtype=torch.long)

    return data
