#!/usr/bin/env python3
"""Visor interactivo para validar video, keypoints y subtítulos en sincronía."""

from __future__ import annotations

import argparse
import csv
import math
import time
from collections import Counter
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import (
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from slt.data.lsa_t_multistream import (
    _resolve_mediapipe_connections,
    _resolve_mediapipe_layout,
)
from slt.utils.metadata import SplitSegment, parse_split_column, sanitize_time_value


@dataclass
class SubtitleConfig:
    """Configuración básica para interpretar el CSV de subtítulos."""

    csv_path: Path
    delimiter: str = ";"
    id_column: str = "id"
    video_column: str = "video"
    text_column: str = "text"
    start_column: str = "start"
    end_column: str = "end"
    split_column: Optional[str] = "split"
    target_id: Optional[str] = None
    target_video: Optional[str] = None
    absolute_times: bool = False


@dataclass
class SubtitleEntry:
    """Segmento individual preparado para mostrar en pantalla."""

    text: str
    start: float
    end: float

    def contains(self, timestamp: float) -> bool:
        return self.start <= timestamp <= self.end


@dataclass
class ViewerConfig:
    """Parámetros del visor."""

    window_name: str = "SLT keypoint viewer"
    wait_time_ms: int = 1
    loop: bool = False
    display_scale: float = 1.0
    font_scale: float = 1.4
    font_thickness: int = 3
    subtitle_margin: int = 24
    subtitle_max_width: int = 900
    confidence_threshold: float = 0.2
    normalised_keypoints: bool = True
    video_offset: float = 0.0
    keypoints_offset: float = 0.0
    seek_to_start: bool = True
    draw_bones: bool = True
    face_point_stride: int = 1
    max_face_points: Optional[int] = None
    start_paused: bool = False
    initial_subtitles: bool = True
    initial_keypoints: bool = True


@dataclass
class KeypointData:
    """Estructura con los keypoints y metadatos auxiliares."""

    frames: np.ndarray
    layout: Dict[str, List[int]]
    connections: Dict[str, List[Tuple[int, int]]]
    fps: float


VIDEO_EXTENSIONS = (".mp4", ".mkv", ".mov", ".avi", ".webm")


@lru_cache(maxsize=8)
def _load_font(size: int) -> ImageFont.ImageFont:
    """Carga la fuente TrueType deseada o informa de los fallos encontrados."""

    module_dir = Path(__file__).resolve().parent
    packaged_font = module_dir / "fonts" / "DejaVuSans.ttf"

    candidates: List[Path] = []
    if packaged_font.exists():
        candidates.append(packaged_font)

    candidates.extend(
        [
            Path("DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/local/share/fonts/DejaVuSans.ttf"),
            module_dir / "DejaVuSans.ttf",
        ]
    )

    errors: List[str] = []
    for candidate in candidates:
        try:
            return ImageFont.truetype(str(candidate), size=size)
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")

    searched_paths = ", ".join(str(path) for path in candidates)
    error_hint = "\n".join(errors) if errors else "Font file not found."
    raise RuntimeError(
        "No se pudo cargar la fuente DejaVuSans.ttf. "
        f"Rutas intentadas: {searched_paths}.\nDetalles: {error_hint}"
    )


def _wrap_text(
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> List[str]:
    """Divide el subtítulo en líneas que quepan en ``max_width`` píxeles."""

    words = text.split()
    if not words:
        return [""]

    max_width = max(max_width, 1)
    dummy_image = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy_image)

    lines: List[str] = []
    current: List[str] = []

    for word in words:
        tentative = " ".join(current + [word])
        bbox = draw.textbbox((0, 0), tentative, font=font)
        width = bbox[2] - bbox[0]
        if width <= max_width or not current:
            current.append(word)
            continue
        lines.append(" ".join(current))
        current = [word]

    if current:
        lines.append(" ".join(current))

    return lines


def _wrap_text_cv2(
    text: str,
    font: int,
    scale: float,
    thickness: int,
    max_width: int,
) -> List[str]:
    """Divide ``text`` en líneas usando ``cv2.getTextSize``."""

    words = text.split()
    if not words:
        return [""]

    max_width = max(max_width, 1)
    lines: List[str] = []
    current: List[str] = []

    for word in words:
        tentative = " ".join(current + [word])
        size, _ = cv2.getTextSize(tentative, font, scale, thickness)
        width = size[0]
        if width <= max_width or not current:
            current.append(word)
            continue
        lines.append(" ".join(current))
        current = [word]

    if current:
        lines.append(" ".join(current))

    return lines


def _parse_face_subset_arg(value: str) -> List[int]:
    """Convierte una lista separada por comas en enteros no negativos."""

    text = value.strip()
    if not text:
        return []
    try:
        items = [int(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Los índices de --face-landmark-subset deben ser enteros"
        ) from exc
    for idx in items:
        if idx < 0:
            raise argparse.ArgumentTypeError(
                "--face-landmark-subset no admite valores negativos"
            )
    # Se preserva el orden y se eliminan duplicados manualmente.
    unique: List[int] = []
    for idx in items:
        if idx not in unique:
            unique.append(idx)
    return unique


def _draw_subtitles(
    frame: np.ndarray,
    subtitle: str,
    viewer_cfg: ViewerConfig,
) -> None:
    """Superpone el subtítulo activo sobre ``frame``."""

    font_size = max(24, int(round(48 * viewer_cfg.font_scale)))
    font = _load_font(font_size)

    available_width = max(
        1,
        min(
            viewer_cfg.subtitle_max_width,
            frame.shape[1] - 2 * viewer_cfg.subtitle_margin,
        ),
    )
    lines = _wrap_text(subtitle, font, available_width)

    if not lines:
        return

    dummy_image = Image.new("RGB", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_image)
    try:
        ascent, descent = font.getmetrics()
        base_height = ascent + descent
    except AttributeError:
        bbox = dummy_draw.textbbox((0, 0), "Ag", font=font)
        base_height = bbox[3] - bbox[1]
    line_spacing = max(4, int(base_height * 0.25))
    line_height = base_height + line_spacing

    y_base = frame.shape[0] - viewer_cfg.subtitle_margin
    text_height = len(lines) * line_height
    x_margin = viewer_cfg.subtitle_margin

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_frame).convert("RGBA")

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    rect_top = max(0, y_base - text_height - viewer_cfg.subtitle_margin // 2)
    rect_bottom = min(frame.shape[0], y_base + viewer_cfg.subtitle_margin // 2)
    rect_left = max(0, x_margin - 10)
    rect_right = min(frame.shape[1], frame.shape[1] - x_margin + 10)
    overlay_draw.rectangle(
        [(rect_left, rect_top), (rect_right, rect_bottom)],
        fill=(0, 0, 0, int(255 * 0.55)),
    )

    image = Image.alpha_composite(image, overlay)
    draw = ImageDraw.Draw(image)

    for idx, line in enumerate(lines):
        y = y_base - (len(lines) - idx) * line_height + line_spacing // 2
        y = max(0, y)
        draw.text((x_margin, y), line, font=font, fill=(255, 255, 255, 255))

    updated = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    frame[:, :, :] = updated


def _load_csv_entries(
    cfg: SubtitleConfig,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Lee el CSV y retorna filas normalizadas junto con el filtro activo."""

    if not cfg.csv_path.exists():
        raise FileNotFoundError(f"No se encontró el CSV: {cfg.csv_path}")

    # ``utf-8-sig`` asegura que los archivos con BOM se decodifiquen correctamente.
    with cfg.csv_path.open("r", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh, delimiter=cfg.delimiter)
        rows = list(reader)

    if not rows:
        raise ValueError(f"El CSV {cfg.csv_path} no contiene filas.")

    normalised: List[Dict[str, str]] = []
    filtered: List[Dict[str, str]] = []
    for row in rows:
        row_norm = {
            key: (value.strip() if isinstance(value, str) else value)
            for key, value in row.items()
        }
        normalised.append(row_norm)
        if cfg.target_id and row_norm.get(cfg.id_column) != cfg.target_id:
            continue
        if cfg.target_video and row_norm.get(cfg.video_column) != cfg.target_video:
            continue
        filtered.append(row_norm)

    if not filtered:
        target = cfg.target_id or cfg.target_video or "<sin filtro>"
        raise ValueError(
            f"No se hallaron filas que coincidan con {target!r} en {cfg.csv_path}."
        )

    return normalised, filtered


def _load_subtitles(cfg: SubtitleConfig) -> Tuple[List[SubtitleEntry], Optional[float]]:
    """Carga los subtítulos y retorna segmentos más el inicio sugerido."""

    _all_rows, filtered = _load_csv_entries(cfg)

    segments: List[SubtitleEntry] = []
    clip_start: Optional[float] = None

    for row in filtered:
        start_raw = row.get(cfg.start_column)
        end_raw = row.get(cfg.end_column)
        start = sanitize_time_value(start_raw) or 0.0
        end = sanitize_time_value(end_raw) or start
        if clip_start is None:
            clip_start = start
        split_raw = row.get(cfg.split_column) if cfg.split_column else None
        if split_raw:
            parsed = parse_split_column(split_raw)
            source_segments: Iterable[SplitSegment] = parsed
        else:
            source_segments = [SplitSegment(row.get(cfg.text_column, ""), start, end)]

        for segment in source_segments:
            rel_start = segment.start
            rel_end = segment.end
            if not cfg.absolute_times and clip_start is not None:
                rel_start -= clip_start
                rel_end -= clip_start
            rel_start = max(rel_start, 0.0)
            rel_end = max(rel_end, rel_start)
            segments.append(SubtitleEntry(segment.text, rel_start, rel_end))

    segments.sort(key=lambda item: (item.start, item.end))
    return segments, (None if cfg.absolute_times else clip_start)


def _load_keypoints(
    path: Path,
    fps: Optional[float],
    *,
    face_subset: Optional[Sequence[int]] = None,
) -> KeypointData:
    """Lee el archivo de keypoints y arma un layout estándar."""

    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de keypoints: {path}")

    ext = path.suffix.lower()
    layout_name: Optional[str] = None

    if ext == ".npz":
        with np.load(path, allow_pickle=True) as data:
            if "keypoints" not in data:
                raise KeyError("El .npz no contiene la clave 'keypoints'.")
            frames = data["keypoints"]
            if "layout" in data:
                layout_name = str(data["layout"])
    else:
        frames = np.load(path)

    if frames.ndim != 3:
        raise ValueError(f"Los keypoints deben tener forma (T, N, C); recibido {frames.shape}.")

    num_landmarks = frames.shape[1]
    layout = _resolve_mediapipe_layout(num_landmarks, face_subset=face_subset)
    connections = _resolve_mediapipe_connections(layout)

    fps_value = float(fps) if fps and fps > 0 else math.nan
    return KeypointData(
        frames=frames.astype(np.float32),
        layout=layout,
        connections=connections,
        fps=fps_value,
    )


def _resolve_path_by_stem(
    directory: Path,
    stem: str,
    allowed_suffixes: Optional[Sequence[str]] = None,
) -> Path:
    """Devuelve el primer archivo en ``directory`` cuyo stem coincide con ``stem``."""

    if allowed_suffixes:
        normalised = tuple(ext.lower() for ext in allowed_suffixes)
        allowed_suffixes = normalised
        for ext in normalised:
            candidate = directory / f"{stem}{ext}"
            if candidate.exists():
                return candidate
    for candidate in directory.glob(f"{stem}.*"):
        if not candidate.is_file():
            continue
        if allowed_suffixes and candidate.suffix.lower() not in allowed_suffixes:
            continue
        return candidate
    raise FileNotFoundError(
        f"No se encontró un archivo con stem '{stem}' dentro de {directory}."
    )


def _iter_clip_resources(
    videos_dir: Path,
    keypoints_dir: Path,
    subtitle_cfg: SubtitleConfig,
) -> Iterator[Tuple[Path, Path, SubtitleConfig, str]]:
    """Itera los clips presentes en ``meta.csv`` resolviendo rutas relativas."""

    if not videos_dir.is_dir():
        raise FileNotFoundError(f"El directorio de videos no existe: {videos_dir}")
    if not keypoints_dir.is_dir():
        raise FileNotFoundError(f"El directorio de keypoints no existe: {keypoints_dir}")

    _all_rows, filtered = _load_csv_entries(subtitle_cfg)

    def _prepare_entry(
        row: Dict[str, str],
        resolved_video: Optional[Path],
    ) -> Tuple[Path, Path, SubtitleConfig, str]:
        clip_id = row.get(subtitle_cfg.id_column)
        if not clip_id:
            raise ValueError(
                f"La fila {row} no contiene la columna {subtitle_cfg.id_column!r}."
            )

        video_value = row.get(subtitle_cfg.video_column)
        video_candidates = [clip_id]
        if video_value and video_value not in video_candidates:
            video_candidates.append(video_value)

        video_path = resolved_video
        if video_path is None:
            for stem in video_candidates:
                try:
                    video_path = _resolve_path_by_stem(
                        videos_dir,
                        stem,
                        VIDEO_EXTENSIONS,
                    )
                    break
                except FileNotFoundError:
                    continue
        if video_path is None:
            raise FileNotFoundError(
                f"No se encontró el video asociado a {clip_id!r} dentro de {videos_dir}."
            )

        try:
            keypoints_path = _resolve_path_by_stem(
                keypoints_dir,
                clip_id,
                (".npy", ".npz"),
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"No se encontró el archivo de keypoints para {clip_id!r} en {keypoints_dir}."
            ) from exc

        clip_cfg = replace(
            subtitle_cfg,
            target_id=clip_id,
            target_video=video_value or subtitle_cfg.target_video,
        )

        return video_path, keypoints_path, clip_cfg, clip_id

    if subtitle_cfg.target_id or subtitle_cfg.target_video:
        for row in filtered:
            yield _prepare_entry(row, None)
        return

    video_files = sorted(
        path
        for path in videos_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )

    rows_by_id: Dict[str, Dict[str, str]] = {}
    rows_by_video: Dict[str, List[Dict[str, str]]] = {}
    for row in filtered:
        clip_id = row.get(subtitle_cfg.id_column)
        if clip_id:
            rows_by_id[clip_id] = row
        video_value = row.get(subtitle_cfg.video_column)
        if video_value:
            rows_by_video.setdefault(video_value, []).append(row)

    yielded: set[str] = set()
    for video_path in video_files:
        stem = video_path.stem
        matched_rows: List[Dict[str, str]] = []
        row_by_id = rows_by_id.get(stem)
        if row_by_id:
            matched_rows.append(row_by_id)
        else:
            matched_rows.extend(rows_by_video.get(stem, []))

        if not matched_rows:
            print(
                f"Advertencia: no se hallaron filas en {subtitle_cfg.csv_path} "
                f"para el video {stem!r}."
            )
            continue

        for row in matched_rows:
            clip_id = row.get(subtitle_cfg.id_column)
            if not clip_id or clip_id in yielded:
                continue
            yielded.add(clip_id)
            yield _prepare_entry(row, video_path)

    missing_rows = [
        row
        for row in filtered
        if row.get(subtitle_cfg.id_column) and row.get(subtitle_cfg.id_column) not in yielded
    ]
    for row in missing_rows:
        clip_id = row.get(subtitle_cfg.id_column) or "<sin id>"
        print(
            f"Advertencia: no se encontró un video en {videos_dir} para el clip {clip_id!r}."
        )


def _format_preview(items: Sequence[str], limit: int = 5) -> str:
    """Construye una representación corta de ``items`` limitando su extensión."""

    limited = list(items[:limit])
    if not limited:
        return "-"
    preview = ", ".join(limited)
    remaining = len(items) - len(limited)
    if remaining > 0:
        preview += f", ... (+{remaining})"
    return preview


def _print_collection(
    label: str,
    items: Sequence[str],
    *,
    indent: str = "    ",
    limit: int = 5,
) -> None:
    """Imprime ``items`` respetando ``limit`` elementos visibles."""

    if not items:
        return
    preview = _format_preview(items, limit=limit)
    print(f"{indent}{label}: {preview}")


def _summarize_dataset(
    videos_dir: Path,
    keypoints_dir: Path,
    subtitle_cfg: SubtitleConfig,
) -> None:
    """Imprime estadísticas resumidas del dataset para depuración temprana."""

    all_rows, filtered_rows = _load_csv_entries(subtitle_cfg)

    video_files = sorted(
        path
        for path in videos_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    keypoint_files = sorted(
        path
        for path in keypoints_dir.iterdir()
        if path.is_file() and path.suffix.lower() in (".npy", ".npz")
    )

    missing_id_count = sum(
        1 for row in filtered_rows if not row.get(subtitle_cfg.id_column)
    )
    clip_ids = [
        row.get(subtitle_cfg.id_column)
        for row in filtered_rows
        if row.get(subtitle_cfg.id_column)
    ]
    unique_clip_ids = sorted(set(clip_ids))
    duplicates = sorted(
        clip_id for clip_id, count in Counter(clip_ids).items() if count > 1
    )

    matched_video_ids: Set[str] = set()
    matched_video_stems: Set[str] = set()
    missing_videos: Set[str] = set()
    for row in filtered_rows:
        clip_id = row.get(subtitle_cfg.id_column)
        if not clip_id:
            continue
        video_value = row.get(subtitle_cfg.video_column)
        candidates = [clip_id]
        if video_value and video_value not in candidates:
            candidates.append(video_value)
        resolved: Optional[Path] = None
        for candidate in candidates:
            try:
                resolved = _resolve_path_by_stem(
                    videos_dir,
                    candidate,
                    VIDEO_EXTENSIONS,
                )
                break
            except FileNotFoundError:
                continue
        if resolved is None:
            missing_videos.add(clip_id)
            continue
        matched_video_ids.add(clip_id)
        matched_video_stems.add(resolved.stem)

    matched_keypoint_ids: Set[str] = set()
    missing_keypoints: Set[str] = set()
    for clip_id in unique_clip_ids:
        try:
            _resolve_path_by_stem(
                keypoints_dir,
                clip_id,
                (".npy", ".npz"),
            )
        except FileNotFoundError:
            missing_keypoints.add(clip_id)
        else:
            matched_keypoint_ids.add(clip_id)

    video_orphans = [
        path.name
        for path in video_files
        if path.stem not in matched_video_stems
    ]
    keypoint_orphans = [
        path.name
        for path in keypoint_files
        if path.stem not in unique_clip_ids
    ]

    missing_videos_list = sorted(missing_videos)
    missing_keypoints_list = sorted(missing_keypoints)
    video_orphans_list = sorted(video_orphans)
    keypoint_orphans_list = sorted(keypoint_orphans)

    error_messages: List[str] = []
    if missing_id_count:
        error_messages.append(
            f"Filas sin columna {subtitle_cfg.id_column!r}: {missing_id_count}"
        )
    if duplicates:
        preview = _format_preview(duplicates)
        error_messages.append(
            f"IDs duplicados ({len(duplicates)}): {preview}"
        )
    if missing_videos_list:
        preview = _format_preview(missing_videos_list)
        error_messages.append(
            f"Videos faltantes ({len(missing_videos_list)}): {preview}"
        )
    if missing_keypoints_list:
        preview = _format_preview(missing_keypoints_list)
        error_messages.append(
            f"Keypoints faltantes ({len(missing_keypoints_list)}): {preview}"
        )
    if video_orphans_list:
        preview = _format_preview(video_orphans_list)
        error_messages.append(
            f"Videos sin referencia en CSV ({len(video_orphans_list)}): {preview}"
        )
    if keypoint_orphans_list:
        preview = _format_preview(keypoint_orphans_list)
        error_messages.append(
            f"Keypoints sin referencia en CSV ({len(keypoint_orphans_list)}): {preview}"
        )

    print("Resumen inicial del dataset")
    print("---------------------------")
    print(f"- CSV: {len(all_rows)} filas totales")
    print(f"  Tras filtros: {len(filtered_rows)} filas")
    print(
        f"  IDs únicos: {len(unique_clip_ids)} | sin ID: {missing_id_count} | "
        f"duplicados: {len(duplicates)}"
    )
    print(f"- Videos en {videos_dir}: {len(video_files)} archivos")
    print(
        f"  Coincidencias: {len(matched_video_ids)} | "
        f"faltantes: {len(missing_videos_list)} | "
        f"sobrantes: {len(video_orphans_list)}"
    )
    _print_collection("Faltantes (IDs)", missing_videos_list)
    _print_collection("Sobrantes (archivos)", video_orphans_list)
    print(f"- Keypoints en {keypoints_dir}: {len(keypoint_files)} archivos")
    print(
        f"  Coincidencias: {len(matched_keypoint_ids)} | "
        f"faltantes: {len(missing_keypoints_list)} | "
        f"sobrantes: {len(keypoint_orphans_list)}"
    )
    _print_collection("Faltantes (IDs)", missing_keypoints_list)
    _print_collection("Sobrantes (archivos)", keypoint_orphans_list)
    if error_messages:
        print(f"- Errores detectados: {len(error_messages)} tipo(s)")
        for message in error_messages:
            print(f"  • {message}")
    else:
        print("- Errores detectados: ninguno")
    print("")


def _select_subtitle(segments: Sequence[SubtitleEntry], timestamp: float) -> str:
    """Encuentra el subtítulo activo para ``timestamp``."""

    for segment in segments:
        if segment.contains(timestamp):
            return segment.text
    return ""


def _draw_keypoints(
    frame: np.ndarray,
    keypoints: np.ndarray,
    layout: Dict[str, List[int]],
    connections: Dict[str, List[Tuple[int, int]]],
    viewer_cfg: ViewerConfig,
    visible_mask: Optional[np.ndarray] = None,
) -> None:
    """Pinta los keypoints sobre la imagen."""

    colors = {
        "body": (0, 255, 0),
        "face": (255, 200, 0),
        "hand_l": (0, 165, 255),
        "hand_r": (255, 105, 180),
    }

    num_points = keypoints.shape[0]
    finite_mask = np.isfinite(keypoints[:, 0]) & np.isfinite(keypoints[:, 1])
    confidences = (
        keypoints[:, 2]
        if keypoints.shape[1] >= 3
        else np.ones(num_points, dtype=np.float32)
    )
    valid_mask = finite_mask.copy()
    if viewer_cfg.confidence_threshold > 0:
        valid_mask &= confidences >= viewer_cfg.confidence_threshold
    if visible_mask is not None:
        visibility = np.ones(num_points, dtype=bool)
        source = np.asarray(visible_mask, dtype=bool)
        limit = min(num_points, source.shape[0])
        visibility[:limit] = source[:limit]
        valid_mask &= visibility

    for name, indices in layout.items():
        color = colors.get(name, (255, 255, 255))
        segment_indices = list(indices)
        if name == "face":
            stride = max(1, viewer_cfg.face_point_stride)
            if stride > 1:
                segment_indices = segment_indices[::stride]
            if viewer_cfg.max_face_points is not None:
                limit = max(0, int(viewer_cfg.max_face_points))
                segment_indices = segment_indices[:limit]
        allowed_indices: Optional[Set[int]] = (
            set(segment_indices) if name == "face" else None
        )
        for idx in segment_indices:
            if idx >= num_points or not valid_mask[idx]:
                continue
            x, y = keypoints[idx, :2]
            point = (
                int(round(float(x))),
                int(round(float(y))),
            )
            cv2.circle(frame, point, 3, color, thickness=-1)

        if not viewer_cfg.draw_bones:
            continue

        for start_idx, end_idx in connections.get(name, []):
            if (
                start_idx >= num_points
                or end_idx >= num_points
                or not valid_mask[start_idx]
                or not valid_mask[end_idx]
            ):
                continue
            if allowed_indices is not None and (
                start_idx not in allowed_indices or end_idx not in allowed_indices
            ):
                continue
            start_point = (
                int(round(float(keypoints[start_idx, 0]))),
                int(round(float(keypoints[start_idx, 1]))),
            )
            end_point = (
                int(round(float(keypoints[end_idx, 0]))),
                int(round(float(keypoints[end_idx, 1]))),
            )
            cv2.line(frame, start_point, end_point, color, thickness=2)


def _resize_frame(frame: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return frame
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def _format_optional_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:0.2f}"


def _draw_info_panel(
    frame: np.ndarray,
    viewer_cfg: ViewerConfig,
    lines: Sequence[str],
) -> None:
    """Renderiza un panel semi-transparente con metadatos del visor."""

    if not lines:
        return

    scale = max(0.7, viewer_cfg.font_scale * 0.75)
    font_size = max(16, int(round(32 * scale)))
    font = _load_font(font_size)

    padding = 14
    margin = viewer_cfg.subtitle_margin
    max_width = max(int(frame.shape[1] * 0.65), 320)
    available_width = max(1, max_width - 2 * padding)

    wrapped: List[str] = []
    for line in lines:
        wrapped.extend(_wrap_text(line, font=font, max_width=available_width))

    if not wrapped:
        return

    dummy_image = Image.new("RGB", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_image)
    try:
        ascent, descent = font.getmetrics()
        base_height = ascent + descent
    except AttributeError:
        bbox = dummy_draw.textbbox((0, 0), "Ag", font=font)
        base_height = bbox[3] - bbox[1]

    text_bboxes = [dummy_draw.textbbox((0, 0), text, font=font) for text in wrapped]
    text_widths = [bbox[2] - bbox[0] for bbox in text_bboxes]
    line_height = base_height + max(6, int(round(base_height * 0.15)))

    panel_width = min(max_width, max(text_widths) + 2 * padding)
    panel_height = len(wrapped) * line_height + 2 * padding

    top_left = (margin, margin)
    bottom_right = (top_left[0] + panel_width, top_left[1] + panel_height)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_frame).convert("RGBA")

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(
        [top_left, bottom_right],
        fill=(0, 0, 0, int(255 * 0.6)),
    )

    image = Image.alpha_composite(image, overlay)
    draw = ImageDraw.Draw(image)

    for idx, text in enumerate(wrapped):
        x = top_left[0] + padding
        y = top_left[1] + padding + idx * line_height
        draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))

    updated = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    frame[:, :, :] = updated


def run_viewer(
    video_path: Path,
    keypoints_path: Path,
    subtitle_cfg: SubtitleConfig,
    viewer_cfg: ViewerConfig,
    video_fps: Optional[float] = None,
    keypoints_fps: Optional[float] = None,
    face_subset: Optional[Sequence[int]] = None,
) -> Literal["next", "previous", "quit"]:
    """Ejecuta el visor y retorna la acción solicitada por el usuario."""

    subtitles, clip_start = _load_subtitles(subtitle_cfg)
    keypoints_data = _load_keypoints(
        keypoints_path,
        keypoints_fps,
        face_subset=face_subset,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    fps_video = video_fps or cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps_video <= 0:
        raise RuntimeError("No fue posible inferir el FPS del video. Usa --fps para forzarlo.")

    fps_keypoints = (
        keypoints_fps
        or (None if math.isnan(keypoints_data.fps) else keypoints_data.fps)
        or fps_video
    )

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_duration = float("nan")
    if frame_count and frame_count > 0 and fps_video > 0:
        video_duration = frame_count / fps_video

    attempted_seek = False
    if viewer_cfg.seek_to_start and clip_start and clip_start > 0:
        if not math.isnan(video_duration):
            margin = 1.0 / max(fps_video, 1.0)
            if clip_start >= video_duration - margin:
                print(
                    "Advertencia: el inicio solicitado para el clip supera la duración del video. "
                    "Se usará el comienzo del archivo segmentado."
                )
            else:
                cap.set(cv2.CAP_PROP_POS_MSEC, clip_start * 1000)
                attempted_seek = True
        else:
            cap.set(cv2.CAP_PROP_POS_MSEC, clip_start * 1000)
            attempted_seek = True

    frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    total_keypoint_frames = keypoints_data.frames.shape[0]
    clip_reference = (clip_start or 0.0) if attempted_seek else 0.0
    playback_anchor = time.perf_counter()
    paused = viewer_cfg.start_paused
    show_subtitles = viewer_cfg.initial_subtitles
    show_keypoints = viewer_cfg.initial_keypoints
    pending_step = 0
    current_original: Optional[np.ndarray] = None
    current_video_pos_ms = float("nan")

    cv2.namedWindow(viewer_cfg.window_name, cv2.WINDOW_NORMAL)

    control_action: Literal["next", "previous", "quit"] = "next"

    try:
        while True:
            step_direction = pending_step
            pending_step = 0
            need_new_frame = (
                current_original is None or not paused or step_direction != 0
            )
            target_index: Optional[int] = None
            if step_direction != 0:
                target = frame_index + step_direction
                if frame_count and frame_count > 0:
                    max_index = max(0, int(frame_count) - 1)
                    target = max(0, min(target, max_index))
                else:
                    target = max(0, target)
                target_index = target
                need_new_frame = True

            if need_new_frame:
                if target_index is not None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_index)
                ok, grabbed = cap.read()
                if not ok:
                    if current_original is None and attempted_seek:
                        print(
                            "Advertencia: no fue posible posicionar el video en el "
                            "timestamp del CSV. Se reproducirá desde el inicio."
                        )
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        attempted_seek = False
                        clip_reference = 0.0
                        frame_index = 0
                        playback_anchor = time.perf_counter()
                        continue
                    if viewer_cfg.loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        frame_index = 0
                        current_original = None
                        current_video_pos_ms = 0.0
                        playback_anchor = time.perf_counter()
                        continue
                    break

                current_original = grabbed
                pos_frames = cap.get(cv2.CAP_PROP_POS_FRAMES)
                if target_index is not None:
                    frame_index = target_index
                elif pos_frames > 0:
                    frame_index = max(0, int(round(pos_frames)) - 1)
                else:
                    frame_index = max(0, frame_index + 1)
                current_video_pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

            if current_original is None:
                break

            original_frame = current_original
            frame = _resize_frame(original_frame.copy(), viewer_cfg.display_scale)

            if current_video_pos_ms > 0:
                video_pos = current_video_pos_ms / 1000.0
            else:
                video_pos = frame_index / fps_video
                if attempted_seek and clip_start:
                    video_pos += clip_start

            relative_time = max(0.0, video_pos - clip_reference)
            subtitle_time = (
                video_pos if subtitle_cfg.absolute_times else relative_time
            ) + viewer_cfg.video_offset
            subtitle_text = _select_subtitle(subtitles, subtitle_time)

            if show_keypoints and total_keypoint_frames > 0:
                kp_time = relative_time + viewer_cfg.keypoints_offset
                kp_frame = int(round(kp_time * fps_keypoints))
                kp_frame = max(0, min(kp_frame, total_keypoint_frames - 1))
                kp_array = keypoints_data.frames[kp_frame].copy()

                if viewer_cfg.normalised_keypoints:
                    height, width = original_frame.shape[:2]
                    kp_array[:, 0] *= width
                    kp_array[:, 1] *= height

                if viewer_cfg.display_scale != 1.0:
                    scale = viewer_cfg.display_scale
                    kp_array[:, :2] *= scale

                visibility = None
                if kp_array.shape[1] > 3:
                    visibility = kp_array[:, 3] > 0.0

                _draw_keypoints(
                    frame,
                    kp_array,
                    keypoints_data.layout,
                    keypoints_data.connections,
                    viewer_cfg,
                    visibility,
                )

            if show_subtitles and subtitle_text:
                _draw_subtitles(frame, subtitle_text, viewer_cfg)

            panel_lines = [
                f"Video: {video_path}",
                f"Keypoints: {keypoints_path}",
                f"CSV: {subtitle_cfg.csv_path}",
                (
                    f"Tiempo clip={relative_time:0.2f}s | video={video_pos:0.2f}s "
                    f"| frame={frame_index}"
                ),
                (
                    f"FPS video={fps_video:0.2f} | keypoints="
                    f"{_format_optional_float(fps_keypoints)}"
                ),
                (
                    f"Offset video={viewer_cfg.video_offset:+0.2f}s | "
                    f"keypoints={viewer_cfg.keypoints_offset:+0.2f}s"
                ),
                (
                    f"Estado={'Pausa' if paused else 'Reproducción'} | "
                    f"Subtítulos={'ON' if show_subtitles else 'OFF'} | "
                    f"Keypoints={'ON' if show_keypoints else 'OFF'}"
                ),
                "Controles: Espacio=pausa | ←/→=paso | S=subtítulos | K=keypoints",
                "           N=siguiente clip | P=clip anterior | Q=salir",
            ]
            _draw_info_panel(frame, viewer_cfg, panel_lines)

            cv2.imshow(viewer_cfg.window_name, frame)

            if paused:
                playback_anchor = time.perf_counter() - relative_time
                wait_arg = 100 if viewer_cfg.wait_time_ms != 0 else 1
            else:
                elapsed = time.perf_counter() - playback_anchor
                remaining = relative_time - elapsed
                if viewer_cfg.wait_time_ms <= 0:
                    wait_arg = 1
                    if remaining > 0:
                        wait_arg = max(wait_arg, int(round(remaining * 1000)))
                else:
                    wait_arg = viewer_cfg.wait_time_ms
                    if remaining > 0:
                        wait_arg = max(wait_arg, int(round(remaining * 1000)))

            key = cv2.waitKey(wait_arg)
            if key != -1:
                key &= 0xFF
                if key in (27, ord("q"), ord("Q")):
                    control_action = "quit"
                    break
                if key in (ord("n"), ord("N")):
                    control_action = "next"
                    break
                if key in (ord("p"), ord("P")):
                    control_action = "previous"
                    break
                if key == ord(" "):
                    paused = not paused
                    playback_anchor = time.perf_counter() - relative_time
                    continue
                if key in (81, ord(","), ord("[")):
                    paused = True
                    pending_step = -1
                    continue
                if key in (83, ord("."), ord("]")):
                    paused = True
                    pending_step = 1
                    continue
                if key in (ord("s"), ord("S")):
                    show_subtitles = not show_subtitles
                    continue
                if key in (ord("k"), ord("K")):
                    show_keypoints = not show_keypoints
                    continue
                if key == ord("r"):
                    paused = viewer_cfg.start_paused
                    show_subtitles = viewer_cfg.initial_subtitles
                    show_keypoints = viewer_cfg.initial_keypoints
                    playback_anchor = time.perf_counter() - relative_time
                    continue
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return control_action


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualiza video, keypoints MediaPipe y subtítulos en tiempo real.",
    )
    parser.add_argument("--video", type=Path, help="Ruta al video base.")
    parser.add_argument(
        "--keypoints",
        type=Path,
        help="Archivo .npz/.npy con los keypoints MediaPipe (forma (T, N, C)).",
    )
    parser.add_argument(
        "--csv",
        required=True,
        type=Path,
        help="CSV con subtítulos y columnas de tiempo (ej. meta.csv).",
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        help="Directorio base que contiene los videos segmentados (alternativa a --video).",
    )
    parser.add_argument(
        "--keypoints-dir",
        type=Path,
        help="Directorio con los keypoints MediaPipe por clip (alternativa a --keypoints).",
    )
    parser.add_argument("--segment-id", help="Valor de la columna 'id' a visualizar.")
    parser.add_argument("--video-id", help="Filtra filas por la columna 'video'.")
    parser.add_argument("--delimiter", default=";", help="Delimitador utilizado en el CSV.")
    parser.add_argument("--id-column", default="id", help="Nombre de la columna con IDs únicos.")
    parser.add_argument(
        "--video-column",
        default="video",
        help="Nombre de la columna que identifica el video fuente.",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Columna con el texto del subtítulo cuando no hay 'split'.",
    )
    parser.add_argument(
        "--start-column",
        default="start",
        help="Columna con el timestamp inicial del clip en segundos.",
    )
    parser.add_argument(
        "--end-column",
        default="end",
        help="Columna con el timestamp final del clip en segundos.",
    )
    parser.add_argument(
        "--split-column",
        default="split",
        help="Columna con la lista de segmentos parciales (literal de Python).",
    )
    parser.add_argument(
        "--absolute-times",
        action="store_true",
        help="No restar el inicio del clip; usa tiempos absolutos del CSV.",
    )
    parser.add_argument(
        "--window-name",
        default="SLT keypoint viewer",
        help="Nombre de la ventana de visualización.",
    )
    parser.add_argument(
        "--wait-ms",
        type=int,
        default=1,
        help="Tiempo de espera para cv2.waitKey (ms). Usa 0 para avanzar manualmente.",
    )
    parser.add_argument(
        "--start-paused",
        action="store_true",
        help="Inicia el visor en pausa y avanza solo con atajos de teclado.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Reinicia el video automáticamente al llegar al final.",
    )
    parser.add_argument(
        "--display-scale",
        type=float,
        default=1.0,
        help="Factor de escala aplicado al frame mostrado.",
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=1.2,
        help="Escala base para subtítulos y panel informativo.",
    )
    parser.add_argument(
        "--font-thickness",
        type=int,
        default=3,
        help="Grosor base de los textos renderizados.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.2,
        help="Confianza mínima para dibujar un keypoint.",
    )
    parser.add_argument(
        "--face-landmark-subset",
        type=_parse_face_subset_arg,
        help=(
            "Selecciona índices faciales (0-467) relativos a MediaPipe. "
            "Ejemplo: 1,133,362,13 para la vista mínima heredada."
        ),
    )
    parser.add_argument(
        "--face-point-stride",
        type=int,
        default=1,
        help="Dibuja un punto facial cada N índices (>=1).",
    )
    parser.add_argument(
        "--max-face-points",
        type=int,
        help=(
            "Máximo de puntos faciales tras aplicar el stride (permite 0 para ocultarlos)."
        ),
    )
    parser.add_argument(
        "--subtitle-width",
        type=int,
        default=900,
        help="Ancho máximo (px) reservado para subtítulos.",
    )
    parser.add_argument(
        "--subtitle-margin",
        type=int,
        default=24,
        help="Margen en píxeles alrededor de los subtítulos.",
    )
    subtitles_group = parser.add_mutually_exclusive_group()
    subtitles_group.add_argument(
        "--subtitles",
        dest="initial_subtitles",
        action="store_true",
        default=True,
        help="Muestra los subtítulos al iniciar (comportamiento por defecto).",
    )
    subtitles_group.add_argument(
        "--no-subtitles",
        dest="initial_subtitles",
        action="store_false",
        help="Oculta los subtítulos al iniciar la sesión.",
    )
    kp_norm_group = parser.add_mutually_exclusive_group()
    kp_norm_group.add_argument(
        "--normalised-keypoints",
        dest="normalised_keypoints",
        action="store_true",
        default=True,
        help=(
            "Interpreta los keypoints en coordenadas normalizadas [0, 1]. "
            "Usa --absolute-keypoints para desactivarlo."
        ),
    )
    kp_norm_group.add_argument(
        "--absolute-keypoints",
        dest="normalised_keypoints",
        action="store_false",
    )
    bones_group = parser.add_mutually_exclusive_group()
    bones_group.add_argument(
        "--draw-bones",
        dest="draw_bones",
        action="store_true",
        default=True,
        help="Dibuja líneas entre keypoints conectados.",
    )
    bones_group.add_argument(
        "--no-draw-bones",
        dest="draw_bones",
        action="store_false",
        help="Desactiva el dibujo de uniones entre keypoints.",
    )
    parser.add_argument(
        "--video-offset",
        type=float,
        default=0.0,
        help="Offset temporal (s) aplicado al video antes de mostrar subtítulos.",
    )
    parser.add_argument(
        "--keypoints-offset",
        type=float,
        default=0.0,
        help="Offset temporal (s) aplicado a los keypoints respecto del video.",
    )
    keypoints_group = parser.add_mutually_exclusive_group()
    keypoints_group.add_argument(
        "--draw-keypoints",
        dest="initial_keypoints",
        action="store_true",
        default=True,
        help="Dibuja los keypoints al iniciar (usa --no-draw-keypoints para desactivarlo).",
    )
    keypoints_group.add_argument(
        "--no-draw-keypoints",
        dest="initial_keypoints",
        action="store_false",
        help="Arranca con los keypoints ocultos en pantalla.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        help="FPS del video si no puede inferirse automáticamente.",
    )
    parser.add_argument(
        "--keypoints-fps",
        type=float,
        help="FPS de los keypoints cuando difiere del video.",
    )
    parser.add_argument(
        "--no-seek",
        action="store_true",
        help="No posicionar el video en el inicio del clip según el CSV.",
    )

    args = parser.parse_args()

    has_directories = args.videos_dir is not None or args.keypoints_dir is not None
    if has_directories:
        if not args.videos_dir or not args.keypoints_dir:
            parser.error("Debe especificar --videos-dir y --keypoints-dir para el modo por lotes.")
    else:
        missing = [
            flag
            for flag, value in (("--video", args.video), ("--keypoints", args.keypoints))
            if value is None
        ]
        if missing:
            parser.error(
                "Los argumentos --video y --keypoints son obligatorios cuando no se utilizan "
                "--videos-dir/--keypoints-dir."
            )

    if args.face_point_stride <= 0:
        parser.error("--face-point-stride debe ser >= 1.")
    if args.max_face_points is not None and args.max_face_points < 0:
        parser.error("--max-face-points no puede ser negativo.")

    return args


def main() -> None:
    args = _parse_args()

    subtitle_cfg = SubtitleConfig(
        csv_path=args.csv,
        delimiter=args.delimiter,
        id_column=args.id_column,
        video_column=args.video_column,
        text_column=args.text_column,
        start_column=args.start_column,
        end_column=args.end_column,
        split_column=args.split_column,
        target_id=args.segment_id,
        target_video=args.video_id,
        absolute_times=args.absolute_times,
    )

    viewer_cfg = ViewerConfig(
        window_name=args.window_name,
        wait_time_ms=args.wait_ms,
        loop=args.loop,
        display_scale=args.display_scale,
        font_scale=args.font_scale,
        font_thickness=args.font_thickness,
        subtitle_margin=args.subtitle_margin,
        subtitle_max_width=args.subtitle_width,
        confidence_threshold=args.confidence_threshold,
        normalised_keypoints=args.normalised_keypoints,
        video_offset=args.video_offset,
        keypoints_offset=args.keypoints_offset,
        seek_to_start=not args.no_seek,
        draw_bones=args.draw_bones,
        face_point_stride=args.face_point_stride,
        max_face_points=args.max_face_points,
        start_paused=args.start_paused,
        initial_subtitles=args.initial_subtitles,
        initial_keypoints=args.initial_keypoints,
    )

    if args.videos_dir and args.keypoints_dir:
        _summarize_dataset(args.videos_dir, args.keypoints_dir, subtitle_cfg)

        clips = list(
            _iter_clip_resources(args.videos_dir, args.keypoints_dir, subtitle_cfg)
        )
        if not clips:
            print("No se hallaron clips compatibles con el CSV y los filtros activos.")
            return

        total = len(clips)
        index = 0
        while 0 <= index < total:
            video_path, keypoints_path, clip_cfg, clip_id = clips[index]
            print(
                f"[{index + 1}/{total}] Visualizando clip {clip_id} "
                f"({video_path.name}, {keypoints_path.name})."
            )
            try:
                action = run_viewer(
                    video_path=video_path,
                    keypoints_path=keypoints_path,
                    subtitle_cfg=clip_cfg,
                    viewer_cfg=viewer_cfg,
                    video_fps=args.fps,
                    keypoints_fps=args.keypoints_fps,
                    face_subset=args.face_landmark_subset,
                )
            except KeyboardInterrupt:
                print("Interrupción del usuario. Finalizando.")
                break

            if action == "quit":
                print("Finalizando por solicitud del usuario.")
                break
            if action == "previous":
                if index == 0:
                    print("No hay clips anteriores; se mantiene el primero.")
                else:
                    index -= 1
                    print(f"Retrocediendo al clip {index + 1} de {total}.")
                continue

            index += 1
        else:
            print("Se completó la visualización de todos los clips.")
    else:
        run_viewer(
            video_path=args.video,
            keypoints_path=args.keypoints,
            subtitle_cfg=subtitle_cfg,
            viewer_cfg=viewer_cfg,
            video_fps=args.fps,
            keypoints_fps=args.keypoints_fps,
            face_subset=args.face_landmark_subset,
        )


if __name__ == "__main__":
    main()
