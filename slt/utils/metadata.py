"""Utilities for working with metadata fields."""
from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class SplitSegment:
    """Representation for a text segment in ``meta.csv`` split annotations."""

    text: str
    start: float
    end: float


def sanitize_time_value(value: str) -> Optional[float]:
    """Normalise a numeric time value coming from metadata sources."""

    if value is None:
        return None

    if isinstance(value, (int, float)):
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return None
        return result

    text = str(value).strip()
    if not text:
        return None

    normalised = text.replace(",", ".").replace(" ", "")

    if normalised.count(".") > 1:
        parts = normalised.split(".")
        integer = parts[0]
        decimals = "".join(parts[1:])
        normalised = f"{integer}.{decimals}" if decimals else integer

    try:
        result = float(normalised)
    except ValueError:
        return None

    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _normalise_double_quotes(raw: str) -> str:
    """Replace Unicode fancy quotes with regular double quotes."""

    replacements = {
        "“": '"',
        "”": '"',
        "‟": '"',
        "„": '"',
    }
    for source, target in replacements.items():
        raw = raw.replace(source, target)
    return raw


def parse_split_column(raw: str) -> List[SplitSegment]:
    """Parse the ``split`` column from ``meta.csv`` into structured segments."""

    if raw is None:
        return []

    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return []

    normalised = _normalise_double_quotes(text)
    try:
        entries = ast.literal_eval(normalised)
    except (ValueError, SyntaxError) as exc:  # pragma: no cover - error path
        raise ValueError(f"Split column inválido: {raw!r}") from exc

    if not isinstance(entries, (list, tuple)):
        raise ValueError(f"Split column inválido: {raw!r}")

    segments: List[SplitSegment] = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, (list, tuple)) or len(entry) != 3:
            raise ValueError(
                f"Segmento inválido en posición {idx}: {entry!r}"  # pragma: no cover
            )
        segment_text = str(entry[0])
        start = sanitize_time_value(entry[1])
        end = sanitize_time_value(entry[2])
        if start is None or end is None:
            raise ValueError(
                f"Segmento con tiempos inválidos en posición {idx}: {entry!r}"
            )
        if start < 0 or end < 0 or end <= start:
            raise ValueError(
                f"Segmento con rango temporal inválido en posición {idx}: {entry!r}"
            )
        segments.append(SplitSegment(segment_text, start, end))

    return segments
