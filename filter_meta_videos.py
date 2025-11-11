#!/usr/bin/env python3
"""Filter meta.csv rows by available videos."""
import argparse
import csv
from pathlib import Path
from typing import Iterable, List

VIDEO_EXTS: tuple[str, ...] = (".mp4", ".mkv", ".mov", ".avi", ".webm")


def find_video(clip_id: str, videos_dir: Path) -> Path | None:
    """Return a matching video path for ``clip_id`` or ``None`` if missing."""
    if clip_id is None:
        return None
    clip_id = clip_id.strip() if isinstance(clip_id, str) else str(clip_id).strip()
    if not clip_id:
        return None

    for ext in VIDEO_EXTS:
        candidate = videos_dir / f"{clip_id}{ext}"
        if candidate.exists():
            return candidate
    matches: List[Path] = sorted(videos_dir.glob(f"{clip_id}.*"))
    return matches[0] if matches else None


def filter_meta(rows: Iterable[dict[str, str]], videos_dir: Path) -> List[dict[str, str]]:
    filtered: List[dict[str, str]] = []
    for row in rows:
        clip_id = row.get("id") or row.get("video_id") or row.get("video")
        if not clip_id:
            raise ValueError("meta.csv must contain an 'id', 'video_id', or 'video' column")
        clip_id = "" if clip_id is None else str(clip_id).strip()
        if clip_id and find_video(clip_id, videos_dir):
            filtered.append(row)
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter meta.csv to rows with available videos.")
    parser.add_argument("--meta", type=Path, default=Path("meta.csv"), help="Input CSV path (semicolon-separated).")
    parser.add_argument("--videos-dir", type=Path, required=True, help="Directory containing video files.")
    parser.add_argument("--output", type=Path, default=Path("meta_2.csv"), help="Filtered CSV output path.")
    args = parser.parse_args()

    if not args.meta.is_file():
        raise FileNotFoundError(f"CSV not found: {args.meta}")
    if not args.videos_dir.is_dir():
        raise NotADirectoryError(f"Videos directory not found: {args.videos_dir}")

    with args.meta.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter=";")
        if reader.fieldnames is None:
            raise ValueError("CSV missing header row")
        rows = list(reader)

    filtered_rows = filter_meta(rows, args.videos_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=reader.fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(filtered_rows)

    kept = len(filtered_rows)
    dropped = len(rows) - kept
    print(f"Wrote {kept} rows to {args.output} (dropped {dropped}).")


if __name__ == "__main__":
    main()
