"""Severity classifier for pothole events: bbox area as a fraction of the
frame -> small / medium / large.

Input: pothole events CSV (event_id, frame, x1, y1, x2, y2, source).
Frame dimensions come from each source video in data/sample_videos/ via
OpenCV (no model, no torch); --frame-size WxH overrides for fake/test runs.

Caveat for the report: pixel area is a PROXY for physical size — boxes near
the bottom of the frame are closer to the camera, hence larger. Thresholds
are calibrated visually in a notebook.

CLI (repo root): python -m src.severity <events.csv> [--frame-size WxH] [--out PATH]
"""

import argparse
from pathlib import Path

import cv2
import pandas as pd

from src import config

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def frame_size_for(path: Path) -> tuple[int, int]:
    """(width, height) of an image or video file, via OpenCV."""
    if path.suffix.lower() in IMAGE_SUFFIXES:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Could not read image: {path}")
        height, width = image.shape[:2]
        return width, height
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    size = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    capture.release()
    if size[0] <= 0 or size[1] <= 0:
        raise RuntimeError(f"No frame size metadata in video: {path}")
    return size


def classify(area_frac: float) -> str:
    """Thresholds live in config (SEVERITY_*_FRAC) — calibrated in a notebook."""
    if area_frac < config.SEVERITY_SMALL_FRAC:
        return "small"
    if area_frac > config.SEVERITY_LARGE_FRAC:
        return "large"
    return "medium"


def add_severity(
    events: pd.DataFrame,
    frame_size: tuple[int, int] | None = None,
    videos_dir: Path = config.SAMPLE_VIDEOS_DIR,
) -> pd.DataFrame:
    """Add a `severity` column; one output row per input event, order kept.

    frame_size override applies to ALL sources (fake/test runs); otherwise
    each source is looked up in videos_dir. Missing videos are collected and
    reported in ONE error, not one per event.
    """
    missing_cols = {"frame", "x1", "y1", "x2", "y2", "source"} - set(events.columns)
    if missing_cols:
        raise RuntimeError(f"events CSV missing columns: {sorted(missing_cols)}")

    sizes: dict[str, tuple[int, int]] = {}
    if frame_size is not None:
        sizes = {source: frame_size for source in events["source"].unique()}
    else:
        missing: list[str] = []
        for source in events["source"].unique():
            video_path = videos_dir / source
            if video_path.exists():
                sizes[source] = frame_size_for(video_path)
            else:
                missing.append(source)
        if missing:
            raise RuntimeError(
                f"source video(s) not found in {videos_dir}: {missing} — "
                "pass --frame-size WxH to use a fixed size instead"
            )

    def severity_of(row: pd.Series) -> str:
        # Pixel area of the box as a fraction of the frame — the proxy for
        # physical pothole size (see module caveat).
        box_area = max(0, int(row["x2"]) - int(row["x1"])) * max(0, int(row["y2"]) - int(row["y1"]))
        width, height = sizes[row["source"]]
        return classify(box_area / (width * height))

    result = events.copy()
    result["severity"] = result.apply(severity_of, axis=1)
    return result


def parse_frame_size(text: str) -> tuple[int, int]:
    """'640x360' -> (640, 360)."""
    try:
        width, height = text.lower().split("x")
        return int(width), int(height)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"bad frame size {text!r} (expected WxH)") from exc


def main() -> None:
    """CLI entry: python -m src.severity <events.csv> [--frame-size WxH] [--out PATH]"""
    parser = argparse.ArgumentParser(description="Classify pothole severity from bbox area")
    parser.add_argument("input", type=Path, help="pothole events CSV")
    parser.add_argument(
        "--frame-size",
        type=parse_frame_size,
        default=None,
        help="fixed frame size WxH (overrides per-video lookup; for fake/test runs)",
    )
    parser.add_argument("--out", type=Path, default=None, help="output CSV (default: <stem>_severity.csv)")
    args = parser.parse_args()

    events = pd.read_csv(args.input)
    result = add_severity(events, frame_size=args.frame_size)
    out_path = args.out or args.input.with_name(args.input.stem + "_severity.csv")
    result.to_csv(out_path, index=False)
    print(f"[severity] rows: {len(result)} -> {out_path}")
    print(f"[severity] distribution: {result['severity'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
