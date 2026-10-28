"""YOLOv8 detection + ByteTrack tracking on one dashcam clip -> raw track log.

This is the only module that touches the model. `ultralytics` is imported
LAZILY inside load_model(), so this file — and its test, which swaps in a stub
model — runs on a machine without torch installed.

Per processed frame:
    cv2 frame -> model.track(frame, persist=True) -> boxes + track ids
One CSV row per detection: frame, track_id, x1, y1, x2, y2, conf, source.
`frame` is the ORIGINAL video frame index (not the processed counter), so
geotag.py can compute event time = start + frame / fps. tracker.py then
collapses the rows into one event per unique pothole.

ByteTrack tuning lives in configs/bytetrack.yaml (static YAML, committed);
the knobs and the reason for each value are explained in that file.

CLI (repo root):
    python -m src.detect <video> [--out PATH] [--weights PATH] [--conf F]
                       [--frame-skip N] [--tracker PATH] [--no-video]
"""

import argparse
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import pandas as pd

from src import config

# Exactly what tracker.py requires — the two modules are joined by this CSV.
LOG_COLUMNS = ["frame", "track_id", "x1", "y1", "x2", "y2", "conf", "source"]

# One colour per track id (BGR) so ID flicker/split is visible in the report
# video: the same pothole changing colour means the tracker changed its id.
TRACK_COLORS: tuple[tuple[int, int, int], ...] = (
    (0, 255, 0),    # green
    (0, 200, 255),  # amber
    (255, 128, 0),  # blue
    (255, 0, 255),  # magenta
    (0, 128, 255),  # orange
    (128, 0, 255),  # violet
    (0, 255, 255),  # yellow
    (255, 255, 0),  # cyan
)


def load_model(weights: Path) -> Any:
    """Load YOLO weights (lazy ultralytics import: torch is not a data-stage
    dependency). Returns the ultralytics YOLO object."""
    if not weights.exists():
        raise RuntimeError(
            f"weights not found: {weights} — fine-tune on Colab (src/train.py) and copy "
            "the best.pt into models/, or pass --weights models/yolov8n.pt for a smoke test"
        )
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "ultralytics is not installed here — run `pip install ultralytics` "
            "(it pulls torch, ~2 GB) or run detect.py where it is available"
        ) from exc

    print(f"[detect] weights: {weights}")
    return YOLO(str(weights))


def video_meta(path: Path) -> tuple[float, int, int]:
    """(fps, width, height) of a clip. fps falls back to 30 when the container
    carries no metadata — geotag.py makes the same assumption."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"No frame size metadata in video: {path}")
    return (float(fps) if fps > 0 else 30.0), width, height


def iter_frames(video_path: Path, frame_skip: int) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (original frame index, frame) for every frame_skip-th frame.

    The ORIGINAL index is yielded, not a processed counter: everything
    downstream (geotag time = frame / fps, tracker gap merging) assumes real
    video frame numbers, so frame skipping must not renumber anything.
    """
    if frame_skip < 1:
        raise ValueError(f"frame_skip must be >= 1, got {frame_skip}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % frame_skip == 0:
                yield index, frame
            index += 1
    finally:
        capture.release()


def boxes_to_rows(result: Any, frame_index: int, source: str) -> list[dict[str, object]]:
    """Turn one ultralytics Results object into track-log rows.

    `boxes.id` is None while ByteTrack has not confirmed a track yet (and on
    every frame when the tracker is off), and such a detection is skipped:
    tracker.py keys everything on track_id, so an id-less box is unusable.
    Zero-area boxes are skipped too — severity.py divides by the frame area,
    so a degenerate box would only produce noise.
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None or getattr(boxes, "id", None) is None:
        return []
    rows: list[dict[str, object]] = []
    for track_id, (x1, y1, x2, y2), conf in zip(
        boxes.id.int().tolist(), boxes.xyxy.tolist(), boxes.conf.tolist()
    ):
        if x2 <= x1 or y2 <= y1:
            continue
        rows.append(
            {
                "frame": frame_index,
                "track_id": int(track_id),
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "conf": round(float(conf), 4),
                "source": source,
            }
        )
    return rows


def annotate(frame: np.ndarray, rows: list[dict[str, object]]) -> np.ndarray:
    """Draw every track box with its id + confidence on a copy of the frame."""
    annotated = frame.copy()
    for row in rows:
        x1, y1, x2, y2 = (int(row[key]) for key in ("x1", "y1", "x2", "y2"))
        color = TRACK_COLORS[int(row["track_id"]) % len(TRACK_COLORS)]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        # Label sits just above the box, clamped so it never leaves the frame.
        cv2.putText(
            annotated,
            f"#{row['track_id']} {float(row['conf']):.2f}",
            (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return annotated


def run_video(
    model: Any,
    video_path: Path,
    conf: float = config.CONF_THRESHOLD,
    frame_skip: int = config.FRAME_SKIP,
    tracker: Path = config.TRACKER_CONFIG,
    save_video: bool = config.WRITE_ANNOTATED_VIDEO,
) -> tuple[pd.DataFrame, Path | None]:
    """Track potholes through one clip.

    `model` only has to provide ultralytics' track() interface, which is what
    lets tests/test_detect.py run the whole loop with a stub model and no GPU.

    persist=True keeps ByteTrack's state between calls — without it every
    frame would start a brand-new track and each pothole would get a new id.
    """
    fps, width, height = video_meta(video_path)
    writer: cv2.VideoWriter | None = None
    video_out: Path | None = None
    if save_video:
        config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        video_out = config.OUTPUTS_DIR / f"detect_{video_path.stem}_annotated.mp4"
        # Only every frame_skip-th frame is written, so the writer fps is the
        # source fps divided by frame_skip — the clip plays at real speed.
        writer = cv2.VideoWriter(
            str(video_out),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps / frame_skip,
            (width, height),
        )

    rows: list[dict[str, object]] = []
    processed = 0
    last_index = 0
    for index, frame in iter_frames(video_path, frame_skip):
        results = model.track(
            frame,
            persist=True,           # keep tracker state across frames
            conf=conf,              # drop weak boxes before tracking
            imgsz=config.IMGSZ,     # same inference size as training
            tracker=str(tracker),   # configs/bytetrack.yaml
            verbose=False,
        )
        frame_rows = boxes_to_rows(results[0], index, video_path.name) if results else []
        rows.extend(frame_rows)
        if writer is not None:
            writer.write(annotate(frame, frame_rows))
        processed += 1
        last_index = index

    if writer is not None:
        writer.release()

    log = pd.DataFrame(rows, columns=LOG_COLUMNS)
    print(f"[detect] frames processed: {processed} (last video frame index {last_index}, skip {frame_skip})")
    print(f"[detect] detections: {len(log)}")
    if not log.empty:
        print(f"[detect] unique track ids: {log['track_id'].nunique()}")
    else:
        print("[detect] WARNING: no tracked detections — check --conf, or that the weights know 'pothole'")
    return log, video_out


def main() -> None:
    """CLI entry: python -m src.detect <video> [flags]"""
    parser = argparse.ArgumentParser(description="YOLOv8 + ByteTrack -> raw track log")
    parser.add_argument("video", type=Path, help="dashcam clip (data/sample_videos/)")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="track log CSV (default: outputs/track_log_<clip>.csv)",
    )
    parser.add_argument("--weights", type=Path, default=config.DETECT_WEIGHTS)
    parser.add_argument(
        "--conf",
        type=float,
        default=config.CONF_THRESHOLD,
        help="confidence cutoff (config: CONF_THRESHOLD)",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=config.FRAME_SKIP,
        help="process every Nth frame, CPU speed (config: FRAME_SKIP)",
    )
    parser.add_argument(
        "--tracker",
        type=Path,
        default=config.TRACKER_CONFIG,
        help="ByteTrack YAML (config: TRACKER_CONFIG)",
    )
    parser.add_argument("--no-video", action="store_true", help="skip the annotated video")
    args = parser.parse_args()

    if not args.video.exists():
        raise SystemExit(f"video not found: {args.video}")

    model = load_model(args.weights)
    log, video_out = run_video(
        model,
        args.video,
        conf=args.conf,
        frame_skip=args.frame_skip,
        tracker=args.tracker,
        save_video=not args.no_video,
    )

    out_path = args.out or config.OUTPUTS_DIR / f"track_log_{args.video.stem}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(out_path, index=False)
    print(f"[detect] track log: {out_path}")
    if video_out is not None:
        print(f"[detect] annotated: {video_out}")


if __name__ == "__main__":
    main()
