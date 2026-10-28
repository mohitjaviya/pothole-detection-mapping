"""Robustness sweep: how much does detection degrade when the FOOTAGE does?

One clip, one model, one frame skip — five versions of the same frames:

    clean    the clip as recorded
    dim      brightness x config.ROBUSTNESS_DIM_FACTOR (night / under-exposed)
    blur     Gaussian blur (defocus, dirty lens, heavy compression)
    noise    Gaussian pixel noise (high-ISO sensor noise, low light)
    motion   horizontal smear (speed blur at a slow shutter)

The degradation is applied to each frame INSIDE the detection loop rather than
written to five new video files: same decode, same frame indices, same model
call pattern, so a difference in the counts can only come from the pixels.
`--write-videos` also saves the degraded clips when the report wants pictures.

Each condition gets a FRESH model instance. ByteTrack keeps state between
track() calls (persist=True), so reusing one model across conditions would
carry track ids from the clean pass into the noisy one and flatter the result.

Output: reports/robustness.csv — per condition: frames processed, detections,
unique tracks, detections per frame, and retention versus the clean pass.

CLI (repo root):
    python -m src.robustness data/sample_videos/clip.mp4 --frame-skip 5
    python -m src.robustness data/sample_videos/clip.mp4 --conditions clean dim blur
    python -m src.robustness data/sample_videos/clip.mp4 --write-videos
"""

import argparse
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd

from src import config, detect

REPORT_COLUMNS = [
    "source",
    "condition",
    "frames_processed",
    "detections",
    "unique_tracks",
    "detections_per_frame",
    "detection_retention",
    "track_retention",
]


# ---------------------------------------------------------------------------
# Degradations (pure frame -> frame functions, no model involved)
# ---------------------------------------------------------------------------
def motion_kernel(length: int = config.ROBUSTNESS_MOTION_KERNEL) -> np.ndarray:
    """Horizontal averaging kernel: a 1 x length box filter = motion smear."""
    if length < 1 or length % 2 == 0:
        raise ValueError(f"motion kernel length must be odd and >= 1, got {length}")
    kernel = np.zeros((1, length), dtype=np.float32)
    kernel[0, :] = 1.0 / length
    return kernel


def degrade(
    frame: np.ndarray,
    condition: str,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply one named degradation to a BGR frame.

    `rng` keeps the noise reproducible; when omitted a fresh generator seeded
    with config.ROBUSTNESS_SEED is used, so a single call is still
    deterministic.
    """
    if condition == "clean":
        return frame
    if condition == "dim":
        # alpha scales every channel; a low alpha is an under-exposed frame.
        return cv2.convertScaleAbs(frame, alpha=config.ROBUSTNESS_DIM_FACTOR, beta=0)
    if condition == "blur":
        return cv2.GaussianBlur(frame, config.ROBUSTNESS_BLUR_KERNEL, 0)
    if condition == "noise":
        generator = rng if rng is not None else np.random.default_rng(config.ROBUSTNESS_SEED)
        noise = generator.normal(0.0, config.ROBUSTNESS_NOISE_SIGMA, frame.shape)
        return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if condition == "motion":
        return cv2.filter2D(frame, -1, motion_kernel())
    raise ValueError(f"unknown condition {condition!r} — allowed: {list(config.ROBUSTNESS_CONDITIONS)}")


def write_degraded_video(video: Path, condition: str, out_path: Path) -> Path:
    """Write one degraded copy of the clip (for report screenshots)."""
    fps, width, height = detect.video_meta(video)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {out_path}")
    rng = np.random.default_rng(config.ROBUSTNESS_SEED)
    try:
        for _, frame in detect.iter_frames(video, 1):  # every frame: this is a real clip
            writer.write(degrade(frame, condition, rng))
    finally:
        writer.release()
    return out_path


# ---------------------------------------------------------------------------
# Detection per condition
# ---------------------------------------------------------------------------
def run_condition(
    model: Any,
    video: Path,
    condition: str,
    conf: float = config.CONF_THRESHOLD,
    frame_skip: int = config.FRAME_SKIP,
    tracker: Path = config.TRACKER_CONFIG,
) -> tuple[pd.DataFrame, int]:
    """Run detect.py's tracking loop on one degraded version of the clip.

    Returns the same track-log table detect.py produces, plus the number of
    processed frames. `model` only needs ultralytics' track() interface, which
    is what lets tests/test_robustness.py run the sweep with a stub model.
    """
    rng = np.random.default_rng(config.ROBUSTNESS_SEED)
    rows: list[dict[str, object]] = []
    processed = 0
    for index, frame in detect.iter_frames(video, frame_skip):
        degraded = degrade(frame, condition, rng)
        results = model.track(
            degraded,
            persist=True,          # ByteTrack state must survive frames
            conf=conf,
            imgsz=config.IMGSZ,
            tracker=str(tracker),
            verbose=False,
        )
        rows.extend(detect.boxes_to_rows(results[0], index, video.name) if results else [])
        processed += 1
    return pd.DataFrame(rows, columns=detect.LOG_COLUMNS), processed


def sweep_row(source: str, condition: str, log: pd.DataFrame, processed: int) -> dict[str, object]:
    """One report row for one condition (retention filled in later).

    `source` is the clip name: the CSV carries it so a reader can never
    mistake a wiring-clip sweep for dashcam footage.
    """
    detections = len(log)
    return {
        "source": source,
        "condition": condition,
        "frames_processed": processed,
        "detections": detections,
        "unique_tracks": int(log["track_id"].nunique()) if detections else 0,
        "detections_per_frame": round(detections / processed, 3) if processed else 0.0,
        "detection_retention": None,
        "track_retention": None,
    }


def add_retention(table: pd.DataFrame) -> pd.DataFrame:
    """Retention = condition / clean, so 'how much survived' is one number.

    With no clean row (or a clean run that found nothing) retention stays
    blank: dividing by zero would invent a 0 % result that never happened.
    """
    result = table.copy()
    clean = result[result["condition"] == "clean"]
    if clean.empty:
        return result
    base_detections = int(clean["detections"].iloc[0])
    base_tracks = int(clean["unique_tracks"].iloc[0])
    if base_detections:
        result["detection_retention"] = (result["detections"] / base_detections).round(3)
    if base_tracks:
        result["track_retention"] = (result["unique_tracks"] / base_tracks).round(3)
    return result


def run_sweep(
    video: Path,
    weights: Path = config.DETECT_WEIGHTS,
    conditions: tuple[str, ...] = config.ROBUSTNESS_CONDITIONS,
    conf: float = config.CONF_THRESHOLD,
    frame_skip: int = config.FRAME_SKIP,
    model_factory: Callable[[], Any] | None = None,
) -> pd.DataFrame:
    """Run every condition on one clip and return the report table.

    `model_factory` returns a FRESH model per condition (see module docstring);
    tests pass a stub factory, the CLI leaves it None and reloads the weights.
    """
    if not video.exists():
        raise RuntimeError(f"video not found: {video}")
    unknown = [condition for condition in conditions if condition not in config.ROBUSTNESS_CONDITIONS]
    if unknown:
        raise RuntimeError(f"unknown condition(s) {unknown} — allowed: {list(config.ROBUSTNESS_CONDITIONS)}")

    rows: list[dict[str, object]] = []
    for condition in conditions:
        print(f"[robustness] {condition}: detecting…")
        model = model_factory() if model_factory is not None else detect.load_model(weights)
        log, processed = run_condition(model, video, condition, conf=conf, frame_skip=frame_skip)
        rows.append(sweep_row(video.name, condition, log, processed))
        print(
            f"[robustness] {condition}: {processed} frames, {len(log)} detections, "
            f"{rows[-1]['unique_tracks']} tracks"
        )
    return add_retention(pd.DataFrame(rows, columns=REPORT_COLUMNS))


def write_report(table: pd.DataFrame, out_path: Path = config.ROBUSTNESS_REPORT) -> Path:
    """Write the sweep table as CSV (committed; feeds the report's robustness section)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    """CLI entry: python -m src.robustness <video> [flags]"""
    parser = argparse.ArgumentParser(
        description="Detect potholes on synthetically degraded copies of one clip"
    )
    parser.add_argument("video", type=Path, help="dashcam clip (data/sample_videos/)")
    parser.add_argument("--weights", type=Path, default=config.DETECT_WEIGHTS)
    parser.add_argument("--conf", type=float, default=config.CONF_THRESHOLD)
    parser.add_argument("--frame-skip", type=int, default=config.FRAME_SKIP)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(config.ROBUSTNESS_CONDITIONS),
        help=f"subset of {list(config.ROBUSTNESS_CONDITIONS)}",
    )
    parser.add_argument("--out", type=Path, default=config.ROBUSTNESS_REPORT)
    parser.add_argument(
        "--write-videos",
        action="store_true",
        help=f"also save the degraded clips to {config.ROBUSTNESS_OUT_DIR}",
    )
    args = parser.parse_args()

    if not args.video.exists():
        raise SystemExit(f"video not found: {args.video}")

    if args.write_videos:
        for condition in args.conditions:
            if condition == "clean":
                continue
            path = write_degraded_video(
                args.video, condition, config.ROBUSTNESS_OUT_DIR / f"{args.video.stem}_{condition}.mp4"
            )
            print(f"[robustness] wrote {path}")

    table = run_sweep(
        args.video,
        weights=args.weights,
        conditions=tuple(args.conditions),
        conf=args.conf,
        frame_skip=args.frame_skip,
    )
    path = write_report(table, args.out)
    print(table.to_string(index=False))
    print(f"[robustness] report: {path}")


if __name__ == "__main__":
    main()
