"""Build a short clip from the IVCNZ pothole PHOTOS — a WIRING-TEST clip only.

Each photo is held for HOLD_FRAMES so ByteTrack sees the same pothole in at
least MIN_TRACK_FRAMES processed frames, even with a large --frame-skip.
Photos with 1-4 labelled potholes are preferred so the track log stays
readable; negatives and crowded images are skipped.

Why this exists: no dashcam footage was available, but the pipeline still has
to be proven end to end. These are close-up photos, NOT road scenes — the
detection scores here say nothing about real-world performance. Use it to
check wiring only; never quote its numbers in the report.

Run (repo root): python tests/make_photo_clip.py
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from src import config

SOURCE_DIR = config.PROCESSED_DATASET_DIR / "val" / "images"
LABELS_DIR = config.PROCESSED_DATASET_DIR / "val" / "labels"
OUT_PATH = config.SAMPLE_VIDEOS_DIR / "clip_synthetic_photos.mp4"

WIDTH, HEIGHT, FPS = 1280, 720, 30
PHOTOS = 10
HOLD_FRAMES = 20  # 4 processed frames at --frame-skip 5 (> MIN_TRACK_FRAMES = 3)
SEED = 7
MIN_BOXES, MAX_BOXES = 1, 4


def letterbox(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale to fit, pad black — keeps pothole shapes undistorted."""
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    top = (height - resized.shape[0]) // 2
    left = (width - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return canvas


def box_count(label_path: Path) -> int:
    """Number of labelled potholes in a YOLO .txt (0 when the file is missing)."""
    if not label_path.exists():
        return 0
    return sum(1 for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip())


def pick_images() -> list[Path]:
    """Photos with a small, readable number of potholes, deterministically."""
    candidates = [
        path
        for path in sorted(SOURCE_DIR.glob("*.jpg"))
        if MIN_BOXES <= box_count(LABELS_DIR / f"{path.stem}.txt") <= MAX_BOXES
    ]
    if not candidates:
        raise RuntimeError(f"no suitable photos in {SOURCE_DIR} — run `python -m src.train --data-only` first")
    random.Random(SEED).shuffle(candidates)
    return candidates[:PHOTOS]


def main() -> None:
    images = pick_images()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(OUT_PATH), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {OUT_PATH}")

    for path in images:
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Could not read image: {path}")
        padded = letterbox(frame, WIDTH, HEIGHT)
        for _ in range(HOLD_FRAMES):
            writer.write(padded)
    writer.release()

    print(f"[photo-clip] {len(images)} photos x {HOLD_FRAMES} frames @ {FPS} fps -> {OUT_PATH}")
    print(f"[photo-clip] WIRING TEST ONLY — close-up photos, not dashcam footage; never report these scores")


if __name__ == "__main__":
    main()
