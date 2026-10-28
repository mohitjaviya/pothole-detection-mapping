"""Generate a tiny synthetic dashcam-style clip for smoke tests — no dataset,
no network, no model.

Two dark potholes grow and drift down-frame as the "car" approaches them, over
noisy asphalt with a horizon band. Enough for detect.py's loop, baseline.py
and the annotated-video writer to be exercised on a laptop.

Run (repo root): python tests/make_fake_video.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from src import config

WIDTH, HEIGHT, FPS, FRAMES = 640, 360, 30, 60
CLIP_PATH = config.OUTPUTS_DIR / "fake" / "synthetic_clip.mp4"

# (first frame, last frame, x centre) — pothole B appears once A is passed,
# so the clip has two well-separated "events" for the tracker to find.
POTHOLES = ((0, 30, 250), (30, 60, 420))


def draw_pothole(frame: np.ndarray, cx: int, cy: int, w: int, h: int) -> None:
    """Filled dark ellipse: reads as a pothole against grey asphalt."""
    cv2.ellipse(frame, (cx, cy), (max(w // 2, 2), max(h // 2, 2)), 0, 0, 360, (35, 35, 38), -1)


def make_clip(
    path: Path = CLIP_PATH,
    frames: int = FRAMES,
    fps: int = FPS,
    size: tuple[int, int] = (WIDTH, HEIGHT),
) -> Path:
    """Write the synthetic clip; returns its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")

    for index in range(frames):
        # Asphalt: mid-grey plus deterministic sensor noise, so Canny and YOLO
        # see texture rather than a synthetic flat field.
        frame = np.full((height, width, 3), 105, dtype=np.int16)
        noise = np.random.default_rng(index).integers(-6, 7, frame.shape, dtype=np.int16)
        frame = np.clip(frame + noise, 0, 255).astype(np.uint8)
        frame[: int(height * 0.45)] = (140, 150, 160)  # sky / horizon band

        for start, end, cx in POTHOLES:
            if start <= index < end:
                # t: 0 = far away (small, high in frame), 1 = under the car
                t = (index - start) / (end - start)
                draw_pothole(
                    frame,
                    cx,
                    int(height * 0.62 + height * 0.33 * t),
                    int(20 + 70 * t),
                    int(12 + 40 * t),
                )
        writer.write(frame)

    writer.release()
    print(f"[fake-video] {frames} frames @ {fps} fps, {width}x{height} -> {path}")
    return path


def main() -> None:
    make_clip()


if __name__ == "__main__":
    main()
