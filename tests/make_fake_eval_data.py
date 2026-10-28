"""Generate fake evaluation fixtures: an eval manifest (clip x condition x
manual pothole count) and a tiny labelled image set for the baseline
detection metrics.

The manifest is engineered so every counting outcome is exercised:

    clip_a  day    gt 4  -> 4 events in make_fake_data.py   exact
    clip_b  night  gt 3  -> 4 events                        over-count by 1
    clip_c  wet    gt 2  -> no events at all                miss (counts as 0)
    clip_d  blur   gt 0  -> no events                       clean road, correct

Images: img1/img2 have label files, img3 has none (a clean-road negative —
missing labels mean "no potholes", not "unknown").

Run (repo root): python tests/make_fake_eval_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd

from src import config

WIDTH, HEIGHT = 640, 360
EVAL_DIR = config.OUTPUTS_DIR / "fake"
MANIFEST_PATH = EVAL_DIR / "eval_manifest.csv"
IMAGES_DIR = EVAL_DIR / "eval_images"
LABELS_DIR = EVAL_DIR / "eval_labels"

# (x1, y1, x2, y2) boxes drawn as dark ellipses — the same shape the label
# file describes, so real baseline boxes have something to match.
IMAGES: dict[str, list[tuple[int, int, int, int]]] = {
    "img1.png": [(300, 250, 340, 290)],
    "img2.png": [(120, 200, 200, 280), (400, 260, 440, 300)],
    "img3.png": [],  # clean road: no label file written at all
}

MANIFEST_ROWS = [
    # (source, condition, gt_count)
    ("clip_a.mp4", "day", 4),
    ("clip_b.mp4", "night", 3),
    ("clip_c.mp4", "wet", 2),
    ("clip_d.mp4", "blur", 0),
]


def label_line(box: tuple[int, int, int, int]) -> str:
    """Absolute xyxy -> one YOLO label line: 'cls cx cy w h', normalized."""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2 / WIDTH
    cy = (y1 + y2) / 2 / HEIGHT
    w = (x2 - x1) / WIDTH
    h = (y2 - y1) / HEIGHT
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def make_images() -> None:
    """Write the labelled image set + YOLO .txt files."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, boxes in IMAGES.items():
        frame = np.full((HEIGHT, WIDTH, 3), 105, dtype=np.uint8)
        for x1, y1, x2, y2 in boxes:
            cv2.ellipse(
                frame,
                ((x1 + x2) // 2, (y1 + y2) // 2),
                (max((x2 - x1) // 2, 2), max((y2 - y1) // 2, 2)),
                0,
                0,
                360,
                (35, 35, 38),
                -1,
            )
        cv2.imwrite(str(IMAGES_DIR / name), frame)
        if boxes:  # negatives get no label file on purpose
            (LABELS_DIR / f"{Path(name).stem}.txt").write_text(
                "\n".join(label_line(box) for box in boxes) + "\n", encoding="utf-8"
            )


def main() -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    make_images()
    pd.DataFrame(MANIFEST_ROWS, columns=["source", "condition", "gt_count"]).to_csv(
        MANIFEST_PATH, index=False
    )
    print(f"[fake-eval] manifest: {len(MANIFEST_ROWS)} clips -> {MANIFEST_PATH}")
    print(f"[fake-eval] images:   {len(IMAGES)} -> {IMAGES_DIR}")
    print(f"[fake-eval] labels:   {sum(1 for boxes in IMAGES.values() if boxes)} -> {LABELS_DIR}")


if __name__ == "__main__":
    main()
