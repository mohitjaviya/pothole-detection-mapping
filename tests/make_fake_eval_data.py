"""Generate fake evaluation fixtures: an eval manifest (clip x condition x
manual pothole count) and a tiny labelled image set for the baseline
detection metrics.

The manifest is engineered so every counting outcome is exercised:

    clip_a  day    gt 4  -> 4 events in make_fake_data.py   exact
    clip_b  night  gt 3  -> 4 events                        over-count by 1
    clip_c  wet    gt 2  -> no events at all                miss (counts as 0)
    clip_d  blur   gt 0  -> no events                       clean road, correct

Images: img1/img2 have label files, img3 has none and img4 has an EMPTY one —
both mean "no potholes here" (a clean-road negative), and a box predicted on
either of them is a false positive.

Prediction files for the --methods path (name=path): preds_weak.csv predicts
img1's box on all four images (tp 1, fp 3, fn 2) and preds_good.csv is
exact (tp 3, fp 0, fn 0).

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
WEAK_PREDICTIONS = EVAL_DIR / "preds_weak.csv"
GOOD_PREDICTIONS = EVAL_DIR / "preds_good.csv"

# (x1, y1, x2, y2) boxes drawn as dark ellipses — the same shape the label
# file describes, so real baseline boxes have something to match.
IMG1_BOX = (300, 250, 340, 290)
IMG2_BOXES = [(120, 200, 200, 280), (400, 260, 440, 300)]
IMAGES: dict[str, list[tuple[int, int, int, int]]] = {
    "img1.png": [IMG1_BOX],
    "img2.png": IMG2_BOXES,
    "img3.png": [],  # clean road: no label file written at all
    "img4.png": [],  # clean road: an EMPTY label file (also background)
}
# Images that DO get a .txt: img4's is empty on purpose.
LABEL_FILES = ("img1.png", "img2.png", "img4.png")

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
        if name in LABEL_FILES:
            # img4 gets an EMPTY file (not a missing one): both mean background.
            label_text = "\n".join(label_line(box) for box in boxes)
            (LABELS_DIR / f"{Path(name).stem}.txt").write_text(
                label_text + "\n" if label_text else "", encoding="utf-8"
            )


def make_predictions() -> None:
    """Two pre-computed prediction CSVs, for the --methods name=path path.

    weak: img1's box on every image -> tp 1, fp 3 (img2, img3, img4), fn 2.
    good: exactly the ground truth -> tp 3, fp 0, fn 0.
    """
    weak_rows = [
        {"image": name, "x1": IMG1_BOX[0], "y1": IMG1_BOX[1], "x2": IMG1_BOX[2], "y2": IMG1_BOX[3]}
        for name in IMAGES
    ]
    pd.DataFrame(weak_rows).to_csv(WEAK_PREDICTIONS, index=False)

    good_rows = [
        {"image": name, "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]}
        for name, boxes in IMAGES.items()
        for box in boxes
    ]
    pd.DataFrame(good_rows).to_csv(GOOD_PREDICTIONS, index=False)


def main() -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    make_images()
    make_predictions()
    pd.DataFrame(MANIFEST_ROWS, columns=["source", "condition", "gt_count"]).to_csv(
        MANIFEST_PATH, index=False
    )
    print(f"[fake-eval] manifest: {len(MANIFEST_ROWS)} clips -> {MANIFEST_PATH}")
    print(f"[fake-eval] images:   {len(IMAGES)} -> {IMAGES_DIR}")
    print(f"[fake-eval] labels:   {len(LABEL_FILES)} -> {LABELS_DIR}")
    print(f"[fake-eval] predictions: weak + good -> {EVAL_DIR}")


if __name__ == "__main__":
    main()
