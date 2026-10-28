"""Generate a fake raw track log exercising every tracker.py rule:
a steady track, flickering IDs (each fragment shorter than MIN_TRACK_FRAMES),
a split track (ID handoff with overlapping boxes), a short noise track
(dropped), and the same track id reused in another source (must NOT merge
across sources).

Run (repo root): python tests/make_fake_track_log.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import config

# (frame, track_id, x1, y1, x2, y2, conf, source)
ROWS = [
    # P1: steady id 1, unique largest box at frame 106 (84x80)
    (100, 1, 100, 200, 180, 280, 0.70, "clip_a.mp4"),
    (102, 1, 100, 200, 180, 280, 0.71, "clip_a.mp4"),
    (104, 1, 100, 200, 180, 280, 0.69, "clip_a.mp4"),
    (106, 1, 100, 200, 184, 280, 0.74, "clip_a.mp4"),
    (108, 1, 100, 200, 180, 280, 0.70, "clip_a.mp4"),
    # P2: flickering ids 2/3 — 2 frames each, would die without merging
    (200, 2, 300, 220, 360, 280, 0.80, "clip_a.mp4"),
    (202, 3, 300, 222, 360, 282, 0.75, "clip_a.mp4"),
    (204, 2, 301, 221, 361, 281, 0.80, "clip_a.mp4"),
    (206, 3, 298, 218, 362, 284, 0.78, "clip_a.mp4"),
    # P3: split track — id 4 hands off to id 5, boxes overlap (IoU ~0.8)
    (300, 4, 500, 250, 570, 310, 0.65, "clip_a.mp4"),
    (302, 4, 500, 250, 570, 310, 0.66, "clip_a.mp4"),
    (304, 4, 500, 250, 570, 310, 0.65, "clip_a.mp4"),
    (306, 5, 505, 252, 575, 312, 0.67, "clip_a.mp4"),
    (308, 5, 505, 252, 575, 312, 0.68, "clip_a.mp4"),
    (310, 5, 500, 248, 580, 314, 0.90, "clip_a.mp4"),  # 80x66 = largest
    # noise: 2-frame spurious track -> dropped by MIN_TRACK_FRAMES
    (400, 6, 50, 150, 80, 175, 0.55, "clip_a.mp4"),
    (402, 6, 52, 150, 82, 175, 0.55, "clip_a.mp4"),
    # P4: separate pothole, no overlap with anything
    (500, 7, 200, 280, 250, 330, 0.70, "clip_a.mp4"),
    (502, 7, 200, 280, 250, 330, 0.71, "clip_a.mp4"),
    (504, 7, 200, 280, 250, 330, 0.70, "clip_a.mp4"),
    (506, 7, 200, 280, 254, 330, 0.73, "clip_a.mp4"),  # 54x50 = largest
    # the same id number in ANOTHER source — must never merge across sources
    (10, 1, 100, 200, 180, 280, 0.72, "clip_b.mp4"),
    (12, 1, 100, 200, 184, 280, 0.76, "clip_b.mp4"),   # largest
    (14, 1, 100, 200, 180, 280, 0.72, "clip_b.mp4"),
]


def main() -> None:
    out_dir = config.OUTPUTS_DIR / "fake"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "track_log.csv"
    pd.DataFrame(
        ROWS, columns=["frame", "track_id", "x1", "y1", "x2", "y2", "conf", "source"]
    ).to_csv(path, index=False)
    print(f"[fake] {len(ROWS)} detections -> {path}")


if __name__ == "__main__":
    main()
