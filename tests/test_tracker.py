"""Assert tracker.py: flicker + split tracks merge, short noise drops,
largest-detection anchoring is right, and nothing merges across sources.

Run (repo root): python tests/test_tracker.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import config
from src.tracker import build_events

FAKE_DIR = config.OUTPUTS_DIR / "fake"


def main() -> None:
    log = pd.read_csv(FAKE_DIR / "track_log.csv")
    events, stats = build_events(
        log,
        min_frames=config.MIN_TRACK_FRAMES,
        max_gap=config.MERGE_MAX_GAP_FRAMES,
        merge_iou=config.MERGE_IOU,
    )
    print(f"[test] stats: {stats}")
    assert stats == {"tracks_in": 8, "dropped_short": 1, "events_out": 5}, stats

    # (event_id, frame, first_frame, last_frame, source) — frame = largest box
    expected = [
        (1, 106, 100, 108, "clip_a.mp4"),
        (2, 206, 200, 206, "clip_a.mp4"),
        (3, 310, 300, 310, "clip_a.mp4"),
        (4, 506, 500, 506, "clip_a.mp4"),
        (5, 12, 10, 14, "clip_b.mp4"),
    ]
    actual = [
        (int(r.event_id), int(r.frame), int(r.first_frame), int(r.last_frame), r.source)
        for r in events.itertuples(index=False)
    ]
    assert actual == expected, actual

    # the 2-frame noise track (id 6, frames 400-402) produced nothing
    assert not (events["first_frame"] == 400).any(), "noise track survived"

    print("[test] flicker merged, split merged, noise dropped, cross-source isolated")
    print("TRACKER TEST PASSED")


if __name__ == "__main__":
    main()
