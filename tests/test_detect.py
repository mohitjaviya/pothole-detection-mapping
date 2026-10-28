"""Assert detect.py's contract without ultralytics/torch.

A stub model stands in for YOLO, so the frame loop, the CSV columns, the
ORIGINAL frame indices, the id-less-detection skip and the annotated-video
fps are all checked on a CPU-only machine. The stub's output is then fed
through tracker.py to prove the two modules really are joined by this CSV.

Run (repo root): python tests/test_detect.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import pandas as pd

from make_fake_video import FRAMES, FPS, make_clip
from src import config
from src.detect import LOG_COLUMNS, iter_frames, run_video
from src.tracker import build_events

FRAME_SKIP = 2


class FakeTensor:
    """Stand-in for a torch tensor: only the calls detect.py makes."""

    def __init__(self, values: list) -> None:
        self._values = values

    def int(self) -> "FakeTensor":
        return self

    def tolist(self) -> list:
        return self._values


class FakeBoxes:
    """Stand-in for ultralytics' Boxes object (id may be None before confirm)."""

    def __init__(self, ids: list[int] | None, xyxy: list[list[float]], conf: list[float]) -> None:
        self.id = None if ids is None else FakeTensor(ids)
        self.xyxy = FakeTensor(xyxy)
        self.conf = FakeTensor(conf)


class FakeResult:
    def __init__(self, boxes: FakeBoxes) -> None:
        self.boxes = boxes


class StubModel:
    """Two potholes: track 1 on frames 0-28, track 2 on frames 30-56, and one
    id-less detection on the final processed frame (58)."""

    def __init__(self, total_frames: int, frame_skip: int) -> None:
        self.processed = list(range(0, total_frames, frame_skip))
        self.calls = 0

    def track(self, frame, **kwargs) -> list[FakeResult]:
        # persist=True is the whole point of ByteTrack: ids must survive frames.
        assert kwargs.get("persist") is True, "tracker state must persist across frames"
        index = self.processed[self.calls]
        self.calls += 1
        if index < 30:
            return [FakeResult(FakeBoxes([1], [[100, 200, 180, 280]], [0.9]))]
        if index < 58:
            return [FakeResult(FakeBoxes([2], [[400, 220, 470, 290]], [0.8]))]
        return [FakeResult(FakeBoxes(None, [], []))]


def main() -> None:
    clip = make_clip()

    log, video_out = run_video(
        StubModel(FRAMES, FRAME_SKIP), clip, frame_skip=FRAME_SKIP, save_video=True
    )

    # 1. the CSV is exactly what tracker.py requires, in order
    assert list(log.columns) == LOG_COLUMNS, list(log.columns)
    assert len(log) == 29, f"expected 29 rows (30 frames - 1 id-less), got {len(log)}"
    assert not log.isna().any().any(), "track log contains NaN"

    # 2. frame skipping must NOT renumber: real video indices, 0, 2, 4, ...
    assert log["frame"].tolist() == list(range(0, 58, FRAME_SKIP)), log["frame"].tolist()
    assert sorted(log["track_id"].unique()) == [1, 2]
    assert set(log["source"]) == {clip.name}
    print(f"[test] track log: {len(log)} rows, frame ids {log['frame'].min()}-{log['frame'].max()}")

    # 3. annotated video: one written frame per processed frame, fps scaled down
    capture = cv2.VideoCapture(str(video_out))
    written = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    written_fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    assert written == FRAMES // FRAME_SKIP, f"expected {FRAMES // FRAME_SKIP} frames, got {written}"
    assert abs(written_fps - FPS / FRAME_SKIP) < 0.01, written_fps
    print(f"[test] annotated video: {written} frames @ {written_fps:.1f} fps -> {video_out}")

    # 4. --no-video path returns no path but the same log
    log_nowrite, no_video = run_video(
        StubModel(FRAMES, FRAME_SKIP), clip, frame_skip=FRAME_SKIP, save_video=False
    )
    assert no_video is None, no_video
    pd.testing.assert_frame_equal(log_nowrite, log)

    # 5. frame_skip below 1 is a hard error, not a silent no-op
    try:
        next(iter_frames(clip, 0))
    except ValueError:
        pass
    else:
        raise AssertionError("frame_skip=0 should raise ValueError")

    # 6. hand-off to tracker.py: two separate potholes, no cross-merge
    events, stats = build_events(log, config.MIN_TRACK_FRAMES, config.MERGE_MAX_GAP_FRAMES, config.MERGE_IOU)
    assert stats == {"tracks_in": 2, "dropped_short": 0, "events_out": 2}, stats
    assert events["event_id"].tolist() == [1, 2], events["event_id"].tolist()
    print(f"[test] tracker hand-off: {stats['events_out']} events from the detect log")

    print("DETECT TEST PASSED")


if __name__ == "__main__":
    main()
