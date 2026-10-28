"""Assert src/robustness.py: every degradation really changes the frame, the
sweep wiring produces one report row per condition, retention is computed
against the clean pass, and the report/CLI paths work.

The model is a stub that only looks at frame brightness, so the retention
numbers are checked against a known, explainable behaviour instead of whatever
YOLOv8 happens to do (measuring the real model is the CLI's job, on real
footage).

Run (repo root): python tests/test_robustness.py
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd

from make_fake_video import make_clip
from src import config
from src.detect import LOG_COLUMNS, video_meta
from src.robustness import (
    REPORT_COLUMNS,
    add_retention,
    degrade,
    motion_kernel,
    run_condition,
    run_sweep,
    sweep_row,
    write_degraded_video,
    write_report,
)
from test_detect import FakeBoxes, FakeResult

FAKE_DIR = config.OUTPUTS_DIR / "fake"
FRAME_SKIP = 2
BRIGHT_CUTOFF = 90.0  # the fake clip's asphalt sits around 105-140


class BrightnessStub:
    """Detects one box while the frame is bright enough, nothing when it is dim.

    The smallest model that reacts to a degradation at all — exactly what the
    retention arithmetic needs to be checked against.
    """

    def __init__(self, cutoff: float = BRIGHT_CUTOFF) -> None:
        self.cutoff = cutoff
        self.calls = 0

    def track(self, frame, **kwargs) -> list[FakeResult]:
        self.calls += 1
        if frame.mean() < self.cutoff:
            return [FakeResult(FakeBoxes(None, [], []))]
        return [FakeResult(FakeBoxes([1], [[100, 200, 180, 280]], [0.9]))]


def first_frame(clip: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(clip))
    ok, frame = capture.read()
    capture.release()
    assert ok, "could not read the fake clip"
    return frame


def test_degradations(clip: Path) -> None:
    """Each degradation must change the pixels and keep the frame shape/dtype."""
    frame = first_frame(clip)
    clean_mean = float(frame.mean())

    dimmed = degrade(frame, "dim")
    blurred = degrade(frame, "blur")
    noisy = degrade(frame, "noise")
    smeared = degrade(frame, "motion")

    for name, image in (("dim", dimmed), ("blur", blurred), ("noise", noisy), ("motion", smeared)):
        assert image.shape == frame.shape, f"{name} changed the frame size"
        assert image.dtype == np.uint8, f"{name} changed the dtype"
        assert not np.array_equal(image, frame), f"{name} did not change the frame"

    # dim: clearly darker. blur: less high-frequency energy. noise: more spread.
    assert dimmed.mean() < clean_mean * 0.6, (dimmed.mean(), clean_mean)
    assert cv2.Laplacian(blurred, cv2.CV_64F).var() < cv2.Laplacian(frame, cv2.CV_64F).var()
    assert noisy.std() > frame.std()
    # motion smear is horizontal: neighbouring pixels along a row become closer
    assert np.abs(np.diff(smeared.astype(np.int16), axis=1)).mean() < np.abs(
        np.diff(frame.astype(np.int16), axis=1)
    ).mean()

    # deterministic: the same seed gives the same noise frame
    assert np.array_equal(degrade(frame, "noise"), degrade(frame, "noise"))
    # clean is a pass-through, not a copy that could drift
    assert degrade(frame, "clean") is frame

    assert motion_kernel(5).shape == (1, 5)
    assert abs(float(motion_kernel(5).sum()) - 1.0) < 1e-6, "kernel must preserve brightness"
    try:
        motion_kernel(4)
    except ValueError as exc:
        assert "odd" in str(exc), exc
    else:
        raise AssertionError("an even kernel length should be rejected")
    print(f"[test] degradations: dim {clean_mean:.0f}->{dimmed.mean():.0f}, blur/noise/motion differ")


def test_unknown_condition(clip: Path) -> None:
    """A typo'd condition must fail loudly, not produce an empty report row."""
    try:
        run_condition(BrightnessStub(), clip, "foggy", frame_skip=FRAME_SKIP)
    except ValueError as exc:
        assert "unknown condition" in str(exc), exc
    else:
        raise AssertionError("an unknown condition should raise")

    try:
        run_sweep(clip, conditions=("clean", "foggy"), model_factory=BrightnessStub)
    except RuntimeError as exc:
        assert "unknown condition" in str(exc), exc
    else:
        raise AssertionError("run_sweep should reject an unknown condition up front")
    print("[test] unknown condition rejected")


def test_condition_log(clip: Path) -> None:
    """run_condition returns detect.py's own log columns and a frame count."""
    log, processed = run_condition(BrightnessStub(), clip, "clean", frame_skip=FRAME_SKIP)
    assert list(log.columns) == LOG_COLUMNS, list(log.columns)
    assert processed == 30, f"expected 30 processed frames, got {processed}"
    assert len(log) == processed, "the bright stub should fire on every clean frame"
    print(f"[test] clean pass: {processed} frames, {len(log)} detections")


def test_sweep(clip: Path) -> pd.DataFrame:
    """One row per condition; dim loses detections, blur/noise keep them."""
    table = run_sweep(
        clip,
        conditions=("clean", "dim", "blur", "noise", "motion"),
        frame_skip=FRAME_SKIP,
        model_factory=lambda: BrightnessStub(),
    )
    assert list(table.columns) == REPORT_COLUMNS, list(table.columns)
    assert table["condition"].tolist() == ["clean", "dim", "blur", "noise", "motion"], table
    assert set(table["source"]) == {clip.name}, "the report must name the clip it measured"

    by_condition = table.set_index("condition")
    assert by_condition.loc["clean", "detections"] == 30
    assert by_condition.loc["clean", "detection_retention"] == 1.0
    assert by_condition.loc["dim", "detections"] == 0, by_condition.loc["dim"]
    assert by_condition.loc["dim", "detection_retention"] == 0.0
    assert by_condition.loc["blur", "detections"] == 30
    assert by_condition.loc["noise", "detections"] == 30
    print(table.to_string(index=False))
    print("[test] sweep: dim -> 0 detections, blur/noise keep all 30")
    return table


def test_retention_without_clean() -> None:
    """No clean row (or a clean run that found nothing) leaves retention blank
    instead of dividing by zero."""
    empty_log = pd.DataFrame(columns=LOG_COLUMNS)
    table = add_retention(
        pd.DataFrame([sweep_row("clip.mp4", "dim", empty_log, 10)], columns=REPORT_COLUMNS)
    )
    assert table["detection_retention"].isna().all(), table
    assert table["track_retention"].isna().all(), table
    print("[test] retention: blank when there is no clean baseline")


def test_report_and_degraded_video(clip: Path, table: pd.DataFrame) -> None:
    """The report CSV and a degraded clip are written where the report expects."""
    path = write_report(table, FAKE_DIR / "robustness.csv")
    written = pd.read_csv(path)
    assert list(written.columns) == REPORT_COLUMNS, list(written.columns)
    assert len(written) == 5

    video_path = write_degraded_video(clip, "dim", FAKE_DIR / "synthetic_clip_dim.mp4")
    fps, width, height = video_meta(video_path)
    assert (width, height) == (640, 360), (width, height)
    assert abs(fps - 30.0) < 0.5, fps
    assert video_path.exists() and video_path.stat().st_size > 0
    print(f"[test] report: {path.name} ({len(written)} rows); degraded clip {video_path.name} {fps:.0f} fps")


def test_cli(clip: Path) -> None:
    """The CLI runs the real weights path when models/best.pt exists."""
    if not config.DETECT_WEIGHTS.exists():
        print("[test] CLI skipped: models/best.pt not present (stub-only environment)")
        return
    out_path = FAKE_DIR / "robustness_cli.csv"
    completed = subprocess.run(
        [
            sys.executable, "-m", "src.robustness", str(clip),
            "--frame-skip", "5",
            "--conditions", "clean", "dim",
            "--out", str(out_path),
        ],
        cwd=config.PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    written = pd.read_csv(out_path)
    assert written["condition"].tolist() == ["clean", "dim"], written
    clean_row = written.loc[written["condition"] == "clean"].iloc[0]
    if int(clean_row["detections"]) == 0:
        # best.pt finds nothing on the synthetic clip (it is trained on real
        # photos), so retention must stay blank rather than divide by zero.
        assert pd.isna(clean_row["detection_retention"]), clean_row
    else:
        assert clean_row["detection_retention"] == 1.0, clean_row
    print(f"[test] CLI: real weights, {len(written)} rows -> {out_path.name}")


def main() -> None:
    clip = make_clip()
    test_degradations(clip)
    test_unknown_condition(clip)
    test_condition_log(clip)
    table = test_sweep(clip)
    test_retention_without_clean()
    test_report_and_degraded_video(clip, table)
    test_cli(clip)
    print("ROBUSTNESS TEST PASSED")


if __name__ == "__main__":
    main()
