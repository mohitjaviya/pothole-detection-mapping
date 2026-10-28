"""Assert src/pipeline.py: the one-command chain, with and without a model,
with real-GPS and simulated-route inputs, plus the CLI.

Everything runs on fake data: a stub model stands in for YOLOv8 (same trick as
tests/test_detect.py) and a pre-built track log proves the --track-log
shortcut needs no weights and no torch at all.

Run (repo root): python tests/test_pipeline.py
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from make_fake_data import FPS, START_ISO, main as make_events
from make_fake_video import FRAMES, make_clip
from src import config
from src.pipeline import run_pipeline, simulated_route
from test_detect import StubModel

FAKE_DIR = config.OUTPUTS_DIR / "fake"
FRAME_SKIP = 2
EXPECTED_EVENTS = 2


def make_track_log(clip: Path) -> Path:
    """Build a real track log from the stub model, to feed back via --track-log."""
    from src.detect import run_video

    log, _ = run_video(StubModel(FRAMES, FRAME_SKIP), clip, frame_skip=FRAME_SKIP, save_video=False)
    path = FAKE_DIR / "pipeline" / "input_track_log.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(path, index=False)
    return path


def test_simulated_route() -> None:
    """The fallback route is deterministic, 1 Hz, and covers the clip."""
    start = datetime.fromisoformat(START_ISO).timestamp()
    route = simulated_route(start, seconds=2.0)
    assert list(route.columns) == ["timestamp_sec", "latitude", "longitude"], list(route.columns)
    assert len(route) == config.ROUTE_SECONDS, f"expected >= {config.ROUTE_SECONDS} fixes"
    assert route["timestamp_sec"].iloc[0] == start
    assert (route["timestamp_sec"].diff().dropna() == 1).all(), "route must be one fix per second"
    assert route["latitude"].is_monotonic_increasing, "northbound straight line"
    assert (route["longitude"] == config.ROUTE_START_LON).all()
    print(f"[test] simulated route: {len(route)} fixes, northbound from the config origin")


def test_track_log_no_model(clip: Path, log_path: Path) -> None:
    """--track-log path: no weights, no model, simulated GPS fallback."""
    out_dir = FAKE_DIR / "pipeline_no_model"
    result = run_pipeline(
        clip,
        track_log=log_path,
        gps_csv=None,
        out_dir=out_dir,
        prefix="sim_",
        frame_skip=FRAME_SKIP,
    )

    assert result.gps_source == "simulated", result.gps_source
    assert result.stats["events_out"] == EXPECTED_EVENTS, result.stats
    assert len(result.events) == EXPECTED_EVENTS
    assert result.annotated_video is None, "detection was skipped, so no annotated clip"
    assert result.untagged == 0, "the simulated route covers every event"
    assert result.geotagged is not None and result.geotagged["latitude"].notna().all()

    # every stage's CSV is on disk, under the requested prefix
    for key, name in (
        ("track_log", "sim_track_log.csv"),
        ("events", "sim_pothole_events.csv"),
        ("severity", "sim_pothole_events_severity.csv"),
        ("geotag", "sim_pothole_events_geo.csv"),
        ("map", "sim_map.html"),
        ("simulated_route", "sim_simulated_route.csv"),
    ):
        assert result.paths[key] == out_dir / name, (key, result.paths.get(key))
        assert (out_dir / name).exists(), f"{name} was not written"

    markers = result.paths["map"].read_text(encoding="utf-8").lower().count("circlemarker(")
    assert markers == EXPECTED_EVENTS, f"expected {EXPECTED_EVENTS} markers, found {markers}"
    print(f"[test] no model: {len(result.events)} events, {markers} markers, simulated route")


def test_real_gps(clip: Path, log_path: Path) -> None:
    """A real GPS CSV plus an explicit start time is used as-is."""
    out_dir = FAKE_DIR / "pipeline_gps"
    result = run_pipeline(
        clip,
        gps_csv=FAKE_DIR / "gps_route.csv",
        track_log=log_path,
        start_time=datetime.fromisoformat(START_ISO).timestamp(),
        out_dir=out_dir,
        prefix="gps_",
    )
    assert result.gps_source == "csv", result.gps_source
    assert result.untagged == 0
    assert "simulated_route" not in result.paths, "a real GPS log must not trigger the fallback"
    assert not (out_dir / "gps_simulated_route.csv").exists()
    print("[test] real GPS: joined against the supplied CSV, no simulated route written")


def test_gps_without_start_time(clip: Path, log_path: Path) -> None:
    """A real GPS CSV without a start time is a hard error, not a silent guess."""
    try:
        run_pipeline(
            clip,
            gps_csv=FAKE_DIR / "gps_route.csv",
            track_log=log_path,
            start_time=None,
            out_dir=FAKE_DIR / "pipeline_no_start",
        )
    except RuntimeError as exc:
        assert "--start-time" in str(exc), exc
    else:
        raise AssertionError("a GPS CSV without a start time should raise")
    print("[test] real GPS without --start-time: rejected with a clear message")


def test_model_branch(clip: Path) -> None:
    """With no --track-log the pipeline runs detect itself (stub model here)."""
    result = run_pipeline(
        clip,
        gps_csv=FAKE_DIR / "gps_route.csv",
        start_time=datetime.fromisoformat(START_ISO).timestamp(),
        frame_skip=FRAME_SKIP,
        out_dir=FAKE_DIR / "pipeline_model",
        prefix="model_",
        save_video=False,
        model=StubModel(FRAMES, FRAME_SKIP),
    )
    assert result.stats["events_out"] == EXPECTED_EVENTS, result.stats
    assert len(result.log) == 29, f"expected 29 detections from the stub, got {len(result.log)}"
    print(f"[test] model branch: {len(result.log)} detections -> {len(result.events)} events")


def test_empty_track_log(clip: Path) -> None:
    """A clip with no detections must not invent events, a map, or coordinates."""
    empty = FAKE_DIR / "pipeline" / "empty_track_log.csv"
    empty.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=["frame", "track_id", "x1", "y1", "x2", "y2", "conf", "source"]).to_csv(
        empty, index=False
    )
    out_dir = FAKE_DIR / "pipeline_empty"
    result = run_pipeline(clip, track_log=empty, out_dir=out_dir, prefix="empty_")

    assert result.events.empty, result.events
    assert result.geotagged is None and result.map_html is None
    assert result.gps_source == "none", result.gps_source
    assert result.stats["events_out"] == 0
    assert (out_dir / "empty_pothole_events.csv").exists()
    print("[test] empty track log: empty tables, no map, no simulated route")


def test_cli(clip: Path, log_path: Path) -> None:
    """python -m src.pipeline --track-log runs the chain without a model."""
    out_dir = FAKE_DIR / "pipeline_cli"
    completed = subprocess.run(
        [
            sys.executable, "-m", "src.pipeline", str(clip),
            "--track-log", str(log_path),
            "--no-video",
            "--out-dir", str(out_dir),
        ],
        cwd=config.PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "weights:" not in completed.stdout, "the --track-log path must not load weights"
    assert "SIMULATED" in completed.stdout, "the simulated-GPS warning must be visible"
    assert "GPS source: simulated" in completed.stdout, completed.stdout
    for name in (
        "synthetic_clip_track_log.csv",
        "synthetic_clip_pothole_events.csv",
        "synthetic_clip_pothole_events_severity.csv",
        "synthetic_clip_pothole_events_geo.csv",
        "synthetic_clip_map.html",
        "synthetic_clip_simulated_route.csv",
    ):
        assert (out_dir / name).exists(), f"{name} was not written"
    print("[test] CLI: chain runs from a track log, warning printed, all CSVs written")


def main() -> None:
    make_events()
    clip = make_clip()
    log_path = make_track_log(clip)
    test_simulated_route()
    test_track_log_no_model(clip, log_path)
    test_real_gps(clip, log_path)
    test_gps_without_start_time(clip, log_path)
    test_model_branch(clip)
    test_empty_track_log(clip)
    test_cli(clip, log_path)
    print("PIPELINE TEST PASSED")


if __name__ == "__main__":
    main()
