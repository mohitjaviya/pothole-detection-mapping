"""Assert the dashboard: run_pipeline() end to end with a stub model (no GPU,
no real clip), the no-GPS path, the no-detections path, and that the Streamlit
page actually renders.

Run (repo root): python tests/test_app.py
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from make_fake_data import FPS, START_ISO, main as make_events
from make_fake_video import FRAMES, make_clip
from src import config
from test_detect import FakeBoxes, FakeResult, StubModel

sys.path.insert(0, str(config.PROJECT_ROOT / "app"))
from streamlit_app import run_pipeline, severity_counts  # noqa: E402

FAKE_DIR = config.OUTPUTS_DIR / "fake"
FRAME_SKIP = 2


class SilentModel:
    """Stub that never detects anything — the 'clean road' case."""

    def track(self, frame, **kwargs) -> list[FakeResult]:
        return [FakeResult(FakeBoxes(None, [], []))]


def test_pipeline_with_gps(clip: Path) -> None:
    """Full chain: detect -> tracker -> severity -> geotag -> map."""
    result = run_pipeline(
        clip,
        gps_csv=FAKE_DIR / "gps_route.csv",
        frame_skip=FRAME_SKIP,
        start_time=datetime.fromisoformat(START_ISO).timestamp(),
        out_dir=FAKE_DIR / "app_test",
        model=StubModel(FRAMES, FRAME_SKIP),
    )

    assert result.stats["events_out"] == 2, result.stats
    assert len(result.events) == 2
    assert set(result.events["severity"]) <= {"small", "medium", "large"}
    assert result.untagged == 0, "the fake route covers every event"
    assert result.geotagged is not None and result.geotagged["latitude"].notna().all()
    assert result.annotated_video is not None and result.annotated_video.exists()
    assert result.map_html is not None and result.map_html.exists()

    markers = result.map_html.read_text(encoding="utf-8").lower().count("circlemarker(")
    assert markers == 2, f"expected 2 map markers, found {markers}"

    for name in ("track_log.csv", "pothole_events.csv", "pothole_events_severity.csv", "pothole_events_geo.csv"):
        assert (result.out_dir / name).exists(), f"{name} was not written"
    print(f"[test] full chain: 2 potholes, 2 markers, {len(result.events)} severity rows")


def test_pipeline_without_gps(clip: Path) -> None:
    """No GPS CSV -> no map, no invented coordinates, events still counted."""
    result = run_pipeline(
        clip,
        gps_csv=None,
        frame_skip=FRAME_SKIP,
        out_dir=FAKE_DIR / "app_test_nogps",
        model=StubModel(FRAMES, FRAME_SKIP),
    )
    assert len(result.events) == 2
    assert result.geotagged is None and result.map_html is None
    assert not (result.out_dir / "pothole_events_geo.csv").exists()
    print("[test] no GPS: events kept, map skipped, no coordinates invented")


def test_pipeline_no_detections(clip: Path) -> None:
    """Nothing detected must be a clean empty result, not a crash."""
    result = run_pipeline(
        clip,
        gps_csv=FAKE_DIR / "gps_route.csv",
        frame_skip=FRAME_SKIP,
        start_time=datetime.fromisoformat(START_ISO).timestamp(),
        out_dir=FAKE_DIR / "app_test_empty",
        model=SilentModel(),
    )
    assert result.events.empty, result.events
    assert result.stats["events_out"] == 0
    assert result.map_html is None and result.geotagged is None
    assert (result.out_dir / "pothole_events.csv").exists()
    print("[test] no detections: empty tables written, no map, no crash")


def test_severity_chart_order() -> None:
    """The severity chart must keep small/medium/large order, including zeros."""
    counts = severity_counts(pd.DataFrame({"severity": ["large", "large"]}))
    assert counts["severity"].tolist() == ["small", "medium", "large"], counts
    assert counts["potholes"].tolist() == [0, 0, 2], counts
    print("[test] severity chart: fixed order with zero-filled classes")


def test_page_renders() -> None:
    """The Streamlit page itself must render without raising."""
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(config.PROJECT_ROOT / "app" / "streamlit_app.py"), default_timeout=60)
    app.run()
    assert not app.exception, app.exception
    assert "pothole" in app.title[0].value.lower(), app.title[0].value
    assert len(app.info) >= 1, "the idle page should explain what it is waiting for"
    print(f"[test] page renders: title {app.title[0].value!r}, {len(app.info)} info block(s)")


def main() -> None:
    make_events()
    clip = make_clip()
    test_pipeline_with_gps(clip)
    test_pipeline_without_gps(clip)
    test_pipeline_no_detections(clip)
    test_severity_chart_order()
    test_page_renders()
    print("APP TEST PASSED")


if __name__ == "__main__":
    main()
