"""End-to-end test of the back half: severity -> geotag -> map_builder,
on the fake fixtures from make_fake_data.py. No video, no model, no network.

Run (repo root): python tests/test_chain.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime

import pandas as pd

from src import config
from src.geotag import geotag_events
from src.map_builder import build_map
from src.severity import add_severity

from make_fake_data import FPS, SPEED_KMH, START_ISO

FAKE_DIR = config.OUTPUTS_DIR / "fake"


def main() -> None:
    events = pd.read_csv(FAKE_DIR / "pothole_events.csv")

    # 1. severity: engineered bbox areas must classify exactly as designed
    sev = add_severity(events, frame_size=(640, 360))
    distribution = sev["severity"].value_counts().to_dict()
    assert distribution == {"small": 3, "medium": 3, "large": 2}, distribution
    print(f"[test] severity distribution: {distribution}")
    sev_path = FAKE_DIR / "pothole_events_severity.csv"
    sev.to_csv(sev_path, index=False)

    # 2. geotag: explicit-offset start time + fixed fps for the fake sources
    start_epoch = datetime.fromisoformat(START_ISO).timestamp()
    geo, untagged = geotag_events(
        sev, FAKE_DIR / "gps_route.csv", start_time=start_epoch, fps=FPS
    )
    assert untagged == 1, f"expected exactly 1 untagged event, got {untagged}"
    untagged_ids = geo[geo["latitude"].isna()]["event_id"].tolist()
    assert untagged_ids == [8], f"expected event 8 untagged, got {untagged_ids}"

    # every tagged event sits where the route says it should:
    # nearest 1 Hz fix = round(frame/fps); exact math, no jitter
    degrees_per_sec = (SPEED_KMH / 3.6) / 111_320.0
    for _, row in geo[geo["latitude"].notna()].iterrows():
        expected_lat = config.ROUTE_START_LAT + round(row["frame"] / FPS) * degrees_per_sec
        assert abs(row["latitude"] - expected_lat) < 1e-6, (
            f"event {row['event_id']}: {row['latitude']} != {expected_lat}"
        )
        assert row["longitude"] == config.ROUTE_START_LON
    print("[test] geotag: 7 events at exact route positions, event 8 left untagged")

    # 3. map: 7 markers (untagged row dropped), heatmap layer on
    fmap = build_map(geo, heatmap=True)
    map_path = config.OUTPUTS_DIR / "map.html"
    fmap.save(str(map_path))
    html = map_path.read_text(encoding="utf-8")
    marker_count = html.lower().count("circlemarker(")
    assert marker_count == 7, f"expected 7 markers in map.html, found {marker_count}"
    assert "pothole #" in html.lower(), "popup text missing"
    assert "heatmap" in html.lower(), "heatmap layer missing"
    print(f"[test] map: 7 markers + heatmap -> {map_path}")

    print("CHAIN TEST PASSED")


if __name__ == "__main__":
    main()
