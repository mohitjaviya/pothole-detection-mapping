"""Generate fake pothole events + a simulated GPS route for testing the
back half of the pipeline locally (no video, no model, no network).

Run (repo root): python tests/make_fake_data.py

Timeline: two short "videos" (clip_a 30 s, clip_b longer) both starting at
the same instant, over a 60 s one-Hz GPS route heading north from
Ahmedabad at ~35 km/h. One event deliberately falls far outside the GPS
window so the chain must leave it untagged, not crash or drop it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime

import pandas as pd

from src import config

# Explicit UTC offset (owner rule): fake timeline starts 2026-10-28 10:00 IST
START_ISO = "2026-10-28 10:00:00+05:30"
FPS = 30.0
ROUTE_SECONDS = 60      # 60 one-Hz GPS fixes
SPEED_KMH = 35.0        # straight-road northbound simulation

# Bbox fractions engineered to hit every severity class
# (frame 640x360 -> 230400 px^2; small < 0.004 < medium < 0.015 < large)
EVENTS = [
    # (event_id, frame, x1, y1, x2, y2, source)
    (1, 60, 300, 250, 322, 271, "clip_a.mp4"),    # 22x21 -> small
    (2, 150, 260, 240, 303, 283, "clip_a.mp4"),   # 43x43 -> medium
    (3, 420, 200, 220, 283, 303, "clip_a.mp4"),   # 83x83 -> large
    (4, 780, 320, 260, 342, 281, "clip_a.mp4"),   # 22x21 -> small
    (5, 90, 100, 270, 143, 313, "clip_b.mp4"),    # 43x43 -> medium
    (6, 300, 80, 230, 163, 313, "clip_b.mp4"),    # 83x83 -> large
    (7, 450, 150, 280, 172, 301, "clip_b.mp4"),   # 22x21 -> small
    (8, 20000, 200, 250, 243, 293, "clip_b.mp4"), # frame/fps >> route -> untagged
]


def make_gps_route() -> pd.DataFrame:
    """1 Hz fixes, driving north from ROUTE_START at SPEED_KMH."""
    start_epoch = datetime.fromisoformat(START_ISO).timestamp()
    meters_per_sec = SPEED_KMH / 3.6
    degrees_per_sec = meters_per_sec / 111_320.0  # ~111.32 km per degree of latitude
    return pd.DataFrame(
        {
            "timestamp_sec": [start_epoch + t for t in range(ROUTE_SECONDS)],
            "latitude": [config.ROUTE_START_LAT + t * degrees_per_sec for t in range(ROUTE_SECONDS)],
            "longitude": [config.ROUTE_START_LON] * ROUTE_SECONDS,
        }
    )


def main() -> None:
    out_dir = config.OUTPUTS_DIR / "fake"
    out_dir.mkdir(parents=True, exist_ok=True)

    events = pd.DataFrame(EVENTS, columns=["event_id", "frame", "x1", "y1", "x2", "y2", "source"])
    events_path = out_dir / "pothole_events.csv"
    events.to_csv(events_path, index=False)

    gps_path = out_dir / "gps_route.csv"
    make_gps_route().to_csv(gps_path, index=False)

    print(f"[fake] {len(events)} events -> {events_path}")
    print(f"[fake] {ROUTE_SECONDS} GPS fixes -> {gps_path}")


if __name__ == "__main__":
    main()
