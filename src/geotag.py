"""Attach GPS coordinates to pothole events by matching video timestamps
to GPS fixes.

Timestamp convention (owner-approved):
- GPS CSV column `timestamp_sec` = Unix epoch seconds; ISO 8601 datetime
  strings are auto-detected and converted (naive datetimes assumed
  Asia/Kolkata, +05:30).
- Video start: --start-time (explicit UTC offset accepted; naive values get
  the +05:30 default) or parsed from dashcam-style filenames
  (YYYYMMDD_HHMMSS). Event time = start_epoch + frame / fps.
- Join: nearest GPS fix; gaps beyond GPS_TOLERANCE_SEC leave
  latitude/longitude blank AND are counted — never silently dropped.
  Nearest-fix only, no interpolation (stretch goal).

CLI (repo root):
    python -m src.geotag <events.csv> <gps.csv>
                         [--start-time ISO] [--fps F] [--out PATH]
"""

import argparse
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from src import config


def _offset_timezone(offset: str) -> timezone:
    """'+05:30' -> timezone. IST has no DST, so a fixed offset is exact."""
    try:
        sign = -1 if offset.startswith("-") else 1
        hours, minutes = offset.lstrip("+-").split(":")
        return timezone(sign * timedelta(hours=int(hours), minutes=int(minutes)))
    except ValueError as exc:
        raise RuntimeError(f"bad UTC offset {offset!r} in config") from exc


DEFAULT_TZ: timezone = _offset_timezone(config.GPS_DEFAULT_TZ_OFFSET)


def to_epoch(value: Any) -> float:
    """Numeric -> epoch seconds; anything else -> ISO 8601 datetime
    (naive strings get the Asia/Kolkata default offset)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        parsed = datetime.fromisoformat(str(value).strip())  # handles 'Z' too
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=DEFAULT_TZ)
        return parsed.timestamp()


def load_gps_track(gps_csv: Path) -> pd.DataFrame:
    """Load + normalize the GPS CSV: numeric OR ISO timestamps -> epoch,
    sorted by time."""
    gps = pd.read_csv(gps_csv)
    missing = {"timestamp_sec", "latitude", "longitude"} - set(gps.columns)
    if missing:
        raise RuntimeError(f"GPS CSV missing columns {sorted(missing)}: {gps_csv}")
    gps = gps.assign(timestamp_sec=[to_epoch(value) for value in gps["timestamp_sec"]])
    gps = gps.sort_values("timestamp_sec").reset_index(drop=True)
    if gps.empty:
        raise RuntimeError(f"GPS CSV has no rows: {gps_csv}")
    return gps


def parse_start_time(text: str) -> float:
    """--start-time value -> epoch seconds. Explicit UTC offset accepted;
    naive datetimes get the +05:30 default (Asia/Kolkata)."""
    try:
        parsed = datetime.fromisoformat(text.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"unparseable start time {text!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=DEFAULT_TZ)
    return parsed.timestamp()


FILENAME_TIME_RE = re.compile(r"(\d{8})[_\- ]?(\d{6})")


def start_time_from_filename(source: str) -> float | None:
    """Dashcam-style names like 20261027_153045.mp4 -> epoch seconds (IST)."""
    match = FILENAME_TIME_RE.search(source)
    if match is None:
        return None
    parsed = datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")
    return parsed.replace(tzinfo=DEFAULT_TZ).timestamp()


def video_fps(path: Path) -> float:
    """fps of a video via OpenCV; missing metadata falls back to 30 (the
    geotag math needs a number)."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    return float(fps) if fps > 0 else 30.0


def attach_gps(
    events: pd.DataFrame, gps: pd.DataFrame, tolerance: float
) -> tuple[pd.DataFrame, int]:
    """Nearest GPS fix by absolute time gap; fixes farther than `tolerance`
    leave latitude/longitude NaN and are COUNTED (never silently dropped)."""
    gps_times = gps["timestamp_sec"].to_numpy()
    lat = gps["latitude"].to_numpy()
    lon = gps["longitude"].to_numpy()
    times = events["event_ts"].to_numpy(dtype=float)

    right = np.clip(np.searchsorted(gps_times, times), 0, len(gps_times) - 1)
    left = np.clip(right - 1, 0, len(gps_times) - 1)
    gap_left = np.abs(times - gps_times[left])
    gap_right = np.abs(times - gps_times[right])
    nearest = np.where(gap_left <= gap_right, left, right)
    in_range = np.minimum(gap_left, gap_right) <= tolerance

    result = events.copy()
    result["latitude"] = np.where(in_range, lat[nearest], np.nan)
    result["longitude"] = np.where(in_range, lon[nearest], np.nan)
    return result, int((~in_range).sum())


def geotag_events(
    events: pd.DataFrame,
    gps_csv: Path,
    start_time: float | None = None,
    fps: float | None = None,
    videos_dir: Path = config.SAMPLE_VIDEOS_DIR,
) -> tuple[pd.DataFrame, int]:
    """Compute event timestamps and attach lat/lon. start_time: epoch
    seconds (None -> dashcam filename parse). fps override applies to ALL
    sources (fake/test runs)."""
    missing_cols = {"frame", "source"} - set(events.columns)
    if missing_cols:
        raise RuntimeError(f"events CSV missing columns: {sorted(missing_cols)}")

    sources = events["source"].unique().tolist()
    if start_time is None:
        if len(sources) != 1:
            raise RuntimeError("no --start-time given and multiple sources — pass one explicitly")
        start_time = start_time_from_filename(sources[0])
        if start_time is None:
            raise RuntimeError(f"no --start-time and no timestamp in filename {sources[0]!r}")

    if fps is None:
        not_found = [s for s in sources if not (videos_dir / s).exists()]
        if not_found:
            raise RuntimeError(f"source video(s) not found in {videos_dir}: {not_found} — pass --fps")
        fps_map = {source: video_fps(videos_dir / source) for source in sources}
    else:
        fps_map = {source: fps for source in sources}

    result = events.copy()
    result["event_ts"] = result.apply(
        lambda row: start_time + float(row["frame"]) / fps_map[row["source"]], axis=1
    )

    gps = load_gps_track(gps_csv)
    video_lo, video_hi = result["event_ts"].min(), result["event_ts"].max()
    gps_lo, gps_hi = gps["timestamp_sec"].min(), gps["timestamp_sec"].max()

    def fmt(ts: float) -> str:
        return datetime.fromtimestamp(ts, tz=DEFAULT_TZ).strftime("%Y-%m-%d %H:%M:%S %z")

    print(f"[geotag] video time range: {fmt(video_lo)} -> {fmt(video_hi)}")
    print(f"[geotag] GPS time range:   {fmt(gps_lo)} -> {fmt(gps_hi)}")
    tolerance = config.GPS_TOLERANCE_SEC
    if video_hi < gps_lo - tolerance or video_lo > gps_hi + tolerance:
        print(
            "[geotag] WARNING: video and GPS time ranges do NOT overlap — "
            "likely a timezone or start-time mistake; most events will stay untagged"
        )

    result, untagged = attach_gps(result, gps, tolerance)
    print(
        f"[geotag] geotagged: {len(result) - untagged} / {len(result)} "
        f"(untagged: {untagged}, tolerance: {tolerance}s)"
    )
    return result, untagged


def main() -> None:
    """CLI entry: python -m src.geotag <events.csv> <gps.csv> [--start-time ISO] [--fps F]"""
    parser = argparse.ArgumentParser(description="Attach GPS coordinates to pothole events")
    parser.add_argument("events", type=Path, help="pothole events CSV (severity optional)")
    parser.add_argument("gps", type=Path, help="GPS CSV: timestamp_sec, latitude, longitude")
    parser.add_argument(
        "--start-time",
        type=parse_start_time,
        default=None,
        help='video start as ISO 8601 ("2026-10-28 10:00:00+05:30"); naive -> Asia/Kolkata',
    )
    parser.add_argument("--fps", type=float, default=None, help="fps override for all sources")
    parser.add_argument("--out", type=Path, default=None, help="output CSV (default: <stem>_geo.csv)")
    args = parser.parse_args()

    events = pd.read_csv(args.events)
    result, _ = geotag_events(events, args.gps, start_time=args.start_time, fps=args.fps)
    out_path = args.out or args.events.with_name(args.events.stem + "_geo.csv")
    result.drop(columns=["event_ts"]).to_csv(out_path, index=False)
    print(f"[geotag] rows: {len(result)} -> {out_path}")


if __name__ == "__main__":
    main()
