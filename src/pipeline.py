"""One command for the whole pipeline: dashcam clip -> pothole map.

    detect.py -> tracker.py -> severity.py -> geotag.py -> map_builder.py

Every stage's CSV lands in one output folder (`--out-dir`, default
`outputs/`), so a run can be re-inspected stage by stage afterwards. The
naming is `<prefix>track_log.csv`, `<prefix>pothole_events.csv`, ... where the
CLI sets the prefix to the clip stem (two clips never overwrite each other).

Two shortcuts keep the chain runnable without a model or without real GPS:

- `--track-log PATH` skips detect.py completely — no torch, no weights. The
  chain starts from an existing track log, which is how the whole pipeline is
  tested on a CPU-only machine.
- no `--gps` means a SIMULATED straight route is generated from
  `config.ROUTE_*` and written next to the outputs (`simulated_route.csv`).
  Its coordinates are NOT where the clip was filmed — the map is a wiring
  demonstration. The run prints a warning and the result records
  `gps_source="simulated"` so the report can never quietly present it as real.

CLI (repo root):
    python -m src.pipeline data/sample_videos/clip.mp4 \
        --gps data/gps_logs/drive.csv --start-time "2026-10-28 10:00:00+05:30"
    python -m src.pipeline data/sample_videos/clip.mp4 --track-log outputs/track_log_clip.csv
    python -m src.pipeline data/sample_videos/clip.mp4 --frame-skip 5 --no-video
"""

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src import config, detect, geotag, map_builder, severity, tracker


@dataclass
class PipelineResult:
    """Everything one pipeline run produced (tables + paths the UI shows)."""

    log: pd.DataFrame
    events: pd.DataFrame
    geotagged: pd.DataFrame | None
    annotated_video: Path | None
    map_html: Path | None
    untagged: int
    stats: dict[str, int]
    out_dir: Path
    gps_source: str  # "csv" | "simulated" | "none"
    paths: dict[str, Path] = field(default_factory=dict)


def simulated_route(
    start_epoch: float,
    seconds: float,
    speed_kmh: float = config.ROUTE_SPEED_KMH,
) -> pd.DataFrame:
    """A straight northbound 1 Hz route from config.ROUTE_START_LAT/LON.

    Only used when no real GPS CSV exists: it gives geotag.py something to
    join against so the map stage can be exercised. The numbers are
    geographically meaningless (see the module docstring).
    """
    meters_per_sec = speed_kmh / 3.6
    degrees_per_sec = meters_per_sec / 111_320.0  # ~111.32 km per degree of latitude
    steps = max(config.ROUTE_SECONDS, int(math.ceil(seconds)) + 2)
    return pd.DataFrame(
        {
            "timestamp_sec": [start_epoch + step for step in range(steps)],
            "latitude": [config.ROUTE_START_LAT + step * degrees_per_sec for step in range(steps)],
            "longitude": [config.ROUTE_START_LON] * steps,
        }
    )


def clip_fps(video: Path) -> float:
    """fps from the container, with the config fallback when it carries none."""
    try:
        return detect.video_meta(video)[0]
    except RuntimeError:
        return config.PIPELINE_VIDEO_FPS_FALLBACK


def resolve_start_time(video: Path, start_time: float | None) -> float | None:
    """Explicit --start-time wins; otherwise a dashcam-style filename; otherwise
    None (the caller decides what that means for a real vs simulated route)."""
    if start_time is not None:
        return start_time
    return geotag.start_time_from_filename(video.name)


def run_pipeline(
    video: Path,
    gps_csv: Path | None = None,
    track_log: Path | None = None,
    weights: Path = config.DETECT_WEIGHTS,
    conf: float = config.CONF_THRESHOLD,
    frame_skip: int = config.FRAME_SKIP,
    min_frames: int = config.MIN_TRACK_FRAMES,
    max_gap: int = config.MERGE_MAX_GAP_FRAMES,
    merge_iou: float = config.MERGE_IOU,
    start_time: float | None = None,
    frame_size: tuple[int, int] | None = None,
    out_dir: Path = config.PIPELINE_OUT_DIR,
    prefix: str = "",
    save_video: bool = config.WRITE_ANNOTATED_VIDEO,
    heatmap: bool = config.PIPELINE_MAP_HEATMAP,
    simulate_gps: bool = True,
    model: Any | None = None,
) -> PipelineResult:
    """Run the five stages on one clip and return every artifact.

    `model` and `track_log` are injectable so the whole chain can be tested
    with a stub model or with no model at all (see tests/test_pipeline.py).
    `simulate_gps=False` refuses to invent coordinates: no GPS CSV then means
    no geotag and no map (the dashboard's rule).
    """
    if not video.exists():
        raise RuntimeError(f"video not found: {video}")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # 1. detections: YOLOv8 + ByteTrack, or an existing track log
    if track_log is not None:
        log = pd.read_csv(track_log)
        missing = tracker.REQUIRED_COLUMNS - set(log.columns)
        if missing:
            raise RuntimeError(f"track log missing columns: {sorted(missing)}")
        log = log.sort_values(["source", "frame", "track_id"]).reset_index(drop=True)
        annotated_video: Path | None = None
        print(f"[pipeline] reusing track log (detect skipped): {track_log}")
    else:
        if model is None:
            model = detect.load_model(weights)
        log, annotated_video = detect.run_video(
            model, video, conf=conf, frame_skip=frame_skip, save_video=save_video
        )

    log_path = out_dir / f"{prefix}track_log.csv"
    log.to_csv(log_path, index=False)
    paths["track_log"] = log_path

    # 2. one event per unique pothole
    events, stats = tracker.build_events(log, min_frames, max_gap, merge_iou)
    events_path = out_dir / f"{prefix}pothole_events.csv"
    events.to_csv(events_path, index=False)
    paths["events"] = events_path
    print(f"[pipeline] events: {stats['events_out']} (tracks in: {stats['tracks_in']}, "
          f"dropped short: {stats['dropped_short']})")

    # 3. severity from bbox area / frame area. The clip's own size is used
    #    because an arbitrary clip is not in data/sample_videos/.
    size = frame_size if frame_size is not None else severity.frame_size_for(video)
    with_severity = severity.add_severity(events, frame_size=size)
    severity_path = out_dir / f"{prefix}pothole_events_severity.csv"
    with_severity.to_csv(severity_path, index=False)
    paths["severity"] = severity_path
    if not with_severity.empty:
        print(f"[pipeline] severity: {with_severity['severity'].value_counts().to_dict()}")

    # 4./5. GPS join + map. Nothing to place -> nothing to draw (an empty map
    #       would suggest a successful geotag of zero potholes).
    geotagged: pd.DataFrame | None = None
    map_html: Path | None = None
    untagged = 0
    gps_source = "none"
    if with_severity.empty:
        print("[pipeline] no potholes survived tracking — geotag and map skipped")
    else:
        fps = clip_fps(video)
        resolved_start = resolve_start_time(video, start_time)
        if gps_csv is not None:
            if resolved_start is None:
                raise RuntimeError(
                    f"no --start-time and no timestamp in filename {video.name!r} — "
                    "pass the clip's real start time so frames can be matched to GPS"
                )
            gps_source = "csv"
        elif not simulate_gps:
            # Dashboard rule: no GPS CSV means no coordinates at all. A pretty
            # map of invented places is worse than no map.
            print("[pipeline] no GPS CSV — geotag and map skipped (simulation disabled)")
            return PipelineResult(
                log=log,
                events=with_severity,
                geotagged=None,
                annotated_video=annotated_video,
                map_html=None,
                untagged=0,
                stats=stats,
                out_dir=out_dir,
                gps_source="none",
                paths=paths,
            )
        else:
            # Simulated fallback: the clock start only has to be deterministic.
            if resolved_start is None:
                resolved_start = geotag.parse_start_time(config.SIM_START_ISO)
            route = simulated_route(resolved_start, float(log["frame"].max()) / fps if len(log) else 0.0)
            gps_csv = out_dir / f"{prefix}simulated_route.csv"
            route.to_csv(gps_csv, index=False)
            paths["simulated_route"] = gps_csv
            gps_source = "simulated"
            print(f"[pipeline] WARNING: no --gps given — using a SIMULATED route "
                  f"({gps_csv}); the map is a wiring demo, not your location")

        geotagged, untagged = geotag.geotag_events(
            with_severity, gps_csv, start_time=resolved_start, fps=fps
        )
        geo_path = out_dir / f"{prefix}pothole_events_geo.csv"
        geotagged.drop(columns=["event_ts"]).to_csv(geo_path, index=False)
        paths["geotag"] = geo_path

        fmap = map_builder.build_map(geotagged, heatmap=heatmap)
        map_html = out_dir / f"{prefix}map.html"
        fmap.save(str(map_html))
        paths["map"] = map_html
        print(f"[pipeline] map: {map_html}")

    return PipelineResult(
        log=log,
        events=with_severity,
        geotagged=geotagged,
        annotated_video=annotated_video,
        map_html=map_html,
        untagged=untagged,
        stats=stats,
        out_dir=out_dir,
        gps_source=gps_source,
        paths=paths,
    )


def main() -> None:
    """CLI entry: python -m src.pipeline <video> [flags]"""
    parser = argparse.ArgumentParser(
        description="detect -> tracker -> severity -> geotag -> map in one command"
    )
    parser.add_argument("video", type=Path, help="dashcam clip (data/sample_videos/)")
    parser.add_argument("--gps", type=Path, default=None, help="GPS CSV (timestamp_sec, latitude, longitude)")
    parser.add_argument(
        "--track-log",
        type=Path,
        default=None,
        help="reuse an existing track log and skip detection (no model, no torch)",
    )
    parser.add_argument("--weights", type=Path, default=config.DETECT_WEIGHTS)
    parser.add_argument("--conf", type=float, default=config.CONF_THRESHOLD)
    parser.add_argument("--frame-skip", type=int, default=config.FRAME_SKIP)
    parser.add_argument("--min-frames", type=int, default=config.MIN_TRACK_FRAMES)
    parser.add_argument("--max-gap", type=int, default=config.MERGE_MAX_GAP_FRAMES)
    parser.add_argument("--iou", type=float, default=config.MERGE_IOU)
    parser.add_argument(
        "--start-time",
        type=geotag.parse_start_time,
        default=None,
        help='clip start as ISO 8601 ("2026-10-28 10:00:00+05:30"); naive -> Asia/Kolkata',
    )
    parser.add_argument("--out-dir", type=Path, default=config.PIPELINE_OUT_DIR)
    parser.add_argument("--no-video", action="store_true", help="skip the annotated video")
    parser.add_argument("--no-heatmap", action="store_true", help="drop the map heatmap layer")
    parser.add_argument(
        "--no-simulated-gps",
        action="store_true",
        help="without --gps, skip geotag/map instead of generating a simulated route",
    )
    args = parser.parse_args()

    if not args.video.exists():
        raise SystemExit(f"video not found: {args.video}")

    result = run_pipeline(
        args.video,
        gps_csv=args.gps,
        track_log=args.track_log,
        weights=args.weights,
        conf=args.conf,
        frame_skip=args.frame_skip,
        min_frames=args.min_frames,
        max_gap=args.max_gap,
        merge_iou=args.iou,
        start_time=args.start_time,
        out_dir=args.out_dir,
        prefix=f"{args.video.stem}_",
        save_video=not args.no_video,
        heatmap=not args.no_heatmap,
        simulate_gps=not args.no_simulated_gps,
    )

    print(f"[pipeline] GPS source: {result.gps_source}")
    if result.geotagged is not None:
        tagged = int(result.geotagged["latitude"].notna().sum())
        print(f"[pipeline] geotagged: {tagged} / {len(result.geotagged)} (untagged: {result.untagged})")
    for name, path in result.paths.items():
        print(f"[pipeline] {name}: {path}")


if __name__ == "__main__":
    main()
