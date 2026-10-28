"""Streamlit demo dashboard: dashcam clip (+ GPS CSV) -> potholes on a map.

Run from the repo root:
    venv\\Scripts\\streamlit.exe run app\\streamlit_app.py

The pipeline itself lives in src/pipeline.py — this file is UI + orchestration
only. All non-UI logic sits in plain functions (run_pipeline, build_demo_result,
severity_counts, available, save_upload) that tests/test_app.py drives
headlessly, so the demo is testable without a browser, a GPU, or a real clip.

Heavy imports stay lazy: torch/ultralytics are imported only inside
src.detect.load_model(), and streamlit's component renderer only inside main().
A machine without torch can still render the page and the demo mode.
"""

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# Streamlit executes this file from app/, so the repo root has to be importable
# before `from src import ...` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, geotag, map_builder, pipeline, severity

VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv")
SEVERITY_ORDER = ["small", "medium", "large"]

# --- Demo mode: synthetic events, no video, no model, no real GPS -----------
# Frame is 1280x720 = 921 600 px^2, so the config thresholds split the boxes
# into 2 small / 2 medium / 2 large. Coordinates come from the simulated
# route: demo mode is a UI walkthrough, never a measurement.
DEMO_FRAME_SIZE = (1280, 720)
DEMO_FPS = 30.0
DEMO_SOURCE = "demo_drive.mp4"
DEMO_SECONDS = 20.0
DEMO_EVENTS = [
    # (event_id, frame, x1, y1, x2, y2) — 40x40 small, 90x90 medium, 160x160 large
    (1, 45, 300, 500, 340, 540),
    (2, 120, 700, 520, 740, 560),
    (3, 210, 500, 480, 590, 570),
    (4, 300, 900, 470, 990, 560),
    (5, 390, 620, 430, 780, 590),
    (6, 480, 200, 420, 360, 580),
]


def run_pipeline(
    video: Path,
    gps_csv: Path | None = None,
    weights: Path = config.DETECT_WEIGHTS,
    conf: float = config.CONF_THRESHOLD,
    frame_skip: int = config.FRAME_SKIP,
    start_time: float | None = None,
    out_dir: Path = config.OUTPUTS_DIR / "app",
    model: Any | None = None,
    min_frames: int = config.MIN_TRACK_FRAMES,
    max_gap: int = config.MERGE_MAX_GAP_FRAMES,
    merge_iou: float = config.MERGE_IOU,
    save_video: bool = True,
) -> pipeline.PipelineResult:
    """detect -> tracker -> severity -> geotag -> map for one uploaded clip.

    A thin wrapper over src.pipeline.run_pipeline so the dashboard and the CLI
    can never drift apart. `model` is injectable for tests (stub instead of
    weights). Simulation is OFF here: without a GPS CSV the dashboard shows no
    map, because inventing coordinates would make a pretty demo and a false
    result.
    """
    return pipeline.run_pipeline(
        video,
        gps_csv=gps_csv,
        weights=weights,
        conf=conf,
        frame_skip=frame_skip,
        min_frames=min_frames,
        max_gap=max_gap,
        merge_iou=merge_iou,
        start_time=start_time,
        out_dir=out_dir,
        prefix="",  # fixed names: track_log.csv, pothole_events.csv, map.html
        save_video=save_video,
        simulate_gps=False,
        model=model,
    )


def demo_events() -> pd.DataFrame:
    """The synthetic events behind demo mode, as tracker.py would emit them."""
    return pd.DataFrame(
        [
            {
                "event_id": event_id,
                "frame": frame,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "conf": 0.9,
                "source": DEMO_SOURCE,
                "first_frame": frame,
                "last_frame": frame + 8,
            }
            for event_id, frame, x1, y1, x2, y2 in DEMO_EVENTS
        ]
    )


def build_demo_result(out_dir: Path = config.OUTPUTS_DIR / "demo") -> pipeline.PipelineResult:
    """Demo mode: severity -> geotag -> map on fake events.

    Needs no video, no weights and no real GPS, so the whole UI (table, chart,
    map, downloads) can be demonstrated on any machine. The simulated route is
    written next to the outputs and reported as gps_source="simulated".
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    events = demo_events()
    events_path = out_dir / "demo_pothole_events.csv"
    events.to_csv(events_path, index=False)
    paths["events"] = events_path

    with_severity = severity.add_severity(events, frame_size=DEMO_FRAME_SIZE)
    severity_path = out_dir / "demo_pothole_events_severity.csv"
    with_severity.to_csv(severity_path, index=False)
    paths["severity"] = severity_path

    start_epoch = geotag.parse_start_time(config.SIM_START_ISO)
    route_path = out_dir / "demo_simulated_route.csv"
    pipeline.simulated_route(start_epoch, seconds=DEMO_SECONDS).to_csv(route_path, index=False)
    paths["simulated_route"] = route_path

    geotagged, untagged = geotag.geotag_events(
        with_severity, route_path, start_time=start_epoch, fps=DEMO_FPS
    )
    geo_path = out_dir / "demo_pothole_events_geo.csv"
    geotagged.drop(columns=["event_ts"]).to_csv(geo_path, index=False)
    paths["geotag"] = geo_path

    fmap = map_builder.build_map(geotagged, heatmap=True)
    map_html = out_dir / "demo_map.html"
    fmap.save(str(map_html))
    paths["map"] = map_html

    return pipeline.PipelineResult(
        log=pd.DataFrame(columns=["frame", "track_id", "x1", "y1", "x2", "y2", "conf", "source"]),
        events=with_severity,
        geotagged=geotagged,
        annotated_video=None,
        map_html=map_html,
        untagged=untagged,
        stats={"tracks_in": 0, "dropped_short": 0, "events_out": len(with_severity)},
        out_dir=out_dir,
        gps_source="simulated",
        paths=paths,
    )


def severity_counts(events: pd.DataFrame) -> pd.DataFrame:
    """small/medium/large counts in a fixed order, so the chart never reorders."""
    counts = events["severity"].value_counts().reindex(SEVERITY_ORDER, fill_value=0)
    return counts.rename_axis("severity").reset_index(name="potholes")


def available(pattern: str, folder: Path) -> list[Path]:
    """Files matching `pattern` in `folder` (empty when the folder is missing)."""
    return sorted(folder.glob(pattern)) if folder.exists() else []


def save_upload(upload: Any, out_dir: Path) -> Path:
    """Persist an uploaded file so OpenCV/pandas can read it and every stage can
    re-open it by path (severity and geotag both need the file, not bytes)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / Path(upload.name).name
    path.write_bytes(upload.getbuffer())
    return path


def render_result(result: pipeline.PipelineResult, caption: str | None = None) -> None:
    """Draw metrics, chart, map, clip, table and downloads for one run."""
    import streamlit.components.v1 as components  # lazy: only the map view needs it

    st.subheader(f"Results — {caption or 'clip'}")
    top = st.columns(4)
    top[0].metric("Unique potholes", len(result.events))
    top[1].metric(
        "Geotagged",
        0 if result.geotagged is None else int(result.geotagged["latitude"].notna().sum()),
    )
    top[2].metric("Track fragments in", result.stats["tracks_in"])
    top[3].metric("Short tracks dropped", result.stats["dropped_short"])

    if result.gps_source == "simulated":
        st.warning(
            "GPS source: **simulated route** — these coordinates are not where any "
            "real clip was filmed. Use a real GPS CSV for reportable results."
        )

    if result.events.empty:
        st.warning(
            "No potholes survived tracking. Check the confidence cutoff, or that the "
            "weights actually know a `pothole` class."
        )
        return

    st.subheader("Severity")
    st.bar_chart(severity_counts(result.events), x="severity", y="potholes")

    if result.map_html is not None and result.geotagged is not None:
        st.subheader("Map")
        if result.untagged:
            st.caption(
                f"{result.untagged} event(s) had no GPS fix within "
                f"{config.GPS_TOLERANCE_SEC}s and are not on the map."
            )
        components.html(result.map_html.read_text(encoding="utf-8"), height=600, scrolling=True)
    else:
        st.info("No map: add a GPS CSV in the sidebar to place these potholes geographically.")

    if result.annotated_video is not None and result.annotated_video.exists():
        st.subheader("Annotated clip")
        st.video(str(result.annotated_video))

    st.subheader("Pothole events")
    table = result.geotagged if result.geotagged is not None else result.events
    st.dataframe(table, use_container_width=True)

    if result.paths:
        st.subheader("Downloads")
        columns: list = []
        for index, path in enumerate(result.paths.values()):
            if not path.exists():
                continue
            if index % 3 == 0 or not columns:
                columns = st.columns(3)
            columns[index % 3].download_button(
                f"⬇ {path.name}", path.read_bytes(), file_name=path.name
            )


def main() -> None:
    """Render the dashboard."""
    st.set_page_config(page_title="Pothole detection & mapping", page_icon="🕳️", layout="wide")
    st.title("Road pothole detection & geo-tagged mapping")
    st.caption(
        "YOLOv8 + ByteTrack → one event per pothole → severity → GPS → Folium map. "
        "Every stage reads/writes CSVs, so a run can be re-inspected afterwards."
    )

    clips = available("*.mp4", config.SAMPLE_VIDEOS_DIR)
    gps_logs = available("*.csv", config.GPS_LOGS_DIR)
    weights_files = available("*.pt", config.MODELS_DIR)
    out_dir = config.OUTPUTS_DIR / "app"

    video_path: Path | None = None
    gps_csv: Path | None = None
    with st.sidebar:
        st.header("Inputs")
        source = st.radio("Video source", ["Sample clip", "Upload"], horizontal=True)
        if source == "Sample clip":
            if clips:
                video_path = config.SAMPLE_VIDEOS_DIR / st.selectbox(
                    "Clip", [path.name for path in clips]
                )
            else:
                st.warning("No clips in `data/sample_videos/` yet — upload one instead.")
        else:
            upload = st.file_uploader(
                "Video file", type=[suffix.lstrip(".") for suffix in VIDEO_SUFFIXES]
            )
            if upload is not None:
                video_path = save_upload(upload, out_dir)
                st.caption(f"saved to `{video_path}`")

        gps_choice = st.selectbox(
            "GPS log (from data/gps_logs/)",
            ["(none)"] + [path.name for path in gps_logs],
            help="A CSV with timestamp_sec, latitude, longitude — one row per second.",
        )
        if gps_choice != "(none)":
            gps_csv = config.GPS_LOGS_DIR / gps_choice
        gps_upload = st.file_uploader("…or upload a GPS CSV", type=["csv"])
        if gps_upload is not None:
            gps_csv = save_upload(gps_upload, out_dir)
            st.caption(f"using uploaded `{gps_csv.name}`")

        if weights_files:
            names = [path.name for path in weights_files]
            default = names.index("best.pt") if "best.pt" in names else 0
            weights = config.MODELS_DIR / st.selectbox("Weights", names, index=default)
        else:
            weights = config.DETECT_WEIGHTS
            st.error("No `*.pt` in `models/` — fine-tune on Colab first, or use demo mode.")

        st.header("Detection settings")
        conf = st.slider("Confidence cutoff", 0.05, 0.95, config.CONF_THRESHOLD, 0.05)
        frame_skip = st.slider(
            "Frame skip", 1, 5, config.FRAME_SKIP, help="Higher = faster on CPU."
        )
        start_text = st.text_input(
            "Video start time (ISO 8601, optional)",
            placeholder="2026-10-28 10:00:00+05:30",
            help="Needed to sync frame numbers with GPS time. Blank → parsed from a "
            "dashcam-style filename (YYYYMMDD_HHMMSS).",
        )

        st.header("Tracker settings")
        min_frames = st.slider(
            "Min frames per pothole",
            1,
            10,
            config.MIN_TRACK_FRAMES,
            help="Merged tracks seen in fewer distinct frames are dropped as noise.",
        )
        max_gap = st.slider(
            "Merge gap (frames)",
            1,
            30,
            config.MERGE_MAX_GAP_FRAMES,
            help="How far apart two fragments may be and still be one pothole.",
        )
        merge_iou = st.slider(
            "Merge IoU", 0.1, 0.9, config.MERGE_IOU, 0.05, help="Box overlap needed to merge."
        )

        run = st.button(
            "Run pipeline", type="primary", disabled=video_path is None or not weights_files
        )
        demo = st.button("Load demo data (no model needed)")

    if demo:
        st.session_state["result"] = build_demo_result()
        st.session_state["video_name"] = "demo mode (fake events, simulated route)"

    if run and video_path is not None:
        try:
            start_time = geotag.parse_start_time(start_text) if start_text.strip() else None
        except Exception:  # noqa: BLE001 - bad user input must not kill the page
            st.error("Could not parse the start time — use e.g. `2026-10-28 10:00:00+05:30`.")
        else:
            with st.spinner("Detecting, tracking, estimating severity, geotagging…"):
                try:
                    st.session_state["result"] = run_pipeline(
                        video_path,
                        gps_csv=gps_csv,
                        weights=weights,
                        conf=conf,
                        frame_skip=frame_skip,
                        start_time=start_time,
                        out_dir=out_dir,
                        min_frames=min_frames,
                        max_gap=max_gap,
                        merge_iou=merge_iou,
                    )
                    st.session_state["video_name"] = video_path.name
                except RuntimeError as exc:  # pipeline says exactly what is wrong
                    st.error(str(exc))
                    st.session_state.pop("result", None)

    result: pipeline.PipelineResult | None = st.session_state.get("result")
    if result is None:
        st.info("Choose a clip in the sidebar and press **Run pipeline** — or try **demo mode**.")
        st.markdown(
            f"""
            **Waiting for inputs**

            | Folder | Found |
            | --- | --- |
            | `data/sample_videos/` | {len(clips)} clip(s) |
            | `data/gps_logs/` | {len(gps_logs)} GPS CSV(s) |
            | `models/` | {len(weights_files)} weight file(s) |

            Without a GPS CSV the pipeline still detects, tracks and grades
            severity — it just cannot place potholes on the map. **Demo mode**
            loads fake events plus a simulated route, so the whole dashboard
            works with no model and no clip.
            """
        )
        return

    render_result(result, st.session_state.get("video_name"))


if __name__ == "__main__":
    main()
