"""Streamlit demo dashboard: dashcam clip (+ GPS CSV) -> potholes on a map.

Run from the repo root:
    venv\\Scripts\\streamlit.exe run app\\streamlit_app.py

The pipeline lives in src/ — this file is UI + orchestration only. All the
work is in run_pipeline(), which tests/test_app.py drives headlessly with a
stub model, so the demo logic is testable without a browser, a GPU, or a real
clip. Streamlit is only touched inside main() and the helpers it calls.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Streamlit executes this file from app/, so the repo root has to be importable
# before `from src import ...` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, detect, geotag, map_builder, severity, tracker

VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv")
SEVERITY_ORDER = ["small", "medium", "large"]


@dataclass
class PipelineResult:
    """Everything one dashboard run produced; the paths are what the UI shows."""

    log: pd.DataFrame
    events: pd.DataFrame
    geotagged: pd.DataFrame | None
    annotated_video: Path | None
    map_html: Path | None
    untagged: int
    stats: dict[str, int]
    out_dir: Path


def run_pipeline(
    video: Path,
    gps_csv: Path | None = None,
    weights: Path = config.DETECT_WEIGHTS,
    conf: float = config.CONF_THRESHOLD,
    frame_skip: int = config.FRAME_SKIP,
    start_time: float | None = None,
    out_dir: Path = config.OUTPUTS_DIR / "app",
    model: Any | None = None,
) -> PipelineResult:
    """detect -> tracker -> severity -> geotag -> map, returning every artifact.

    `model` is injectable so the test can run the whole chain with a stub
    instead of loading weights (same trick as detect.py's contract test).
    The geotag and map steps are skipped when no GPS CSV is given: inventing
    coordinates would make a pretty demo and a false result.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. YOLOv8 + ByteTrack -> one row per detection
    if model is None:
        model = detect.load_model(weights)
    log, annotated = detect.run_video(
        model, video, conf=conf, frame_skip=frame_skip, save_video=True
    )
    log.to_csv(out_dir / "track_log.csv", index=False)

    # 2. collapse the log into one event per unique pothole
    events, stats = tracker.build_events(
        log, config.MIN_TRACK_FRAMES, config.MERGE_MAX_GAP_FRAMES, config.MERGE_IOU
    )
    events.to_csv(out_dir / "pothole_events.csv", index=False)

    # 3. severity: frame size comes from the clip itself, because an uploaded
    #    video is not in data/sample_videos/ where severity.py would look.
    frame_size = severity.frame_size_for(video)
    with_severity = severity.add_severity(events, frame_size=frame_size)
    with_severity.to_csv(out_dir / "pothole_events_severity.csv", index=False)

    # 4./5. GPS join + map, only when a GPS log exists and there is something
    #       to place on the map.
    geotagged: pd.DataFrame | None = None
    map_html: Path | None = None
    untagged = 0
    if gps_csv is not None and not with_severity.empty:
        fps = detect.video_meta(video)[0]
        geotagged, untagged = geotag.geotag_events(
            with_severity, gps_csv, start_time=start_time, fps=fps
        )
        geotagged.drop(columns=["event_ts"]).to_csv(out_dir / "pothole_events_geo.csv", index=False)
        fmap = map_builder.build_map(geotagged, heatmap=True)
        map_html = out_dir / "map.html"
        fmap.save(str(map_html))

    return PipelineResult(
        log=log,
        events=with_severity,
        geotagged=geotagged,
        annotated_video=annotated,
        map_html=map_html,
        untagged=untagged,
        stats=stats,
        out_dir=out_dir,
    )


def severity_counts(events: pd.DataFrame) -> pd.DataFrame:
    """small/medium/large counts in a fixed order, so the chart never reorders."""
    counts = events["severity"].value_counts().reindex(SEVERITY_ORDER, fill_value=0)
    return counts.rename_axis("severity").reset_index(name="potholes")


def available(pattern: str, folder: Path) -> list[Path]:
    """Files matching `pattern` in `folder` (empty when the folder is missing)."""
    return sorted(folder.glob(pattern)) if folder.exists() else []


def save_upload(upload: Any, out_dir: Path) -> Path:
    """Persist an uploaded clip so OpenCV can read it and every stage can
    re-open it by path (severity and geotag both need the file, not bytes)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / Path(upload.name).name
    path.write_bytes(upload.getbuffer())
    return path


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
            upload = st.file_uploader("Video file", type=[suffix.lstrip(".") for suffix in VIDEO_SUFFIXES])
            if upload is not None:
                video_path = save_upload(upload, out_dir)
                st.caption(f"saved to `{video_path}`")

        gps_name = st.selectbox(
            "GPS log",
            ["(none)"] + [path.name for path in gps_logs],
            help="A CSV with timestamp_sec, latitude, longitude — one row per second.",
        )
        gps_csv = config.GPS_LOGS_DIR / gps_name if gps_name != "(none)" else None

        if weights_files:
            names = [path.name for path in weights_files]
            default = names.index("best.pt") if "best.pt" in names else 0
            weights = config.MODELS_DIR / st.selectbox("Weights", names, index=default)
        else:
            weights = config.DETECT_WEIGHTS
            st.error("No `*.pt` in `models/` — fine-tune on Colab first.")

        st.header("Settings")
        conf = st.slider("Confidence cutoff", 0.05, 0.95, config.CONF_THRESHOLD, 0.05)
        frame_skip = st.slider("Frame skip", 1, 5, config.FRAME_SKIP, help="Higher = faster on CPU.")
        start_text = st.text_input(
            "Video start time (ISO 8601, optional)",
            placeholder="2026-10-28 10:00:00+05:30",
            help="Needed to sync frame numbers with GPS time. Blank → parsed from a "
            "dashcam-style filename (YYYYMMDD_HHMMSS).",
        )

        run = st.button("Run pipeline", type="primary", disabled=video_path is None or not weights_files)

    if run and video_path is not None:
        try:
            start_time = geotag.parse_start_time(start_text) if start_text.strip() else None
        except Exception:  # noqa: BLE001 - bad user input must not kill the page
            st.error("Could not parse the start time — use e.g. `2026-10-28 10:00:00+05:30`.")
            start_time = None
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
                    )
                    st.session_state["video_name"] = video_path.name
                except RuntimeError as exc:  # pipeline says exactly what is wrong
                    st.error(str(exc))
                    st.session_state.pop("result", None)

    result: PipelineResult | None = st.session_state.get("result")
    if result is None:
        st.info("Choose a clip in the sidebar and press **Run pipeline**.")
        st.markdown(
            f"""
            **Waiting for inputs**

            | Folder | Found |
            | --- | --- |
            | `data/sample_videos/` | {len(clips)} clip(s) |
            | `data/gps_logs/` | {len(gps_logs)} GPS CSV(s) |
            | `models/` | {len(weights_files)} weight file(s) |

            Without a GPS CSV the pipeline still detects, tracks and grades
            severity — it just cannot place potholes on the map.
            """
        )
        return

    video_name = st.session_state.get("video_name", "clip")
    st.subheader(f"Results — {video_name}")
    top = st.columns(4)
    top[0].metric("Unique potholes", len(result.events))
    top[1].metric("Geotagged", 0 if result.geotagged is None else int(result.geotagged["latitude"].notna().sum()))
    top[2].metric("Track fragments in", result.stats["tracks_in"])
    top[3].metric("Short tracks dropped", result.stats["dropped_short"])

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

    st.subheader("Downloads")
    downloads = st.columns(3)
    for column, name in zip(downloads, ["pothole_events.csv", "pothole_events_severity.csv", "pothole_events_geo.csv"]):
        path = result.out_dir / name
        if path.exists():
            column.download_button(f"⬇ {name}", path.read_bytes(), file_name=name, mime="text/csv")
    if result.map_html is not None and result.map_html.exists():
        st.caption(f"Map saved to `{result.map_html}` — open it in a browser to share it.")


if __name__ == "__main__":
    main()
