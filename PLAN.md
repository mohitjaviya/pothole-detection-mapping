# PLAN.md — Road Pothole Detection & Geo-Tagged Road Quality Mapping

Approved plan for the SEM 7 Computer Vision project. Progress checklist lives
in README.md; agent working rules in AGENTS.md.

## Pipeline

```
dashcam video + GPS CSV
  -> src/detect.py        (YOLOv8 + ByteTrack -> raw per-frame track log)
  -> src/tracker.py       (one event per unique pothole)
  -> src/severity.py      (bbox-area fraction -> small/medium/large)
  -> src/geotag.py        (frame timestamp -> lat/lon)
  -> src/map_builder.py   (Folium map + CSV report)
  -> app/streamlit_app.py (demo dashboard)
```

Each stage reads and writes CSVs, so the back half (severity -> map) can be
built, tested, and demoed repeatedly without a GPU, while only
`detect.py`/`train.py` touch the model.

## Folder structure

```
pothole-detection-mapping/
├── data/                  # never committed
│   ├── raw/               # pothole image dataset (YOLO .txt format)
│   ├── processed/         # extracted frames, cleaned splits
│   ├── sample_videos/     # short dashcam clips driving the demo
│   └── gps_logs/          # one CSV per video: timestamp_sec, latitude, longitude
├── src/                   # one job per module, each with a main() CLI
│   ├── config.py          # shared paths, thresholds, conf, frame skip
│   ├── baseline.py        # classical OpenCV baseline (comparison point)
│   ├── train.py           # dataset prep + fine-tuning (runs on Colab via notebook)
│   ├── detect.py          # frame extraction + model.track() -> raw track log
│   ├── tracker.py         # track log -> one event per unique pothole
│   ├── severity.py        # bbox-area fraction -> small/medium/large
│   ├── geotag.py          # frame timestamp -> GPS lat/lon join
│   ├── map_builder.py     # Folium map + CSV report (no model dependency)
│   └── evaluate.py        # baseline vs YOLOv8n vs YOLOv8s; day/night/wet/blur
├── configs/               # static YAML (bytetrack.yaml); data.yaml is generated
│                          #   per run into data/processed/ivcnz/ (never committed)
├── app/streamlit_app.py   # demo dashboard
├── notebooks/train_colab.ipynb  # Colab entry point: dataset, train, export best.pt
├── models/best.pt         # trained weights (never committed)
├── outputs/               # annotated videos, maps, per-run CSVs (never committed)
├── reports/               # evaluation figures + metric tables (committed)
├── requirements.txt
├── README.md              # overview + progress checklist
└── AGENTS.md              # agent working rules
```

## Build order and milestones

1. [x] **Scaffold** — folders, `.gitignore`, `requirements.txt`, README.
2. [ ] **Classical OpenCV baseline** (`baseline.py`, `config.py`) — done when a
       results table in `reports/` shows why fine-tuning is needed.
3. [ ] **Training** (`src/train.py`, `notebooks/train_colab.ipynb`) — done when
       `models/best.pt` exists locally and val mAP is reported.
4. [x] **GPS source decision** — phone GPS logger CSV is primary, simulated
       route is the fallback/demo. Done: both timestamp forms (epoch + ISO)
       and both start-time forms (filename + `--start-time`) are supported,
       and `tests/make_fake_data.py` provides a simulated route.
5. [x] **detect.py** — YOLOv8 + ByteTrack on a sample video (`persist=True`),
       CPU-friendly (yolov8n, frame skip). Done: `track_log.csv` columns match
       `tracker.py` and frame indices are real video frames — asserted by
       `tests/test_detect.py` with a stub model. A real-clip run (stable IDs
       on actual footage) needs `ultralytics` + fine-tuned weights.
6. [ ] **tracker.py** — collapse the track log into unique pothole events,
       tolerating ID flicker. Done when the event count on a short clip
       matches a manual count.
7. [ ] **severity.py** — thresholds calibrated visually in a notebook. Done
       when every event carries small/medium/large and spot-checked crops
       agree with the labels.
8. [ ] **geotag.py** — frame N timestamp = video start + N/fps -> nearest GPS
       row within tolerance. Done when `pothole_events_geo.csv` has plausible
       lat/lon values.
9. [ ] **map_builder.py** — severity-colored markers, optional heatmap,
       snapshot popups. Done when `outputs/map.html` opens with potholes in
       sensible positions.
10. [ ] **evaluate.py** — detection metrics (mAP50, precision/recall) plus
        system-level count accuracy, broken down by condition. Done when
        figures land in `reports/`.
11. [ ] **Streamlit dashboard** — wire the pipeline end to end. Done when the
        demo runs video -> map in the browser.

## Decisions and risks

1. **GPS source (step 4)** — many dashcams store GPS in proprietary formats or
   not at all. Decide between parsing dashcam metadata, a phone GPS logger
   app (exports CSV), or a simulated route for the demo. Do not discover this
   late.
2. **Severity is a proxy** — bbox pixel area grows for near-bottom-of-frame
   boxes (closer to camera). Fine for coursework; calibrate thresholds
   visually and state the limitation in the report.
3. **Double counting** — driving the same road twice counts each pothole
   twice. Stretch goal: merge repeat passes via coarse geo-cell hashing;
   otherwise document as a known limitation.
4. **Sample video size** — keep clips under ~30 s and compressed; GitHub
   rejects files over 100 MB, and the CPU demo must run in minutes.
