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

or in one command: src/pipeline.py <video> [--gps CSV] [--track-log CSV]
```

Each stage reads and writes CSVs, so the back half (severity -> map) can be
built, tested, and demoed repeatedly without a GPU, while only
`detect.py`/`train.py` touch the model. `robustness.py` re-runs detection on
degraded copies of a clip for the report's robustness section.

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
│   ├── pipeline.py        # one command: detect -> tracker -> severity -> geotag -> map
│   ├── evaluate.py        # baseline vs YOLOv8n vs YOLOv8s; day/night/wet/blur
│   └── robustness.py      # dim/blur/noise/motion sweep -> reports/robustness.csv
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
2. [x] **Classical OpenCV baseline** (`baseline.py`, `config.py`) — done when a
       results table in `reports/` shows why fine-tuning is needed.
       Done: `reports/detection_metrics.csv` on the 125-image test split —
       baseline precision 0.058 / recall 0.029 / F1 0.039 vs YOLOv8n
       precision 0.741 / recall 0.628 / mAP50 0.690.
3. [x] **Training** (`src/train.py`, `notebooks/train_colab.ipynb`) — done when
       `models/best.pt` exists locally and val mAP is reported.
       Done: `models/best.pt` = fine-tuned yolov8n (`names: {0: 'pothole'}`),
       test split (125 images / 346 instances): P 0.741, R 0.628, mAP50 0.690,
       mAP50-95 0.367 (`reports/detection_metrics.csv`). `reports/train_metrics.csv`
       from the Colab run is still to be copied over.
4. [x] **GPS source decision** — phone GPS logger CSV is primary, simulated
       route is the fallback/demo. Done: both timestamp forms (epoch + ISO)
       and both start-time forms (filename + `--start-time`) are supported,
       and `tests/make_fake_data.py` provides a simulated route.
5. [x] **detect.py** — YOLOv8 + ByteTrack on a sample video (`persist=True`),
       CPU-friendly (yolov8n, frame skip). Done: `track_log.csv` columns match
       `tracker.py` and frame indices are real video frames — asserted by
       `tests/test_detect.py` with a stub model, and run for real on
       `data/sample_videos/` (see the sample-clip table in README).
6. [x] **tracker.py** — collapse the track log into unique pothole events,
       tolerating ID flicker. Done: flicker/split merges and the short-track
       drop are asserted by `tests/test_tracker.py`; `tests/test_detect.py`
       proves the detect -> tracker hand-off. Event-count-vs-manual-count on a
       real pothole clip still needs that clip.
7. [x] **severity.py** — thresholds calibrated visually in a notebook. Done:
       every event carries small/medium/large, empty inputs handled, and
       `tests/test_chain.py` asserts an engineered 3/3/2 distribution.
       Visual calibration on real footage is still to do.
8. [x] **geotag.py** — frame N timestamp = video start + N/fps -> nearest GPS
       row within tolerance. Done: epoch + ISO timestamps, tolerance handling,
       untagged events counted (not dropped) — asserted by `tests/test_chain.py`
       (7 tagged at exact route positions, 1 deliberately untagged).
9. [x] **map_builder.py** — severity-colored markers, optional heatmap,
       snapshot popups. Done: `outputs/map.html` opens with severity colours,
       heatmap layer and per-pothole popups (`tests/test_chain.py` counts markers).
10. [x] **evaluate.py** — detection metrics (mAP50, precision/recall) plus
        system-level count accuracy, broken down by condition. Done when
        figures land in `reports/`.
        Detection half DONE: `reports/detection_metrics.csv`,
        `eval_comparison.csv` + `eval_comparison.png`, `eval_summary.md`
        (baseline vs yolov8n on the test split; yolov8s still needs its
        weights). `--methods name=path` scores pre-computed prediction files
        with no torch. Counting half still needs clips + `data/eval_manifest.csv`.
11. [x] **Streamlit dashboard** — wire the pipeline end to end. Done: the demo
        runs video -> map in the browser, with detection sliders, tracker
        sliders, video + GPS upload, and a demo mode that needs no model or
        clip. Verified headless (`tests/test_app.py`) and the server boots
        (health `ok`, page `200`).
12. [x] **pipeline.py** — one command for detect -> tracker -> severity ->
        geotag -> map, with `--track-log` (skip detection) and a simulated
        route when no GPS CSV is given. Asserted by `tests/test_pipeline.py`.
13. [x] **robustness.py** — dim / blur / noise / motion-blur copies of one
        clip -> `reports/robustness.csv` (detections, unique tracks, retention).
        Asserted with a stub model by `tests/test_robustness.py`; the committed
        table is from the wiring clip and must be re-run on real footage.

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
