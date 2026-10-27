# AGENTS.md

Agent instructions for the Road Pothole Detection & Geo-Tagged Mapping project
(SEM 7 Computer Vision coursework). Project-facing docs live in README.md;
the approved plan with per-step milestones lives in PLAN.md.

## Stack & pipeline (decided)

- Python 3.10+, Ultralytics YOLOv8 (detection + built-in ByteTrack), OpenCV,
  pandas, numpy, Folium, Streamlit.
- Flow: dashcam video + GPS CSV -> detect -> tracker (unique IDs, count each
  pothole once) -> severity (bbox-area fraction -> small/medium/large) ->
  geotag (timestamp -> lat/lon) -> Folium map + CSV report.

## Project layout

- `data/` (raw, processed, sample_videos, gps_logs), `models/`, `outputs/` —
  never committed.
- `src/` — `baseline.py`, `train.py`, `detect.py`, `tracker.py`,
  `severity.py`, `geotag.py`, `map_builder.py`, `evaluate.py`, `config.py`.
- `configs/` — static YAML only (e.g. custom `bytetrack.yaml` tracker tuning).
  `data.yaml` is GENERATED per run by train.py into the dataset root —
  machine-specific, never committed.
- `app/` — `streamlit_app.py`.
- `notebooks/` — `train_colab.ipynb`.
- `reports/` — evaluation figures and metric tables for the semester report
  (committed).

## Environment

- Training runs ONLY on Colab/Kaggle: `src/train.py` holds the training logic
  (dataset prep + fine-tune); `notebooks/train_colab.ipynb` wraps it for the
  Colab GPU and exports `models/best.pt`. Do not assume a local GPU.
- Local execution is CPU-only: use yolov8n, resize frames, support frame
  skipping.
- Never commit `data/`, `models/`, `outputs/`, or `venv/`
  (.gitignore enforces this).

## Conventions (owner-set)

- `pathlib` for paths; type hints on all functions.
- Every `src/` module does exactly one job and has a `main()` for CLI use.
- All tunables (paths, thresholds, conf, frame skip) live in `src/config.py` —
  never hardcoded in modules. Static Ultralytics YAML lives in `configs/`;
  `data.yaml` is generated per run into the dataset root (never commit it).
- GPS CSV columns: `timestamp_sec, latitude, longitude`.
- Severity classes: small / medium / large, from bbox area as a fraction of
  the frame.
- Write short comments explaining the CV logic — this is a student project
  presented to faculty.

## Working agreement with the owner

- Plan first for any new module, then build.
- Make small changes and tell the owner how to test each one.
- Do not install heavy packages without asking.
- After each working step, remind the owner to commit to git.
- Follow the build order in README.md (milestones in PLAN.md); check off
  steps as they complete.

## Gotchas

- `ultralytics` brings its own `torch` dependency — do not add torch to
  requirements.txt; on Colab the CUDA build comes preinstalled.
- Colab/Kaggle sessions are ephemeral — download `best.pt` and any
  metrics/CSVs before the session ends.
- Detections are pixel boxes only; the lat/lon join happens in `geotag.py` by
  matching frame timestamps (video start + N/fps) to GPS rows. Keep mapping
  code free of model dependencies so the back half of the pipeline runs
  without a GPU.
- The GPS source (phone GPS logger CSV or simulated route) is build-order
  step 4 — settle it early; logs must use the `timestamp_sec, latitude,
  longitude` columns.
- `evaluate.py` compares baseline vs YOLOv8n vs YOLOv8s across
  day/night/wet/blur — test clips need condition labels; plan a conditions
  column when building the eval split.
- Re-driving the same road double-counts potholes unless deduped (geo-cell
  hashing is a stretch goal).
