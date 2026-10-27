# Road Pothole Detection & Geo-Tagged Mapping

## Overview

Semester 7 Computer Vision project. Detects potholes in dashcam video using
YOLOv8, tracks them so each pothole is counted once, estimates severity,
attaches GPS coordinates, and shows them on an interactive map.

## Stack

- Python 3.10+
- Ultralytics YOLOv8 (detection + built-in ByteTrack tracking)
- OpenCV (video I/O, classical baseline)
- pandas, numpy
- Folium (map + heatmap)
- Streamlit (demo dashboard)

## Main flow

```
dashcam video + GPS CSV -> YOLO detection -> tracking (unique IDs)
-> severity (by bbox area) -> match timestamp to GPS -> Folium map + CSV report
```

## Environment rules

- Training happens on Google Colab / Kaggle (GPU). Do NOT assume a local GPU.
- Local code must run on CPU (use yolov8n, resize frames, allow frame skipping).
- Training logic goes in `src/train.py`; `notebooks/train_colab.ipynb` wraps
  it for Colab. Trained weights are downloaded into `models/best.pt`.
- Never commit `data/`, `models/`, `outputs/` or `venv/`.

## Project structure

- `data/` — never committed. `raw/` (pothole image dataset, YOLO format),
  `processed/` (extracted frames), `sample_videos/` (short dashcam clips),
  `gps_logs/` (one CSV per video: `timestamp_sec, latitude, longitude`)
- `src/` — one job per module, each with a `main()` for CLI use:
  - `config.py` — shared paths, thresholds, confidence, frame skip (single source of truth)
  - `baseline.py` — classical OpenCV baseline to compare against YOLOv8
  - `train.py` — dataset prep + fine-tuning logic (executed on Colab via the notebook)
  - `detect.py` — frame extraction + YOLOv8 detection + ByteTrack -> raw track log
  - `tracker.py` — collapses the track log into one record per unique pothole
  - `severity.py` — bbox area as fraction of frame -> small / medium / large
  - `geotag.py` — matches frame timestamps to GPS rows -> attaches lat/lon
  - `map_builder.py` — Folium map (severity-colored markers + heatmap) + CSV report
  - `evaluate.py` — baseline vs YOLOv8n vs YOLOv8s, broken down by day/night/wet/blur
- `configs/` — YAML configs consumed by Ultralytics: `data.yaml` (dataset spec),
  `bytetrack.yaml` (tracker tuning)
- `app/` — `streamlit_app.py` (demo dashboard: video -> map, table, severity chart)
- `notebooks/` — `train_colab.ipynb` (dataset download, run `src/train.py`, export `best.pt`)
- `models/` — `best.pt` (never committed)
- `outputs/` — annotated videos, maps, per-run CSVs (never committed)
- `reports/` — evaluation figures and metric tables for the semester report

## Conventions

- Use `pathlib` for paths and type hints on all functions.
- Keep each module in `src/` focused on one job, with a `main()` for CLI use.
- Config (paths, thresholds, conf, frame skip) lives in `config.py`,
  not hardcoded.
- GPS CSV format: `timestamp_sec, latitude, longitude`
- Severity: small / medium / large from bbox area as a fraction of the frame.
- Write short comments explaining the CV logic (this is a student project
  that will be explained to faculty).

## Build order

1. [x] Folder structure, requirements.txt, README
2. [ ] Classical OpenCV baseline
3. [ ] Colab training notebook (dataset download, train, export best.pt)
4. [ ] GPS source decision (phone GPS logger CSV or simulated route)
5. [ ] detect.py (YOLOv8 + ByteTrack -> raw track log)
6. [ ] tracker.py (unique pothole IDs, one event per pothole)
7. [ ] severity.py
8. [ ] geotag.py
9. [ ] map_builder.py
10. [ ] evaluate.py (baseline vs YOLOv8n vs YOLOv8s, day/night/wet/blur)
11. [ ] Streamlit dashboard

---

Development conventions and agent working rules: see [AGENTS.md](AGENTS.md).
Detailed plan with per-step milestones: see [PLAN.md](PLAN.md).
