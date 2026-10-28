# Semester report outline

Chapter-by-chapter skeleton for the SEM 7 Computer Vision report, with the file
each section draws its numbers from. Anything marked **[TODO]** needs footage
that does not exist in the repository yet — do not fill it with wiring-clip
numbers.

---

## 1. Introduction

- Problem: road surface damage is under-reported; manual inspection does not
  scale; drivers get no warning.
- Goal: from a dashcam video plus a GPS log, detect potholes, count each one
  once, grade severity, and place them on a map.
- Contributions (all implemented here):
  1. a five-stage CSV pipeline (detect → track → severity → geotag → map) that
     can be run stage-by-stage or in one command;
  2. a tracking-based counting scheme that survives ID flicker;
  3. an evaluation that separates detection quality from counting accuracy;
  4. a synthetic-degradation robustness sweep;
  5. a Streamlit dashboard with a model-free demo mode.
- Scope: single pothole class, monocular video, no depth, no metric scale.

## 2. Related work

- Classical road-damage detection (thresholding, Canny + contours) — the
  baseline implemented in `src/baseline.py`.
- CNN / YOLO-family object detection for road damage.
- Multi-object tracking (ByteTrack) and why detection alone double-counts.
- Crowdsourced road-quality mapping.

## 3. Dataset

- Source: IVCNZ pothole dataset (YOLO format, single class), downloaded by
  `src/train.py`; split 80/10/10 with seed 42 → **994 / 124 / 125** images.
  Numbers from `data/processed/ivcnz/data.yaml` and `reports/detection_metrics.csv`.
- Test split: **125 images, 346 pothole instances**.
- Character of the data: close-up photographs → state the domain gap here and
  refer forward to §8.
- Clip material: public stock clips + a photo-based wiring clip; **[TODO]**
  a dashcam clip with real potholes.
- GPS: phone logger CSV format (`timestamp_sec, latitude, longitude`);
  **[TODO]** one real log.

## 4. Method

### 4.1 Detection — `src/detect.py`
- YOLOv8n fine-tuned on the dataset, `imgsz 640`, confidence 0.25.
- Frame skipping (`FRAME_SKIP`) for CPU; frame indices stay original so timing
  downstream is unaffected.

### 4.2 Tracking and counting — `src/tracker.py`
- ByteTrack (`persist=True`), tracker tuning in `configs/bytetrack.yaml`.
- One event per unique pothole: union-find merge of fragments with
  IoU ≥ `MERGE_IOU` within `MERGE_MAX_GAP_FRAMES`, then drop tracks seen in
  fewer than `MIN_TRACK_FRAMES` distinct frames.
- Report the merge rules as the answer to "why not count every detection".

### 4.3 Severity — `src/severity.py`
- bbox area / frame area → small / medium / large, thresholds in `config.py`.
- State the proxy caveat (see §8.5).

### 4.4 Geotagging — `src/geotag.py`
- Event time = video start + frame / fps; nearest GPS fix within 2 s;
  untagged events counted, not dropped.
- Video start from `--start-time` or a dashcam-style filename.

### 4.5 Mapping — `src/map_builder.py`
- Folium: severity-coloured circle markers + heatmap layer + popups that trace
  a marker back to clip/frame/bbox.

### 4.6 Orchestration — `src/pipeline.py`
- One command; `--track-log` to start from an existing log (no model);
  simulated route only when no GPS CSV is given, and labelled as such.

### 4.7 Dashboard — `app/streamlit_app.py`
- Sidebar inputs (clip/upload, GPS CSV/upload, weights, detection + tracker
  sliders), results (metrics, severity chart, map, annotated clip, table,
  downloads), demo mode.
- Screenshot from demo mode: it needs no model and no clip.

## 5. Experiments

### 5.1 Setup
- CPU-only for inference; training on Colab (GPU).
- IoU 0.5 matching for the classical baseline; `ultralytics val()` for YOLO.
- Split, seeds, confidence, frame skip.

### 5.2 Detector comparison
- Table: `reports/detection_metrics.csv`, `reports/eval_comparison.csv`.
- Figure: `reports/eval_comparison.png`.
- Numbers: baseline P 0.058 / R 0.029 / F1 0.039 vs yolov8n P 0.741 / R 0.628 /
  mAP50 0.690; yolov8s P 0.777 / R 0.581 / mAP50 0.686 (from the Colab run —
  weights not in this working copy).
- Discussion: why the classical baseline collapses (no learned prior, road
  texture, no confidence score → cannot trade precision for recall).

### 5.3 Counting accuracy
- **[TODO]** needs `data/eval_manifest.csv` + `reports/count_metrics.csv` and
  `reports/count_by_condition.csv` (day / night / wet / blur).
- Report MAE, exact-match rate and mean count accuracy per condition, plus the
  per-clip table. Do not report the pipeline's event count as accuracy before
  this exists.

### 5.4 Robustness
- `reports/robustness.csv`: dim 0.862, blur 0.775, noise 0.625, motion 0.512
  detection retention.
- **[TODO]** re-run on a real dashcam clip; the committed table is from the
  photo-based wiring clip.
- Discussion: motion blur dominates; track counts fall faster than detections
  (ID fragmentation).

### 5.5 End-to-end demo
- One-command run: `python -m src.pipeline …`; screenshots of the map and the
  dashboard. **[TODO]** real clip + real GPS log.

## 6. Discussion

- What works: detection quality on the dataset's own domain; the counting
  scheme; the CSV-per-stage design (each stage independently testable, the back
  half needs no GPU).
- What does not: cross-domain generalisation (dashcam), severity calibration,
  duplicate potholes across passes.
- Design decisions worth defending: one event per track instead of per
  detection; count untagged events rather than dropping them; refuse to invent
  coordinates when there is no GPS.

## 7. Future work

- Fine-tune on dashcam frames (transfer learning), larger `imgsz`, small-object
  augmentation.
- Linear interpolation between GPS fixes; fuse with IMU / OBD speed.
- Geo-cell hashing to de-duplicate repeat passes.
- Metric size estimation (depth or known camera geometry) instead of bbox area.
- Edge deployment (Jetson / phone) with a quantised model.

## 8. Limitations

Write these verbatim from `reports/results_summary.md` §Limitations (1–8):
domain gap, no real GPS yet, nearest-fix join without interpolation, double
counting, bbox area as a size proxy, CPU/frame-skip constraints, robustness
measured on the wiring clip, single class.

## 9. Conclusion

- Restate the contributions and the honest state: the pipeline is complete and
  verified end to end; detection quality is measured on the dataset domain;
  dashcam deployment needs footage-specific fine-tuning and a real GPS log.

## Appendices

- **A. Reproducibility:** `requirements.txt`, Python version, commands from
  README ("One command" + "Stage by stage"), and the test list.
- **B. Configuration:** dump of `src/config.py` thresholds with one line of
  justification each.
- **C. File map:** which script writes which file (see README structure and
  `reports/results_summary.md`).
- **D. Test evidence:** output of the seven test scripts (`tests/test_*.py`).

---

### Result file → report section cheat sheet

| File | Feeds |
| --- | --- |
| `reports/detection_metrics.csv` | §5.2 detector table |
| `reports/eval_comparison.csv` / `.png` | §5.2 table + figure |
| `reports/eval_summary.md` | §5.2/§5.3 quick reference |
| `reports/count_metrics.csv` | §5.3 per-clip counting **[TODO]** |
| `reports/count_by_condition.csv` | §5.3 condition table **[TODO]** |
| `reports/eval_counts_by_condition.png` | §5.3 figure **[TODO]** |
| `reports/robustness.csv` | §5.4 robustness table |
| `reports/results_summary.md` | §5 numbers + §8 limitations |
| `outputs/*_map.html` | §4.5, §5.5 screenshots |
| `data/processed/ivcnz/data.yaml` | §3 split sizes |
