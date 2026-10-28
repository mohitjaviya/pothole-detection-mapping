# OVERNIGHT_LOG.md

Autonomous session on branch **`overnight`**. Nothing was pushed; `main` is
untouched at `ffdd7e5`. Five commits sit on top of it:

```
191028a docs: pipeline/robustness/demo usage in README, PLAN ticks, results_summary + report_outline
a26801f feat(robustness): dim/blur/noise/motion sweep -> reports/robustness.csv, stub-tested
105a26d feat(app): tracker-threshold sliders, GPS upload, demo mode, pipeline delegation
a1b283c feat(pipeline): one-command detect->tracker->severity->geotag->map with --track-log and simulated-route fallback
ba27649 feat(evaluate): name=path --methods scoring, eval_comparison.csv+png, weak fixture tp=1 fp=3 fn=2
```

Every task ends with all seven test scripts passing (exit code 0). No downloads,
no installs, no files deleted, `models/`, `data/raw/`, `data/processed/` untouched.

---

## Task 1 — Health check: DONE

All five existing tests already passed; no real failures to fix.

```powershell
foreach ($t in @("test_detect.py","test_tracker.py","test_chain.py","test_evaluate.py","test_app.py")) { .\venv\Scripts\python.exe "tests\$t"; "exit=$LASTEXITCODE" }
```

Result: `test_detect`, `test_tracker`, `test_chain`, `test_evaluate`, `test_app`
all exit 0. No files changed in this task.

---

## Task 2 — evaluate.py: DONE (module already existed)

`src/evaluate.py` was already built, so this task was verification + the missing
pieces rather than a rewrite.

What was corrected / added:

- **Fixture numbers fixed as specified.** The "weak" method is now
  **TP 1 / FP 3 / FN 2 → P 0.25, R 1/3, F1 0.2857**. That needed a fourth
  image: `img3` has *no* label file, the new `img4` has an **empty** label file,
  so both flavours of background are exercised and both produce a false
  positive. Assertions in `tests/test_evaluate.py` updated from the old
  1/2/2 numbers.
- **`--methods name=path`** CLI: score pre-computed predictions with no model
  and no torch (CSV `image,x1,y1,x2,y2`, or a folder of YOLO `.txt`). Both go
  through the same `score_images()` gate as the classical baseline, so the
  comparison is fair.
- **`reports/eval_comparison.csv` + `reports/eval_comparison.png`** now written
  by `write_reports()`; F1 is derived from P/R when a method does not report it
  (ultralytics reports P/R/mAP). The committed pair was regenerated from the
  existing real `reports/detection_metrics.csv`.
- `--iou` flag (default `config.EVAL_IOU` = 0.5).
- Fixture now also writes `outputs/fake/preds_weak.csv` and `preds_good.csv`.

Files changed: `src/evaluate.py`, `tests/make_fake_eval_data.py`,
`tests/test_evaluate.py`, `reports/eval_comparison.csv`,
`reports/eval_comparison.png`.

Test:

```powershell
.\venv\Scripts\python.exe tests\test_evaluate.py
```

---

## Task 3 — src/pipeline.py: DONE

New module: one command for detect → tracker → severity → geotag → map.

- `--track-log PATH` skips detection entirely (no weights, no torch) — the
  chain starts from an existing track log.
- No `--gps` → a **simulated** straight route is generated from
  `config.ROUTE_*`, written next to the outputs as `<clip>_simulated_route.csv`,
  and loudly labelled; `--no-simulated-gps` refuses to invent coordinates.
- Per-stage CSVs land in one folder with a clip-stem prefix
  (`<clip>_track_log.csv`, `<clip>_pothole_events.csv`,
  `..._severity.csv`, `..._geo.csv`, `<clip>_map.html`).
- Tracker thresholds and conf/frame-skip are all CLI flags defaulting to config.

Also fixed a real crash found on the way: `geotag.geotag_events()` died on an
empty events table (`datetime.fromtimestamp(nan)`); it now returns empty
lat/lon and says so.

Files changed: `src/pipeline.py` (new), `src/geotag.py`, `src/config.py`
(`ROUTE_SPEED_KMH`, `ROUTE_SECONDS`, `SIM_START_ISO`, `PIPELINE_*`),
`tests/test_pipeline.py` (new).

Test:

```powershell
.\venv\Scripts\python.exe tests\test_pipeline.py
```

---

## Task 4 — app/streamlit_app.py: DONE

- Sidebar now has **detection** sliders (confidence, frame skip) and **tracker**
  sliders (min frames, merge gap, merge IoU) — all wired into tracker.py.
- **GPS CSV upload** added next to the `data/gps_logs/` picker; video upload kept.
- **Demo mode**: *Load demo data (no model needed)* renders metrics, severity
  chart, map, table and downloads from six synthetic events + a simulated route.
  No clip, no weights, no GPU. Labelled as simulated on the page.
- `run_pipeline()` now delegates to `src/pipeline.py` (`simulate_gps=False`), so
  the CLI and dashboard cannot drift. Still never invents coordinates.
- Heavy imports stay lazy (`ultralytics`/torch only inside `detect.load_model`,
  the component renderer only inside `render_result`).

Files changed: `app/streamlit_app.py`, `src/pipeline.py` (simulate_gps flag),
`tests/test_app.py` (demo mode, tracker-threshold test, 5-slider assertion,
demo-button click via `AppTest`).

Tests:

```powershell
.\venv\Scripts\python.exe tests\test_app.py
# server check (used once, then stopped):
venv\Scripts\streamlit.exe run app\streamlit_app.py   # health 200, page 200
```

---

## Task 5 — src/robustness.py: DONE

New module: degrades one clip four ways (dim, blur, noise, motion blur) plus a
clean pass, runs the detect loop on each, and writes `reports/robustness.csv`
(detections, unique tracks, detections/frame, retention vs clean).

- Degradation is applied **per frame inside the loop**, so model, frame skip and
  clip are identical across conditions.
- A **fresh model per condition**: ByteTrack state (`persist=True`) would
  otherwise leak track ids from the clean pass into the next one.
- `--write-videos` saves the degraded clips; `source` column in the CSV names
  the clip so nobody mistakes it for real footage.

Committed sweep (real `models/best.pt`, `--frame-skip 5`, 54 s):

| condition | detections | tracks | detection retention |
| --- | --- | --- | --- |
| clean | 80 | 17 | 1.000 |
| dim | 69 | 14 | 0.862 |
| blur | 62 | 9 | 0.775 |
| noise | 50 | 11 | 0.625 |
| motion | 41 | 8 | 0.512 |

**Caveat, on purpose:** that run is on `clip_synthetic_photos.mp4`, the wiring
clip — it proves the sweep works, it is not a dashcam robustness result.

Files changed: `src/robustness.py` (new), `src/config.py` (`ROBUSTNESS_*`),
`tests/test_robustness.py` (new), `reports/robustness.csv`.

Tests:

```powershell
.\venv\Scripts\python.exe tests\test_robustness.py
.\venv\Scripts\python.exe -m src.robustness data\sample_videos\clip_synthetic_photos.mp4 --frame-skip 5
```

---

## Task 6 — Docs: DONE

- **README.md**: one-command pipeline section, stage-by-stage commands (incl.
  `robustness.py`), `--methods name=path` + `eval_comparison` outputs, new
  robustness section with the caveat, dashboard section rewritten (tracker
  sliders, GPS upload, demo mode), build order ticked to 13 steps, "where things
  stand" rewritten.
- **PLAN.md**: steps 3, 6–11 ticked with the evidence for each; new steps 12
  (pipeline) and 13 (robustness); structure list updated.
- **reports/results_summary.md** (new): measured detection table (baseline vs
  yolov8n), the training comparison (yolov8n P 0.741 / R 0.628 / mAP50 0.690;
  yolov8s P 0.777 / R 0.581 / mAP50 0.686), the robustness table, the
  "verified, not measured" pipeline behaviour table, and an eight-point
  **Limitations** section (domain gap + snowy-verge false positives, no real GPS
  yet, nearest-fix join without interpolation, double counting, bbox area as a
  size proxy, CPU/frame-skip constraints, robustness on the wiring clip, single
  class).
- **reports/report_outline.md** (new): chapter-by-chapter outline with the file
  each section draws from, `[TODO]` markers where footage is missing, and a
  result-file → section cheat sheet.

Files changed: `README.md`, `PLAN.md`, `reports/results_summary.md`,
`reports/report_outline.md`.

---

## Final verification (all seven tests)

```powershell
foreach ($t in @("test_detect.py","test_tracker.py","test_chain.py","test_evaluate.py","test_app.py","test_pipeline.py","test_robustness.py")) { .\venv\Scripts\python.exe "tests\$t"; "exit=$LASTEXITCODE" }
```

All exit 0 (last run before the docs commit).

---

## Blocked / not done (needs you, not code)

1. **A real dashcam clip containing potholes.** `clip_a.mp4` (smooth street)
   and `clip_b.mp4` (snowy road) have no potholes; `clip_synthetic_photos.mp4`
   is a wiring fixture and must never produce report numbers.
2. **A real phone GPS log** (`data/gps_logs/*.csv`, columns
   `timestamp_sec, latitude, longitude`) plus the clip's true start time. Until
   then every geotagged result uses the simulated route.
3. **`models/yolov8s_best.pt`** — the yolov8s row in
   `reports/results_summary.md` is transcribed from the Colab run, not
   reproducible locally.
4. **`reports/train_metrics.csv`** — still needs copying out of the Colab
   session (it disappears with the runtime).
5. **Counting evaluation** (`reports/count_metrics.csv`,
   `count_by_condition.csv`, `eval_counts_by_condition.png`) — needs
   `data/eval_manifest.csv` with per-clip `condition` + manual `gt_count`.
6. **Robustness on real footage** — re-run the one-liner below once (1) exists.
7. `yolov8s` and the real-clip runs need GPU/Colab for anything time-boxed;
   local CPU is ~6.5 min for a 78 s 1080p50 clip at `--frame-skip 2`.

## What you must do manually

```powershell
# 1. record a real drive (20-30 s of broken road) + export the GPS logger CSV
#    -> data\sample_videos\real_clip.mp4
#    -> data\gps_logs\real_clip.csv   (timestamp_sec, latitude, longitude)

# 2. run the whole chain on it
.\venv\Scripts\python.exe -m src.pipeline data\sample_videos\real_clip.mp4 `
    --gps data\gps_logs\real_clip.csv --start-time "<clip start, ISO 8601>" --frame-skip 5

# 3. robustness on the same clip -> overwrite the wiring-clip table
.\venv\Scripts\python.exe -m src.robustness data\sample_videos\real_clip.mp4 --frame-skip 5

# 4. counting evaluation: write data\eval_manifest.csv (source,condition,gt_count)
.\venv\Scripts\python.exe -m src.evaluate --manifest data\eval_manifest.csv `
    --events outputs\real_clip_pothole_events.csv

# 5. Colab: run the training notebook, copy models\yolov8s_best.pt and
#    reports\train_metrics.csv back before the runtime dies, then
.\venv\Scripts\python.exe -m src.evaluate `
    --images data\processed\ivcnz\test\images --labels data\processed\ivcnz\test\labels `
    --data-yaml data\processed\ivcnz\data.yaml --models yolov8n yolov8s

# 6. when happy, merge (do not push from this branch unattended):
git checkout main; git merge overnight
```
