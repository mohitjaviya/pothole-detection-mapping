# Results summary

Numbers produced by this repository, with the file each one comes from. Read
the **Limitations** section before quoting any of them: the detection metrics
are measured, the mapping numbers are not yet.

## 1. Detection quality (measured, reportable)

Test split of the IVCNZ pothole dataset: **125 images, 346 pothole instances**
(`data/processed/ivcnz`, split 80/10/10 with seed 42).
Source: `reports/detection_metrics.csv`, `reports/eval_comparison.csv`.

| Method | Precision | Recall | F1 | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- | --- |
| Classical OpenCV baseline | 0.058 | 0.029 | 0.039 | – | – |
| YOLOv8n (fine-tuned, `models/best.pt`) | 0.741 | 0.628 | 0.680\* | 0.690 | 0.367 |

\* derived as the harmonic mean of the reported precision and recall
(`eval_comparison.csv` does this automatically when a method has no F1).

The baseline has no confidence score, so it has no PR curve and therefore no
mAP — `–` is by design, not a failure. It is compared on precision / recall /
F1 at IoU 0.5, the fair comparison for a threshold-free detector.

Figure: `reports/eval_comparison.png`.

### Training comparison (from the Colab runs)

| Model | Test precision | Test recall | mAP50 |
| --- | --- | --- | --- |
| YOLOv8n (deployed) | 0.741 | 0.628 | 0.690 |
| YOLOv8s | 0.777 | 0.581 | 0.686 |

YOLOv8s is 0.9 pp more precise but 4.7 pp less sensitive, with a marginally
lower mAP50, at several times the inference cost on CPU. **YOLOv8n was
deployed**: same accuracy, smaller, CPU-friendly. `models/yolov8s_best.pt` is
not kept in this working copy, so the three-way table above is from the Colab
run rather than reproducible here.

## 2. Robustness sweep (measured, but on the wrong footage)

Source: `reports/robustness.csv`. Clip: `clip_synthetic_photos.mp4` —
**a wiring clip built from close-up photos, not dashcam footage.** The sweep
therefore proves the procedure works and shows the *shape* of the degradation;
it is not a dashcam robustness result.

| Condition | Frames | Detections | Unique tracks | Detection retention |
| --- | --- | --- | --- | --- |
| clean | 40 | 80 | 17 | 1.000 |
| dim | 40 | 69 | 14 | 0.862 |
| blur | 40 | 62 | 9 | 0.775 |
| noise | 40 | 50 | 11 | 0.625 |
| motion blur | 40 | 41 | 8 | 0.512 |

Reading: motion blur hurts most, then sensor noise, then defocus, then low
light. Track count falls faster than detection count, i.e. IDs fragment before
boxes disappear — a tracking-stability problem, not only a detection problem.

To make this reportable, re-run on a real clip:

```bash
python -m src.robustness data/sample_videos/<real_clip>.mp4 --frame-skip 5
```

## 3. Pipeline behaviour (verified, not a performance measurement)

Every stage runs end to end on fake data and on the wiring clip:

| Check | Result | Evidence |
| --- | --- | --- |
| detect -> tracker hand-off | 29 detections -> 2 events | `tests/test_detect.py` |
| flicker/split merge, short-track drop | 8 tracks -> 5 events, 1 dropped | `tests/test_tracker.py` |
| severity classes | 3 small / 3 medium / 2 large | `tests/test_chain.py` |
| geotag | 7 / 8 tagged, 1 correctly left blank | `tests/test_chain.py` |
| map | 7 markers + heatmap layer | `tests/test_chain.py` |
| one-command pipeline | 2 events, 2 markers, simulated route | `tests/test_pipeline.py` |
| dashboard | page renders, 5 sliders, demo mode | `tests/test_app.py` |

## 4. Counting accuracy (not measured yet)

`reports/count_metrics.csv` and `reports/count_by_condition.csv` are produced by
`evaluate.py` from `data/eval_manifest.csv` (per clip: condition + manual
count). That manifest needs real clips with manual counts, so **no counting
number is reportable yet.** Do not present the pipeline's event count as an
accuracy figure until this exists.

## Limitations

1. **Domain gap (largest problem).** The model is fine-tuned on the IVCNZ
   dataset, whose images are **close-up photographs of potholes**. On dashcam
   footage the pothole is a small, low-contrast patch tens of pixels wide, so
   the model keys on the wrong cues: it found nothing on a smooth city street
   (correct, no potholes) but produced two false positives on the **snowy
   verge** of a Nordic road. Fix: collect dashcam frames of real potholes and
   fine-tune on them (transfer learning from `best.pt`), and/or train at a
   larger `imgsz` with small-object augmentation.
2. **No real GPS data yet.** Every geotagged result in this repository comes
   from a **simulated** straight route (`config.ROUTE_*`) or from the fake
   fixture route. The coordinates are geographically meaningless. The map is
   a demonstration of the join, not of where potholes are.
3. **Nearest-fix join, no interpolation.** Each event takes the nearest GPS fix
   within `GPS_TOLERANCE_SEC` (2 s). With a 1 Hz logger at 35 km/h that is up
   to ~10 m of positional error along the direction of travel; at higher speed
   or a slower logger it grows. Events with no fix in tolerance are left blank
   and counted — never guessed. Linear interpolation between fixes is a
   documented stretch goal, not implemented.
4. **Double counting.** One event per track means the same pothole counts once
   per pass. Drive the same road twice and every pothole is counted twice; a
   pothole that reappears after a gap longer than `MERGE_MAX_GAP_FRAMES`
   (~0.27 s at 30 fps) starts a new event even within one clip. Geo-cell
   hashing across passes is a stretch goal, not implemented.
5. **Bounding-box area is only a proxy for size.** Severity comes from box area
   as a fraction of the frame. Boxes low in the frame are closer to the camera
   and therefore larger for the same pothole, so severity depends on where the
   pothole sits in the frame, not only on how big it is. There is no metric
   scale, no depth, and no calibration. The thresholds
   (`SEVERITY_SMALL_FRAC` / `SEVERITY_LARGE_FRAC`) were chosen as sensible
   starting values and have **not** been calibrated against measured potholes.
6. **CPU constraints shape the experiments.** Detection runs at `frame_skip 2-5`
   on CPU; a 78 s 1080p50 clip took 6.5 min. Larger skips mean a pothole can be
   missed entirely between processed frames, and ByteTrack has fewer frames to
   confirm an ID, so `MIN_TRACK_FRAMES` interacts with `frame_skip`.
7. **Robustness table is from the wiring clip** (see §2), and the fake-data
   tests use stubs, so they prove wiring and arithmetic — not accuracy.
8. **Single class.** The model detects `pothole` only. Cracks, patches and
   manholes are not distinguished, which is part of why road-edge texture
   produces false positives.
