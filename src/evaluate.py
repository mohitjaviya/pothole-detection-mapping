"""Evaluate the detectors and score the pipeline's pothole COUNTS, broken down
by driving condition (day / night / wet / blur).

Two independent jobs, because they need different ground truth:

1. Detection quality — needs LABELLED IMAGES
   - baseline: run baseline.detect_potholes on every image and greedily match
     its boxes to the YOLO .txt labels at IoU >= EVAL_IOU -> precision /
     recall / F1. The classical detector emits no confidence score, so there
     is no PR curve and no mAP; the table says so instead of inventing one.
   - yolov8n / yolov8s: ultralytics val() on the dataset split -> P, R,
     mAP50, mAP50-95 (imported lazily: only this branch needs torch).
2. System-level counting — needs ONE ROW PER CLIP (condition + manual count)
   - events CSV from tracker.py -> events per clip vs the manual count ->
     absolute error, count accuracy, exact-match rate, per condition.
   - a clip that produced no events scores 0, it does not vanish.

Outputs (committed, they feed the semester report):
    reports/detection_metrics.csv
    reports/eval_comparison.csv        one row per method: tp/fp/fn + P/R/F1
    reports/eval_comparison.png        grouped bars of P/R/F1 per method
    reports/count_metrics.csv
    reports/count_by_condition.csv
    reports/eval_summary.md
    reports/eval_counts_by_condition.png

CLI (repo root):
    python -m src.evaluate --manifest data/eval_manifest.csv --events outputs/pothole_events.csv
    python -m src.evaluate --images data/processed/ivcnz/test/images --labels data/processed/ivcnz/test/labels
    python -m src.evaluate --data-yaml data/processed/ivcnz/data.yaml [--models yolov8n yolov8s]
    # score PRE-COMPUTED predictions (no model, no torch) — name=path pairs:
    python -m src.evaluate --images <dir> --labels <dir> \
        --methods baseline=outputs/preds_baseline.csv yolov8n=outputs/preds_yolov8n.csv

`name=path` accepts either a CSV (columns image, x1, y1, x2, y2 — absolute
pixels) or a folder of YOLO .txt files (cls cx cy w h, normalized). That is
how a Colab-produced prediction set can be scored locally without torch.
"""

import argparse
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd

from src import baseline, config

Box = tuple[float, float, float, float]
MANIFEST_COLUMNS = {"source", "condition", "gt_count"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

COUNT_COLUMNS = [
    "source", "condition", "gt_count", "predicted", "abs_error", "count_accuracy", "exact",
]
CONDITION_COLUMNS = [
    "condition", "clips", "gt_total", "predicted_total", "mae", "exact_rate", "mean_count_accuracy",
]
DETECTION_COLUMNS = [
    "detector", "images", "instances", "tp", "fp", "fn", "precision", "recall", "f1", "map50", "map50_95",
]


# ---------------------------------------------------------------------------
# Detection quality
# ---------------------------------------------------------------------------
def load_yolo_label(label_path: Path, width: int, height: int) -> list[Box]:
    """Read a YOLO .txt label (cls cx cy w h, normalized) -> absolute xyxy.

    A MISSING file means "no potholes in this image", which is legitimate
    ground truth (a clean-road negative), so it returns [] rather than raising.
    The class column is ignored: the dataset is single-class (pothole).
    """
    if not label_path.exists():
        return []
    boxes: list[Box] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue  # blank line or malformed row
        _, cx, cy, w, h = (float(value) for value in parts[:5])
        boxes.append(
            (
                (cx - w / 2) * width,
                (cy - h / 2) * height,
                (cx + w / 2) * width,
                (cy + h / 2) * height,
            )
        )
    return boxes


def iou_matrix(pred: list[Box], gt: list[Box]) -> np.ndarray:
    """IoU of every prediction against every ground-truth box."""
    matrix = np.zeros((len(pred), len(gt)), dtype=float)
    for i, (px1, py1, px2, py2) in enumerate(pred):
        for j, (gx1, gy1, gx2, gy2) in enumerate(gt):
            ix1, iy1 = max(px1, gx1), max(py1, gy1)
            ix2, iy2 = min(px2, gx2), min(py2, gy2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            area_p = max(0.0, px2 - px1) * max(0.0, py2 - py1)
            area_g = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
            union = area_p + area_g - inter
            matrix[i, j] = inter / union if union > 0 else 0.0
    return matrix


def match_boxes(pred: list[Box], gt: list[Box], iou_thr: float) -> tuple[int, int, int]:
    """Greedy one-to-one matching at IoU >= iou_thr -> (tp, fp, fn).

    Each box may be used once: a single prediction cannot claim two potholes,
    and two predictions cannot both claim the same one. Best-IoU pairs are
    taken first, which is the standard cheap stand-in for optimal assignment.
    """
    if not gt:
        return 0, len(pred), 0
    if not pred:
        return 0, 0, len(gt)
    ious = iou_matrix(pred, gt)
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    tp = 0
    while True:
        best_iou, best_pair = iou_thr, None
        for i in range(len(pred)):
            if i in used_pred:
                continue
            for j in range(len(gt)):
                if j in used_gt:
                    continue
                if ious[i, j] >= best_iou and (best_pair is None or ious[i, j] > ious[best_pair[0], best_pair[1]]):
                    best_iou, best_pair = ious[i, j], (i, j)
        if best_pair is None:
            break
        tp += 1
        used_pred.add(best_pair[0])
        used_gt.add(best_pair[1])
    return tp, len(pred) - tp, len(gt) - tp


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Standard P/R/F1; an undefined ratio (no predictions / no potholes) is 0.0."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def label_path_for(image: Path, labels_dir: Path | None) -> Path:
    """Labels live either in a mirrored folder (--labels) or next to the image."""
    if labels_dir is not None:
        return labels_dir / f"{image.stem}.txt"
    return image.with_suffix(".txt")


def image_paths(images_dir: Path) -> list[Path]:
    """Every image file under images_dir, sorted (deterministic scoring order)."""
    images = sorted(
        path for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise RuntimeError(f"no images found in {images_dir}")
    return images


def score_images(
    images_dir: Path,
    labels_dir: Path | None,
    iou_thr: float,
    boxes_for_image: Callable[[Path, np.ndarray], list[Box]],
) -> dict[str, int]:
    """Accumulate tp / fp / fn over an image set, whatever produced the boxes.

    One shared loop for the classical baseline and for pre-computed
    prediction files, so both are scored by exactly the same matching code
    (a comparison is only fair if the two sides go through the same gate).

    An image with an EMPTY or MISSING label file is background: every box
    predicted on it is a false positive.
    """
    images = image_paths(images_dir)
    tp = fp = fn = 0
    for image_path in images:
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        height, width = frame.shape[:2]
        gt = load_yolo_label(label_path_for(image_path, labels_dir), width, height)
        image_tp, image_fp, image_fn = match_boxes(boxes_for_image(image_path, frame), gt, iou_thr)
        tp, fp, fn = tp + image_tp, fp + image_fp, fn + image_fn
    return {"images": len(images), "tp": tp, "fp": fp, "fn": fn}


def evaluate_baseline(
    images_dir: Path,
    labels_dir: Path | None = None,
    iou_thr: float = config.EVAL_IOU,
    detector: Callable[[np.ndarray], list[Box]] = baseline.detect_potholes,
) -> dict[str, object]:
    """Baseline P/R/F1 against YOLO labels.

    `detector` is injectable so tests can score known boxes through the exact
    same matching path (no dependence on what Canny happens to find).
    """
    counts = score_images(images_dir, labels_dir, iou_thr, lambda _path, frame: detector(frame))

    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    print(f"[evaluate] baseline on {counts['images']} images: tp={tp} fp={fp} fn={fn}")
    return {
        "detector": "baseline",
        "images": counts["images"],
        "instances": tp + fn,  # ground-truth potholes = hits + misses
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": None,      # no confidence score -> no PR curve, hence no mAP
        "map50_95": None,
    }


# ---------------------------------------------------------------------------
# Pre-computed predictions (--methods name=path): no model, no torch
# ---------------------------------------------------------------------------
PREDICTION_COLUMNS = {"image", "x1", "y1", "x2", "y2"}


def load_prediction_csv(path: Path) -> dict[str, list[Box]]:
    """Prediction CSV (image, x1, y1, x2, y2 — absolute pixels) -> boxes per
    image NAME, so a path in the CSV still matches a flat image folder."""
    table = pd.read_csv(path)
    missing = PREDICTION_COLUMNS - set(table.columns)
    if missing:
        raise RuntimeError(f"prediction CSV missing columns {sorted(missing)}: {path}")
    predictions: dict[str, list[Box]] = {}
    for row in table.itertuples(index=False):
        predictions.setdefault(Path(str(row.image)).name, []).append(
            (float(row.x1), float(row.y1), float(row.x2), float(row.y2))
        )
    return predictions


def load_prediction_dir(path: Path, images_dir: Path) -> dict[str, list[Box]]:
    """A folder of YOLO .txt predictions (cls cx cy w h, normalized).

    The same parser as the labels: a prediction file is in the same format,
    it just comes from a detector instead of a human annotator.
    """
    by_stem = {image.stem: image for image in image_paths(images_dir)}
    predictions: dict[str, list[Box]] = {}
    for txt in sorted(path.glob("*.txt")):
        image_path = by_stem.get(txt.stem)
        if image_path is None:
            print(f"[evaluate] WARNING: prediction {txt.name} has no matching image — skipped")
            continue
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        height, width = frame.shape[:2]
        predictions[image_path.name] = load_yolo_label(txt, width, height)
    return predictions


def load_predictions(path: Path, images_dir: Path) -> dict[str, list[Box]]:
    """--methods target: a prediction CSV or a folder of YOLO .txt files."""
    return load_prediction_dir(path, images_dir) if path.is_dir() else load_prediction_csv(path)


def parse_method(text: str) -> tuple[str, Path]:
    """'yolov8n=outputs/preds.csv' -> ('yolov8n', Path('outputs/preds.csv'))."""
    name, separator, raw_path = text.partition("=")
    if not separator or not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError(f"bad --methods value {text!r} (expected name=path)")
    return name.strip(), Path(raw_path.strip())


def evaluate_predictions(
    name: str,
    path: Path,
    images_dir: Path,
    labels_dir: Path | None = None,
    iou_thr: float = config.EVAL_IOU,
) -> dict[str, object]:
    """P/R/F1 for a pre-computed prediction set (no weights, no torch).

    This is the local half of the Colab workflow: run the detectors on the
    GPU, dump their boxes, then score every method here through one gate.
    """
    predictions = load_predictions(path, images_dir)
    counts = score_images(
        images_dir, labels_dir, iou_thr, lambda image_path, _frame: predictions.get(image_path.name, [])
    )

    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    print(f"[evaluate] {name} on {counts['images']} images: tp={tp} fp={fp} fn={fn}")
    return {
        "detector": name,
        "images": counts["images"],
        "instances": tp + fn,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": None,      # boxes carry no scores here -> no PR curve
        "map50_95": None,
    }


def split_images_dir(data_yaml: Path, split: str) -> Path | None:
    """Resolve a split's image folder from the generated data.yaml."""
    try:
        import yaml  # pyyaml ships with ultralytics; only needed for YOLO metrics
    except ImportError:
        return None
    try:
        cfg = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8")) or {}
    except OSError:
        return None
    entry = cfg.get(split)
    if not isinstance(entry, str):
        return None
    folder = Path(entry)
    return folder if folder.is_absolute() else Path(cfg.get("path") or ".") / folder


def count_split_images(data_yaml: Path, split: str) -> int | None:
    """How many images the split holds (None when it cannot be resolved)."""
    folder = split_images_dir(data_yaml, split)
    if folder is None or not folder.exists():
        return None
    return sum(1 for path in folder.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def evaluate_yolo(weights: Path, data_yaml: Path, split: str = config.EVAL_SPLIT) -> dict[str, object]:
    """P/R/mAP for one YOLOv8 weight file on the labelled dataset split."""
    from ultralytics import YOLO  # lazy: only this branch needs torch

    model = YOLO(str(weights))
    metrics = model.val(data=str(data_yaml), split=split, imgsz=config.IMGSZ, verbose=False)
    return {
        "detector": weights.stem,
        "images": count_split_images(data_yaml, split),
        "instances": int(sum(metrics.nt_per_class)) if hasattr(metrics, "nt_per_class") else None,
        "tp": None,  # ultralytics reports ratios, not raw counts
        "fp": None,
        "fn": None,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "f1": None,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }


# ---------------------------------------------------------------------------
# System-level counting
# ---------------------------------------------------------------------------
def load_manifest(path: Path, conditions: tuple[str, ...] = config.EVAL_CONDITIONS) -> pd.DataFrame:
    """Load + validate the eval manifest: source, condition, gt_count.

    Fails loudly on an unknown condition or a negative count: a typo like
    'nigth' would otherwise silently drop a clip out of the per-condition
    table and quietly flatter the results.
    """
    manifest = pd.read_csv(path)
    missing = MANIFEST_COLUMNS - set(manifest.columns)
    if missing:
        raise RuntimeError(f"manifest missing columns {sorted(missing)}: {path}")

    manifest = manifest[["source", "condition", "gt_count"]].copy()
    manifest["source"] = manifest["source"].astype(str).str.strip()
    manifest["condition"] = manifest["condition"].astype(str).str.strip().str.lower()
    unknown = sorted(set(manifest["condition"]) - set(conditions))
    if unknown:
        raise RuntimeError(f"unknown condition(s) {unknown} — allowed: {list(conditions)}")
    if manifest["source"].duplicated().any():
        duplicated = sorted(manifest.loc[manifest["source"].duplicated(), "source"])
        raise RuntimeError(f"duplicate source(s) in manifest: {duplicated}")

    manifest["gt_count"] = pd.to_numeric(manifest["gt_count"], errors="raise").astype(int)
    if (manifest["gt_count"] < 0).any():
        raise RuntimeError("gt_count must be >= 0")
    if manifest.empty:
        raise RuntimeError(f"manifest has no rows: {path}")
    return manifest


def count_accuracy(predicted: int, actual: int) -> float:
    """1.0 = exact count, 0.0 = off by 100 % or more (clipped, never negative)."""
    if actual == 0:
        return 1.0 if predicted == 0 else 0.0
    return max(0.0, 1.0 - abs(predicted - actual) / actual)


def evaluate_counts(events: pd.DataFrame, manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-clip and per-condition count tables.

    A pothole is counted per EVENT (tracker.py's output), so an untagged
    event (GPS gap) still counts — a missing GPS fix is not a missed pothole.
    Clips in the manifest that produced no events score 0 rather than being
    skipped, and events from clips absent from the manifest are reported.
    """
    if "source" not in events.columns:
        raise RuntimeError("events CSV needs a 'source' column (tracker.py output)")

    predicted = events.groupby("source").size()
    unknown = sorted(set(predicted.index) - set(manifest["source"]))
    if unknown:
        print(f"[evaluate] WARNING: {len(unknown)} source(s) not in the manifest, excluded: {unknown}")

    rows = [
        {
            "source": row.source,
            "condition": row.condition,
            "gt_count": int(row.gt_count),
            "predicted": int(predicted.get(row.source, 0)),
        }
        for row in manifest.itertuples(index=False)
    ]
    per_clip = pd.DataFrame(rows)
    per_clip["abs_error"] = (per_clip["predicted"] - per_clip["gt_count"]).abs()
    per_clip["count_accuracy"] = [
        count_accuracy(int(p), int(g)) for p, g in zip(per_clip["predicted"], per_clip["gt_count"])
    ]
    per_clip["exact"] = per_clip["predicted"] == per_clip["gt_count"]

    condition_rows = []
    for condition, group in per_clip.groupby("condition", sort=True):
        condition_rows.append(_condition_row(condition, group))
    # ALL row last, so the report reads per-condition first, overall second.
    condition_rows.append(_condition_row("ALL", per_clip))
    per_condition = pd.DataFrame(condition_rows, columns=CONDITION_COLUMNS)
    return per_clip.reindex(columns=COUNT_COLUMNS), per_condition


def _condition_row(condition: str, group: pd.DataFrame) -> dict[str, object]:
    """Aggregate one condition (or all clips) into a single report row."""
    return {
        "condition": condition,
        "clips": len(group),
        "gt_total": int(group["gt_count"].sum()),
        "predicted_total": int(group["predicted"].sum()),
        "mae": round(float(group["abs_error"].mean()), 3),
        "exact_rate": round(float(group["exact"].mean()), 3),
        "mean_count_accuracy": round(float(group["count_accuracy"].mean()), 3),
    }


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
INTEGER_COLUMNS = {"images", "instances", "tp", "fp", "fn"}


def markdown_table(frame: pd.DataFrame) -> str:
    """Pipe table for eval_summary.md (pandas' to_markdown needs `tabulate`,
    which is not a project dependency).

    Count columns are rendered as integers: a missing value in one detector's
    row (YOLO reports ratios, not raw counts) would otherwise turn the whole
    column into floats and print '10.000 potholes'.
    """
    display = frame.copy()
    for column in display.columns:
        if column in INTEGER_COLUMNS:
            # dtype=object up front: pandas would otherwise infer float64 from
            # the None, turning 10 into 10.0 and printing '10.000 potholes'.
            display[column] = pd.Series(
                [None if pd.isna(value) else int(value) for value in display[column]],
                index=display.index,
                dtype=object,
            )
    header = "| " + " | ".join(str(column) for column in display.columns) + " |"
    divider = "| " + " | ".join("---" for _ in display.columns) + " |"
    body = [
        "| " + " | ".join(_cell(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *body])


def _cell(value: object) -> str:
    """Render a metric cell: None -> '-', floats to 3 decimals."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def comparison_table(detection: pd.DataFrame) -> pd.DataFrame:
    """Per-method comparison table for eval_comparison.csv / .png.

    `method` is the detector name. When a method reports precision and recall
    but no F1 (ultralytics val() gives P/R/mAP, not F1), F1 is derived from
    them — the harmonic mean, exactly what F1 means — so the table compares
    like with like instead of showing a blank.

    Types are part of the contract, because this CSV is pasted into the report:
    counts (images, instances, tp, fp, fn) are nullable Int64, so they print as
    `10` and not `10.0`, and stay BLANK for a method that reports no raw counts
    (ultralytics gives ratios, not tp/fp/fn). Scores are floats rounded to 3
    decimals — the precision anyone actually quotes.
    """
    table = detection.copy()
    table = table.rename(columns={"detector": "method"})
    for column in ("precision", "recall", "f1"):
        if column not in table.columns:
            table[column] = float("nan")
    precision = pd.to_numeric(table["precision"], errors="coerce")
    recall = pd.to_numeric(table["recall"], errors="coerce")
    f1 = pd.to_numeric(table["f1"], errors="coerce")
    derived = 2 * precision * recall / (precision + recall)
    table["f1"] = f1.fillna(derived)
    columns = ["method", "images", "instances", "tp", "fp", "fn", "precision", "recall", "f1"]
    extra = [column for column in ("map50", "map50_95") if column in table.columns]
    table = table.reindex(columns=columns + extra)

    for column in ("images", "instances", "tp", "fp", "fn"):
        if column in table.columns:
            # Counts are whole numbers; round() only clears float noise from a
            # CSV round-trip. Int64 keeps NA representable -> an empty cell.
            table[column] = pd.to_numeric(table[column], errors="coerce").round(0).astype("Int64")
    for column in ("precision", "recall", "f1", "map50", "map50_95"):
        if column in table.columns:
            table[column] = pd.to_numeric(table[column], errors="coerce").round(3)
    return table


def _pyplot() -> Any:
    """matplotlib.pyplot with the headless Agg backend, or None when missing.

    A missing figure must never fail an evaluation run: the CSVs are the
    source of truth, the PNG is a convenience for the report.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: no display on the lab machines
        import matplotlib.pyplot as plt
    except ImportError:
        print("[evaluate] matplotlib not installed — skipping the figure")
        return None
    return plt


def plot_comparison(detection: pd.DataFrame, out_path: Path) -> Path | None:
    """Grouped bars: precision / recall / F1 per method (eval_comparison.png).

    Uses comparison_table(), so a method without a reported F1 still gets the
    derived value rather than a misleading zero-height bar.
    """
    plt = _pyplot()
    if plt is None or detection.empty:
        return None

    data = comparison_table(detection)
    for column in ("precision", "recall", "f1"):
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0.0)

    positions = np.arange(len(data))
    width = 0.26
    figure, axis = plt.subplots(figsize=(8, 4))
    for offset, (column, label, color) in zip(
        (-width, 0.0, width),
        (
            ("precision", "precision", "#4c72b0"),
            ("recall", "recall", "#dd8452"),
            ("f1", "F1", "#55a868"),
        ),
    ):
        axis.bar(positions + offset, data[column], width, label=label, color=color)
    axis.set_xticks(positions, [str(value) for value in data["method"]], rotation=15)
    axis.set_ylim(0, 1)
    axis.set_ylabel("score (IoU 0.5)")
    axis.set_title("Detector comparison")
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return out_path


def plot_counts(per_condition: pd.DataFrame, out_path: Path) -> Path | None:
    """Grouped bars: manual count vs pipeline count per condition.

    Returns None when matplotlib is missing — a missing figure must never
    fail an evaluation run (the CSVs are the source of truth).
    """
    plt = _pyplot()
    if plt is None:
        return None

    data = per_condition[per_condition["condition"] != "ALL"]
    if data.empty:
        return None

    positions = np.arange(len(data))
    width = 0.38
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(positions - width / 2, data["gt_total"], width, label="manual count", color="#4c72b0")
    axis.bar(positions + width / 2, data["predicted_total"], width, label="pipeline count", color="#dd8452")
    axis.set_xticks(positions, [str(value) for value in data["condition"]])
    axis.set_ylabel("potholes")
    axis.set_title("Pothole count by driving condition")
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return out_path


def write_reports(
    detection: pd.DataFrame | None,
    per_clip: pd.DataFrame | None,
    per_condition: pd.DataFrame | None,
    out_dir: Path = config.REPORTS_DIR,
) -> list[Path]:
    """Write whichever tables were produced; returns the written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if detection is not None:
        path = out_dir / "detection_metrics.csv"
        detection.to_csv(path, index=False)
        written.append(path)
        # Same numbers under the name the report refers to, plus the figure.
        comparison = out_dir / "eval_comparison.csv"
        comparison_table(detection).to_csv(comparison, index=False)
        written.append(comparison)
        figure = plot_comparison(detection, out_dir / "eval_comparison.png")
        if figure is not None:
            written.append(figure)
    if per_clip is not None:
        path = out_dir / "count_metrics.csv"
        per_clip.to_csv(path, index=False)
        written.append(path)
    if per_condition is not None:
        path = out_dir / "count_by_condition.csv"
        per_condition.to_csv(path, index=False)
        written.append(path)
        figure = plot_counts(per_condition, out_dir / "eval_counts_by_condition.png")
        if figure is not None:
            written.append(figure)

    summary = out_dir / "eval_summary.md"
    parts = ["# Evaluation summary", ""]
    if detection is not None:
        parts += [
            "## Detection quality",
            "",
            markdown_table(detection),
            "",
            "The classical baseline emits no confidence score, so it has no PR curve:",
            "mAP is reported as `-` for it by design, not because it failed.",
            "",
        ]
    if per_condition is not None:
        parts += ["## Pothole counting by condition", "", markdown_table(per_condition), ""]
    if per_clip is not None:
        parts += ["## Per clip", "", markdown_table(per_clip), ""]
    summary.write_text("\n".join(parts), encoding="utf-8")
    written.append(summary)
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    """CLI entry: python -m src.evaluate [flags]"""
    parser = argparse.ArgumentParser(
        description="Compare baseline vs YOLOv8 and score pothole counts by condition"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="eval manifest CSV: source, condition, gt_count (counting mode)",
    )
    parser.add_argument("--events", type=Path, default=None, help="events CSV from tracker.py")
    parser.add_argument(
        "--images",
        type=Path,
        default=None,
        help="labelled image dir (baseline detection mode)",
    )
    parser.add_argument("--labels", type=Path, default=None, help="YOLO .txt dir (default: next to the images)")
    parser.add_argument(
        "--methods",
        nargs="+",
        type=parse_method,
        default=None,
        metavar="NAME=PATH",
        help="pre-computed predictions to score (CSV of image,x1,y1,x2,y2 or a folder "
        "of YOLO .txt) — needs --images; no model, no torch",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=config.EVAL_IOU,
        help="IoU a predicted box needs to count as a hit (config: EVAL_IOU)",
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=None,
        help="dataset data.yaml for YOLO val() (yolov8n/s detection mode)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(config.EVAL_MODELS),
        help="weight names compared as models/<name>_best.pt (config: EVAL_MODELS)",
    )
    parser.add_argument("--split", default=config.EVAL_SPLIT, help="dataset split for YOLO metrics")
    parser.add_argument("--out-dir", type=Path, default=config.REPORTS_DIR, help="where the reports go")
    args = parser.parse_args()

    if (args.manifest is None) != (args.events is None):
        parser.error("counting mode needs BOTH --manifest and --events")
    if args.manifest is None and args.images is None and args.data_yaml is None:
        parser.error("nothing to evaluate: pass --manifest/--events, --images, and/or --data-yaml")

    detection_rows: list[dict[str, object]] = []
    if args.images is not None:
        detection_rows.append(evaluate_baseline(args.images, args.labels, args.iou))
        for name, path in args.methods or []:
            if not path.exists():
                print(f"[evaluate] skipping {name}: {path} not found")
                continue
            detection_rows.append(evaluate_predictions(name, path, args.images, args.labels, args.iou))
    elif args.methods:
        parser.error("--methods needs --images (the labelled image set to score against)")
    if args.data_yaml is not None:
        for name in args.models:
            weights = config.MODELS_DIR / f"{name}_best.pt"
            if not weights.exists():
                print(f"[evaluate] skipping {name}: {weights} not found")
                continue
            print(f"[evaluate] validating {weights.name} on the {args.split} split")
            detection_rows.append(evaluate_yolo(weights, args.data_yaml, args.split))

    detection = pd.DataFrame(detection_rows, columns=DETECTION_COLUMNS) if detection_rows else None
    per_clip = per_condition = None
    if args.manifest is not None and args.events is not None:
        manifest = load_manifest(args.manifest)
        events = pd.read_csv(args.events)
        per_clip, per_condition = evaluate_counts(events, manifest)

    written = write_reports(detection, per_clip, per_condition, args.out_dir)
    if detection is not None:
        # fillna for the console only: '-' reads better than NaN for the
        # baseline's by-design missing mAP.
        print(detection.fillna("-").to_string(index=False))
    if per_condition is not None:
        print(per_condition.to_string(index=False))
    for path in written:
        print(f"[evaluate] wrote {path}")


if __name__ == "__main__":
    main()
