"""Assert evaluate.py on fake fixtures: greedy IoU matching, YOLO-label
parsing, baseline P/R/F1 through an injected detector, count scoring per clip
and per condition, the report files, and the CLI guards. No dataset, no GPU.

Run (repo root): python tests/test_evaluate.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import subprocess

import pandas as pd

from make_fake_data import main as make_events
from make_fake_eval_data import (
    GOOD_PREDICTIONS,
    IMAGES_DIR,
    LABELS_DIR,
    MANIFEST_PATH,
    WEAK_PREDICTIONS,
    main as make_eval_data,
)
from src import config
from src.evaluate import (
    count_accuracy,
    evaluate_baseline,
    evaluate_counts,
    evaluate_predictions,
    load_manifest,
    load_yolo_label,
    markdown_table,
    match_boxes,
    parse_method,
    write_reports,
)

FAKE_DIR = config.OUTPUTS_DIR / "fake"
BOX = (100.0, 100.0, 200.0, 200.0)


def test_matching() -> None:
    """Greedy one-to-one IoU matching, including the threshold boundary."""
    assert match_boxes([BOX], [BOX], 0.5) == (1, 0, 0)
    assert match_boxes([BOX], [(400.0, 300.0, 500.0, 400.0)], 0.5) == (0, 1, 1)
    assert match_boxes([BOX, BOX], [BOX], 0.5) == (1, 1, 0)  # one box per pothole
    assert match_boxes([], [BOX], 0.5) == (0, 0, 1)
    assert match_boxes([BOX], [], 0.5) == (0, 1, 0)

    # IoU = 6000 / 14000 = 0.4286: below 0.5 is a miss, above 0.4 is a hit
    shifted = (140.0, 100.0, 240.0, 200.0)
    assert match_boxes([BOX], [shifted], 0.5) == (0, 1, 1)
    assert match_boxes([BOX], [shifted], 0.4) == (1, 0, 0)
    print("[test] IoU matching: one-to-one, threshold boundary respected")


def test_labels() -> None:
    """YOLO .txt -> absolute xyxy round trip; missing AND empty labels mean
    background (no potholes), never 'unknown'."""
    parsed = load_yolo_label(LABELS_DIR / "img1.txt", 640, 360)
    expected = (300.0, 250.0, 340.0, 290.0)
    assert len(parsed) == 1, parsed
    assert all(abs(p - e) < 1e-3 for p, e in zip(parsed[0], expected)), parsed
    assert load_yolo_label(LABELS_DIR / "does_not_exist.txt", 640, 360) == []
    assert (LABELS_DIR / "img4.txt").exists(), "img4 must have an EMPTY label file"
    assert load_yolo_label(LABELS_DIR / "img4.txt", 640, 360) == []
    print("[test] labels: round-trip exact, missing and empty files -> empty ground truth")


def test_baseline_detection() -> None:
    """P/R/F1 through a stub detector with known tp/fp/fn, then the real one."""
    img1_box = (300.0, 250.0, 340.0, 290.0)

    # Weak stub: always returns img1's box. Hits img1 (1 tp), misses img2's
    # two potholes (2 fn) and false-positives on BOTH clean-road negatives
    # img3 (no label file) and img4 (empty label file) -> 3 fp.
    # P = 1/4 = 0.25, R = 1/3, F1 = 0.286
    scored = evaluate_baseline(IMAGES_DIR, LABELS_DIR, detector=lambda frame: [img1_box])
    assert (scored["tp"], scored["fp"], scored["fn"]) == (1, 3, 2), scored
    assert abs(scored["precision"] - 0.25) < 1e-9, scored["precision"]
    assert abs(scored["recall"] - 1 / 3) < 1e-9, scored["recall"]
    assert abs(scored["f1"] - 0.2857142857) < 1e-6, scored["f1"]
    assert scored["instances"] == 3, "3 ground-truth potholes (1 + 2 + 0 + 0)"
    assert scored["map50"] is None, "baseline has no confidence -> no mAP, by design"

    silent = evaluate_baseline(IMAGES_DIR, LABELS_DIR, detector=lambda frame: [])
    assert (silent["tp"], silent["fp"], silent["fn"]) == (0, 0, 3), silent
    assert silent["precision"] == 0.0 and silent["recall"] == 0.0

    # The real classical detector must run end to end; its accuracy is not
    # asserted (that is the point of the comparison), only its bookkeeping.
    real = evaluate_baseline(IMAGES_DIR, LABELS_DIR)
    assert real["tp"] + real["fn"] == 3, real
    assert real["fp"] >= 0, real
    print(
        f"[test] baseline: weak stub P=0.25 R=1/3 F1=0.286, "
        f"real run tp={real['tp']} fp={real['fp']} fn={real['fn']}"
    )


def test_predictions() -> None:
    """--methods name=path: pre-computed boxes scored with no model at all."""
    weak = evaluate_predictions("weak", WEAK_PREDICTIONS, IMAGES_DIR, LABELS_DIR)
    assert (weak["tp"], weak["fp"], weak["fn"]) == (1, 3, 2), weak
    assert abs(weak["precision"] - 0.25) < 1e-9
    assert abs(weak["recall"] - 1 / 3) < 1e-9
    assert abs(weak["f1"] - 0.2857142857) < 1e-6
    assert weak["detector"] == "weak"

    good = evaluate_predictions("good", GOOD_PREDICTIONS, IMAGES_DIR, LABELS_DIR)
    assert (good["tp"], good["fp"], good["fn"]) == (3, 0, 0), good
    assert abs(good["precision"] - 1.0) < 1e-9 and abs(good["recall"] - 1.0) < 1e-9
    assert abs(good["f1"] - 1.0) < 1e-9

    assert parse_method("yolov8n=outputs/preds.csv") == ("yolov8n", Path("outputs/preds.csv"))
    try:
        parse_method("no-separator")
    except Exception as exc:  # argparse.ArgumentTypeError
        assert "name=path" in str(exc), exc
    else:
        raise AssertionError("a --methods value without '=' should be rejected")
    print("[test] predictions: weak 1/3/2 (P=0.25, R=1/3), good 3/0/0 (P=R=F1=1)")


def test_counts() -> None:
    """Per-clip and per-condition count scoring against the fake manifest."""
    assert count_accuracy(0, 0) == 1.0  # clean road, nothing found: correct
    assert count_accuracy(3, 0) == 0.0  # potholes invented on a clean road
    assert count_accuracy(5, 4) == 0.75
    assert count_accuracy(1, 4) == 0.25

    manifest = load_manifest(MANIFEST_PATH)
    events = pd.read_csv(FAKE_DIR / "pothole_events.csv")
    per_clip, per_condition = evaluate_counts(events, manifest)

    by_source = per_clip.set_index("source")
    assert by_source.loc["clip_a.mp4", "predicted"] == 4
    assert by_source.loc["clip_a.mp4", "exact"]  # 4 vs 4
    assert by_source.loc["clip_b.mp4", "abs_error"] == 1  # 4 vs 3
    assert abs(by_source.loc["clip_b.mp4", "count_accuracy"] - 2 / 3) < 1e-9
    assert by_source.loc["clip_c.mp4", "predicted"] == 0  # no events at all
    assert by_source.loc["clip_c.mp4", "count_accuracy"] == 0.0
    assert by_source.loc["clip_d.mp4", "exact"]  # 0 vs 0

    overall = per_condition.set_index("condition").loc["ALL"]
    assert overall["clips"] == 4
    assert overall["gt_total"] == 9 and overall["predicted_total"] == 8
    assert overall["mae"] == 0.75
    assert overall["exact_rate"] == 0.5
    # report values are rounded to 3 dp, so compare at that precision
    assert abs(overall["mean_count_accuracy"] - (1 + 2 / 3 + 0 + 1) / 4) < 1e-3
    assert set(per_condition["condition"]) == {"blur", "day", "night", "wet", "ALL"}
    print(f"[test] counts: {len(per_clip)} clips, ALL mae={overall['mae']} exact_rate={overall['exact_rate']}")
    return per_clip, per_condition


def test_manifest_guard() -> None:
    """A typo'd condition must fail loudly, not drop a clip from the table."""
    bad_path = FAKE_DIR / "bad_manifest.csv"
    pd.DataFrame([{"source": "x.mp4", "condition": "nigth", "gt_count": 1}]).to_csv(
        bad_path, index=False
    )
    try:
        load_manifest(bad_path)
    except RuntimeError as exc:
        assert "unknown condition" in str(exc), exc
    else:
        raise AssertionError("a misspelt condition should raise")
    print("[test] manifest: unknown condition rejected")


def test_reports(per_clip: pd.DataFrame, per_condition: pd.DataFrame) -> None:
    """Report files + summary content, written to outputs/ (never reports/)."""
    out_dir = FAKE_DIR / "reports"
    detection = pd.DataFrame([evaluate_baseline(IMAGES_DIR, LABELS_DIR, detector=lambda frame: [])])
    written = write_reports(detection, per_clip, per_condition, out_dir)
    names = {path.name for path in written}

    required = {
        "detection_metrics.csv",
        "eval_comparison.csv",
        "count_metrics.csv",
        "count_by_condition.csv",
        "eval_summary.md",
    }
    assert required <= names, names
    summary = (out_dir / "eval_summary.md").read_text(encoding="utf-8")
    assert "Detection quality" in summary and "Pothole counting by condition" in summary
    assert "| night |" in summary, "per-condition row missing from the summary"
    assert "mAP is reported as" in summary, "the baseline no-mAP caveat must be stated"
    assert "| baseline | 4 | 3 | 0 | 0 | 3 |" in summary, "detection counts must render as integers"
    print(f"[test] reports: {sorted(names)}")


def test_markdown_integers() -> None:
    """A None in one row must not turn the whole count column into floats."""
    frame = pd.DataFrame([{"detector": "baseline", "tp": 10.0}, {"detector": "yolo", "tp": None}])
    table = markdown_table(frame)
    assert "| baseline | 10 |" in table, table
    assert "| yolo | - |" in table, table
    print("[test] markdown: counts as integers, missing values as '-'")


def test_cli() -> None:
    """Both CLI modes parse, and a half-specified counting run is rejected."""
    out_dir = FAKE_DIR / "reports_cli"
    good = subprocess.run(
        [
            sys.executable, "-m", "src.evaluate",
            "--manifest", str(MANIFEST_PATH),
            "--events", str(FAKE_DIR / "pothole_events.csv"),
            "--out-dir", str(out_dir),
        ],
        cwd=config.PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert good.returncode == 0, good.stderr
    assert "count_by_condition.csv" in good.stdout, good.stdout

    # --methods name=path: pre-computed predictions, still no torch
    methods_dir = FAKE_DIR / "reports_methods"
    methods = subprocess.run(
        [
            sys.executable, "-m", "src.evaluate",
            "--images", str(IMAGES_DIR),
            "--labels", str(LABELS_DIR),
            f"--methods", f"weak={WEAK_PREDICTIONS}", f"good={GOOD_PREDICTIONS}",
            "--out-dir", str(methods_dir),
        ],
        cwd=config.PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert methods.returncode == 0, methods.stderr
    comparison = pd.read_csv(methods_dir / "eval_comparison.csv").set_index("method")
    assert comparison.loc["weak", "tp"] == 1 and comparison.loc["weak", "fp"] == 3
    assert abs(comparison.loc["weak", "precision"] - 0.25) < 1e-9
    assert comparison.loc["good", "tp"] == 3
    assert (methods_dir / "eval_comparison.png").exists(), "comparison figure missing"
    print("[test] CLI: counting mode runs, --methods scored, half-specified mode rejected")

    bad = subprocess.run(
        [sys.executable, "-m", "src.evaluate", "--manifest", str(MANIFEST_PATH)],
        cwd=config.PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert bad.returncode != 0 and "BOTH" in bad.stderr, (bad.returncode, bad.stderr)

    # --methods without --images has nothing to score against
    orphan = subprocess.run(
        [sys.executable, "-m", "src.evaluate", "--methods", f"weak={WEAK_PREDICTIONS}"],
        cwd=config.PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert orphan.returncode != 0 and "--images" in orphan.stderr, (orphan.returncode, orphan.stderr)
    print("[test] CLI: counting mode runs, --methods scored, bad invocations rejected")


def main() -> None:
    make_events()
    make_eval_data()
    test_matching()
    test_labels()
    test_baseline_detection()
    test_predictions()
    per_clip, per_condition = test_counts()
    test_manifest_guard()
    test_markdown_integers()
    test_reports(per_clip, per_condition)
    test_cli()
    print("EVALUATE TEST PASSED")


if __name__ == "__main__":
    main()
