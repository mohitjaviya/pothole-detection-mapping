"""Dataset preparation + YOLOv8 fine-tuning for pothole detection.

Stages: download -> extract + pair -> reproducible split -> data.yaml ->
fine-tune yolov8n then yolov8s.

`ultralytics` is imported LAZILY inside train_model() — the data stages
(download, split, data.yaml) must run on machines without torch installed.
The GPU training itself happens on Colab/Kaggle.

CLI (run from the repo root):
    python -m src.train [--models yolov8n yolov8s] [--epochs 50]
                        [--skip-data] [--drive-dir PATH]
                        [--zip PATH] [--source-dir PATH] [--data-dir PATH]

    --skip-data    reuse the existing split. Works only while the same Colab
                   VM is alive — after a runtime restart, rerun WITHOUT it
                   (re-clone + re-download happen automatically).
"""

import argparse
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from src import config

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def download_dataset(zip_path: Path, url: str) -> Path:
    """Download the dataset zip unless it already exists (idempotent — Colab
    restarts must not repeat a big download)."""
    if zip_path.exists():
        print(f"[train] zip already present, skipping download: {zip_path}")
        return zip_path
    print(f"[train] downloading {url}")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    last = [0]

    def report(blocks: int, block_size: int, total: int) -> None:
        done = blocks * block_size
        if done - last[0] >= 5_000_000:  # log every ~5 MB
            last[0] = done
            if total > 0:
                print(f"[train]   {done / 1e6:7.1f} / {total / 1e6:.1f} MB")

    urllib.request.urlretrieve(url, zip_path, reporthook=report)
    print(f"[train] downloaded -> {zip_path}")
    return zip_path


def extract_dataset(zip_path: Path, out_dir: Path) -> Path:
    """Extract the zip. Idempotent: skipped when out_dir already exists."""
    if out_dir.exists():
        print(f"[train] extract skipped, dir exists: {out_dir}")
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(out_dir)
    print(f"[train] extracted -> {out_dir}")
    return out_dir


def find_label_for(image: Path, dataset_dir: Path) -> Path | None:
    """Locate the YOLO .txt label for an image regardless of zip layout.

    YOLO zips come in two common shapes:
      a) label next to the image:        .../x.jpg  and  .../x.txt
      b) sibling folders:                .../images/x.jpg, .../labels/x.txt
    We try (a), then every 'images' -> 'labels' substitution of path parts.
    """
    candidates = [image.with_suffix(".txt")]
    parts = image.relative_to(dataset_dir).parts
    for i, part in enumerate(parts):
        if part.lower() == "images":
            candidates.append(
                dataset_dir.joinpath(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt")
            )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def collect_pairs(dataset_dir: Path) -> tuple[list[tuple[Path, Path]], int]:
    """Return (image, label) pairs and the count of images with no label
    (unpaired files are skipped and counted, never silently dropped)."""
    pairs: list[tuple[Path, Path]] = []
    unpaired = 0
    for image in sorted(dataset_dir.rglob("*")):
        if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = find_label_for(image, dataset_dir)
        if label is None:
            unpaired += 1
            print(f"[train] no label for {image.name}, skipping")
        else:
            pairs.append((image, label))
    print(f"[train] paired {len(pairs)} images ({unpaired} unpaired skipped)")
    return pairs, unpaired


def build_split(dataset_dir: Path, out_dir: Path) -> dict[str, int]:
    """Reproducible 80/10/10 split: copy image+label pairs into
    out_dir/{train,val,test}/{images,labels}.

    Determinism: rglob order varies across filesystems, so pairs are SORTED
    first and then shuffled with a seeded Random — same files + same seed
    give a byte-identical split on every machine and every rerun.
    """
    pairs, unpaired = collect_pairs(dataset_dir)
    ordered = sorted(pairs, key=lambda pair: pair[0].as_posix().lower())
    random.Random(config.SPLIT_SEED).shuffle(ordered)

    n = len(ordered)
    train_ratio, val_ratio, _ = config.SPLIT_RATIOS
    n_train = round(train_ratio * n)
    n_val = round(val_ratio * n)
    splits = {
        "train": ordered[:n_train],
        "val": ordered[n_train : n_train + n_val],
        "test": ordered[n_train + n_val :],
    }

    used: set[str] = set()  # stems already copied — for cross-folder name collisions
    for split_name, split_pairs in splits.items():
        images_dir = out_dir / split_name / "images"
        labels_dir = out_dir / split_name / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        for image, label in split_pairs:
            image_name = image.name
            if image_name.lower() in used:
                # Two folders in the zip contain the same stem: prefix with the
                # parent folder name so the pair stays unique and together.
                image_name = f"{image.parent.name}_{image.name}"
            label_name = Path(image_name).stem + ".txt"
            used.add(image_name.lower())
            shutil.copy2(image, images_dir / image_name)
            shutil.copy2(label, labels_dir / label_name)
        print(f"[train] {split_name}: {len(split_pairs)} images")

    return {
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "test": len(splits["test"]),
        "unpaired_skipped": unpaired,
    }


def write_data_yaml(dataset_dir: Path) -> Path:
    """Write the Ultralytics data.yaml into the dataset root.

    Contains machine-specific absolute paths, so it is a GENERATED file:
    never commit it (configs/ holds static YAML only).
    """
    yaml_path = dataset_dir / "data.yaml"
    content = (
        f"path: {dataset_dir.as_posix()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        f"nc: {len(config.CLASS_NAMES)}\n"
        f"names: {list(config.CLASS_NAMES)}\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    print(f"[train] wrote {yaml_path}")
    return yaml_path


def validate_model(model: Any, data_yaml: Path) -> dict[str, Any]:
    """Precision/recall/mAP@0.5 on the val and test splits (best weights)."""
    results: dict[str, Any] = {}
    for split in ("val", "test"):
        metrics = model.val(data=str(data_yaml), split=split, imgsz=config.IMGSZ)
        results[f"{split}_precision"] = float(metrics.box.mp)
        results[f"{split}_recall"] = float(metrics.box.mr)
        results[f"{split}_map50"] = float(metrics.box.map50)
    return results


def train_model(
    model_name: str, data_yaml: Path, epochs: int, drive_dir: Path | None
) -> dict[str, Any]:
    """Fine-tune one YOLOv8 variant; saves best.pt into models/ — and into
    Drive IMMEDIATELY when --drive-dir is given, so a dead session only ever
    loses the model currently training, never a finished one."""
    # Lazy import (owner decision): torch/ultralytics are not needed for the
    # data stages and may not exist on the local machine.
    from ultralytics import YOLO

    print(f"[train] === fine-tuning {model_name} for {epochs} epochs ===")
    model = YOLO(f"{model_name}.pt")  # pretrained COCO weights, auto-downloaded
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=config.IMGSZ,
        project=str(config.PROJECT_ROOT / "runs"),  # runs/ is gitignored
        name=f"pothole_{model_name}",
        exist_ok=True,
        seed=config.SPLIT_SEED,
    )

    best: Path = Path(model.trainer.best)
    local_copy = config.MODELS_DIR / f"{model_name}_best.pt"
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, local_copy)
    print(f"[train] best weights -> {local_copy}")

    if drive_dir is not None:
        drive_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, drive_dir / local_copy.name)
        # The whole run folder (results.csv, plots, sample predictions) is a
        # few MB — save it too, it feeds the semester report.
        run_dir = Path(model.trainer.save_dir)
        shutil.copytree(run_dir, drive_dir / run_dir.name, dirs_exist_ok=True)
        print(f"[train] weights + run folder -> {drive_dir}")

    return {"model": model_name, **validate_model(model, data_yaml)}


def write_metrics(rows: list[dict[str, Any]], out_path: Path, drive_dir: Path | None) -> None:
    """Write one row per model: precision/recall/mAP@0.5 on val and test."""
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[train] metrics -> {out_path}")
    if drive_dir is not None:
        drive_path = drive_dir / out_path.name
        df.to_csv(drive_path, index=False)
        print(f"[train] metrics -> {drive_path}")
    print(df.to_string(index=False))


def main() -> None:
    """CLI entry point: python -m src.train [flags]"""
    parser = argparse.ArgumentParser(description="Dataset prep + YOLOv8 fine-tuning (GPU on Colab)")
    parser.add_argument("--models", nargs="+", default=["yolov8n", "yolov8s"])
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="reuse an existing split (same Colab VM only — after a runtime restart, rerun WITHOUT this)",
    )
    parser.add_argument("--zip", type=Path, default=config.DATASET_ZIP, help="local dataset zip override")
    parser.add_argument("--source-dir", type=Path, default=None, help="pre-extracted dataset dir override (skips download)")
    parser.add_argument("--data-dir", type=Path, default=config.PROCESSED_DATASET_DIR, help="split output dir override (tests)")
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="prepare the dataset split + data.yaml and stop (no GPU, no training)",
    )
    parser.add_argument("--drive-dir", type=Path, default=None, help="save weights/metrics here immediately (Colab)")
    args = parser.parse_args()

    if args.skip_data and (args.data_dir / "data.yaml").exists():
        print(f"[train] --skip-data: using existing split at {args.data_dir}")
    else:
        if args.skip_data:
            print("[train] --skip-data requested but no split exists; building it")
        source_dir = args.source_dir
        if source_dir is None:
            zip_path = download_dataset(args.zip, config.DATASET_URL)
            source_dir = extract_dataset(zip_path, config.RAW_DATASET_DIR)
        build_split(source_dir, args.data_dir)
        write_data_yaml(args.data_dir)
    data_yaml = args.data_dir / "data.yaml"

    if args.data_only:
        # Dataset prep is CPU-only and cheap: useful before evaluate.py, and
        # on a Colab restart where only the split is needed.
        print(f"[train] --data-only: dataset ready at {data_yaml}, skipping training")
        return

    rows = [train_model(name, data_yaml, args.epochs, args.drive_dir) for name in args.models]
    write_metrics(rows, config.REPORTS_DIR / "train_metrics.csv", args.drive_dir)


if __name__ == "__main__":
    main()
