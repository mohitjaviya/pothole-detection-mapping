"""Classical OpenCV pothole detector — the baseline for the YOLOv8 comparison.

Pipeline per frame:
resize -> grayscale -> ROI mask (road region) -> Gaussian blur -> Canny edges
-> morphological close -> contour filtering (area + shape).

The point of this module is not accuracy: it is a reference point so
evaluate.py can later show how much a fine-tuned YOLOv8 improves over it.

CLI (run from the repo root):
    python -m src.baseline <image-or-video-path>
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src import config


def detect_potholes(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Run the classical pipeline on one frame.

    Returns boxes as (x1, y1, x2, y2) in ORIGINAL frame pixels, so CSVs stay
    comparable with YOLO detections later.
    """
    height, width = frame.shape[:2]

    # 1. Resize to a fixed width (keeping aspect): stable scale for area
    #    thresholds, and cheap enough for CPU.
    scale: float = config.RESIZE_WIDTH / width
    resized = cv2.resize(frame, (config.RESIZE_WIDTH, int(height * scale)))
    r_h, r_w = resized.shape[:2]

    # 2. Grayscale: potholes are a brightness/texture contrast problem;
    #    color channels add cost but no signal here.
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # 3. ROI mask: keep only the road region (starts at ROI_TOP_FRAC of the
    #    frame height — dashcam horizon sits mid-frame). We MASK instead of
    #    slicing so coordinates stay in full-frame space: no offset math
    #    anywhere else in the pipeline.
    roi_mask = np.zeros((r_h, r_w), dtype=np.uint8)
    roi_top = int(r_h * config.ROI_TOP_FRAC)
    roi_mask[roi_top:, :] = 255

    # 4. Gaussian blur: suppress speckle noise so Canny doesn't fire on grain.
    blurred = cv2.GaussianBlur(gray, config.BLUR_KERNEL, 0)

    # 5. Canny edges: potholes read as broken dark outlines against asphalt.
    edges = cv2.Canny(blurred, config.CANNY_LOW, config.CANNY_HIGH)
    edges = cv2.bitwise_and(edges, edges, mask=roi_mask)

    # 6. Morphological close (dilate -> erode): reconnects the broken outline
    #    into one solid blob so it becomes a single contour.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.CLOSE_KERNEL)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 7. Outer contours only — we want the blob, not internal holes.
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frame_area = r_h * r_w
    boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        # 8a. Area filter: reject speckle below MIN_AREA_FRAC and huge blobs
        #     (shadows, road texture) above MAX_AREA_FRAC.
        if area < config.MIN_AREA_FRAC * frame_area or area > config.MAX_AREA_FRAC * frame_area:
            continue
        # 8b. Shape filter: extent = contour area / bbox area rejects spindly
        #     contours (lane markings); aspect range keeps blob-like boxes.
        bbox_area = w * h
        extent = area / bbox_area if bbox_area > 0 else 0.0
        aspect = w / h if h > 0 else 0.0
        if extent < config.MIN_EXTENT or not (config.ASPECT_MIN <= aspect <= config.ASPECT_MAX):
            continue
        # 9. Rescale back to ORIGINAL frame pixels before saving.
        boxes.append(
            (
                int(x / scale),
                int(y / scale),
                int((x + w) / scale),
                int((y + h) / scale),
            )
        )
    return boxes


def annotate(frame: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    """Draw green detection boxes on a copy of the frame."""
    annotated = frame.copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return annotated


def run_image(image_path: Path) -> tuple[pd.DataFrame, Path]:
    """Detect on a single image; returns (detections DataFrame, annotated image path)."""
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    boxes = detect_potholes(frame)
    out_path = config.OUTPUTS_DIR / f"baseline_{image_path.stem}.jpg"
    cv2.imwrite(str(out_path), annotate(frame, boxes))
    records = [
        dict(frame=0, x1=x1, y1=y1, x2=x2, y2=y2, source=image_path.name)
        for x1, y1, x2, y2 in boxes
    ]
    columns = ["frame", "x1", "y1", "x2", "y2", "source"]
    return pd.DataFrame.from_records(records, columns=columns), out_path


def run_video(video_path: Path) -> tuple[pd.DataFrame, Path]:
    """Detect on every FRAME_SKIP-th frame; returns (detections DataFrame, annotated video path)."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # some containers lack fps metadata; assume 30 for later geotag math
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # We keep only 1/FRAME_SKIP frames, so the annotated video's fps is divided
    # by the same factor — otherwise it would play FRAME_SKIP-times too fast.
    out_path = config.OUTPUTS_DIR / f"baseline_{video_path.stem}_annotated.mp4"
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps / config.FRAME_SKIP,
        (width, height),
    )

    records: list[dict] = []
    processed = 0
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % config.FRAME_SKIP == 0:
            boxes = detect_potholes(frame)
            # CSV `frame` column = ORIGINAL frame index (not the processed
            # counter) so geotag.py can compute timestamp = frame / fps.
            for x1, y1, x2, y2 in boxes:
                records.append(
                    dict(frame=index, x1=x1, y1=y1, x2=x2, y2=y2, source=video_path.name)
                )
            writer.write(annotate(frame, boxes))
            processed += 1
        index += 1
    capture.release()
    writer.release()
    print(f"[baseline] frames processed: {processed} of {index}")

    columns = ["frame", "x1", "y1", "x2", "y2", "source"]
    df = pd.DataFrame.from_records(records, columns=columns)
    return df, out_path


def main() -> None:
    """CLI entry point: python -m src.baseline <image-or-video-path>"""
    parser = argparse.ArgumentParser(description="Classical OpenCV pothole baseline detector")
    parser.add_argument("input", type=Path, help="Path to an image or video file")
    args = parser.parse_args()
    input_path: Path = args.input

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    if input_path.suffix.lower() in image_suffixes:
        df, annotated_path = run_image(input_path)
        print(f"[baseline] detections: {len(df)}")
    else:
        df, annotated_path = run_video(input_path)
        print(f"[baseline] detections: {len(df)}")

    csv_path = config.OUTPUTS_DIR / f"baseline_{input_path.stem}.csv"
    df.to_csv(csv_path, index=False)
    print(f"[baseline] annotated: {annotated_path}")
    print(f"[baseline] csv:       {csv_path}")


if __name__ == "__main__":
    main()
