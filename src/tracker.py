"""Collapse a raw track log into one event per unique pothole.

Input: raw track log CSV (frame, track_id, x1, y1, x2, y2, conf, source) —
the per-frame output of detect.py's YOLOv8 + ByteTrack stage.
Output: pothole_events.csv, ONE row per unique pothole:
event_id, frame (largest detection), bbox + conf at that frame, source,
first_frame, last_frame.

Rules (per source, never across sources):
- merge: two tracks merge if ANY pair of their boxes (one from each) is
  within MERGE_MAX_GAP_FRAMES (ORIGINAL-video frame indices) of each other
  and has IoU >= MERGE_IOU — this single rule covers both failure modes:
  flickering IDs (boxes at nearly the same time, nearly the same place) and
  split tracks (ID handoff with overlapping boxes). Edges are closed
  transitively with union-find, so A-B-C chains become one event.
- merge BEFORE the short-track filter: a flickering pothole's fragments are
  individually short and would be wiped by a pre-filter; merged, the track
  survives.
- drop: merged tracks appearing in fewer than MIN_TRACK_FRAMES distinct
  frames are discarded as likely false positives (counted, not silent).

CLI (repo root):
    python -m src.tracker <track_log.csv> [--out PATH]
        [--min-frames N] [--max-gap N] [--iou F]
The three threshold flags override config.py for tuning on real clips.
"""

import argparse
from pathlib import Path

import pandas as pd

from src import config

REQUIRED_COLUMNS = {"frame", "track_id", "x1", "y1", "x2", "y2", "conf", "source"}
EVENT_COLUMNS = [
    "event_id", "frame", "x1", "y1", "x2", "y2", "conf", "source",
    "first_frame", "last_frame",
]


def iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    ix1, iy1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    ix2, iy2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _find(parent: dict[int, int], node: int) -> int:
    """Union-find root lookup with path halving."""
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


def merge_fragments(
    log: pd.DataFrame, max_gap: int, merge_iou: float
) -> list[set[int]]:
    """Group track ids of ONE source into merged fragments via union-find.

    Edge rule: any box pair (one box from each track) with frame distance
    <= max_gap and IoU >= merge_iou means the two tracks are the same
    pothole. Frame values are original-video indices (the `frame` column).
    """
    boxes: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = {}
    for row in log.itertuples(index=False):
        boxes.setdefault(int(row.track_id), []).append(
            (int(row.frame), (row.x1, row.y1, row.x2, row.y2))
        )
    ids = sorted(boxes)
    parent = {tid: tid for tid in ids}

    def union(a: int, b: int) -> None:
        root_a, root_b = _find(parent, a), _find(parent, b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)  # deterministic root

    for i, tid_a in enumerate(ids):
        for tid_b in ids[i + 1 :]:
            for frame_a, box_a in boxes[tid_a]:
                matched = False
                for frame_b, box_b in boxes[tid_b]:
                    if abs(frame_a - frame_b) <= max_gap and iou(box_a, box_b) >= merge_iou:
                        union(tid_a, tid_b)
                        matched = True
                        break
                if matched:
                    break

    groups: dict[int, set[int]] = {}
    for tid in ids:
        groups.setdefault(_find(parent, tid), set()).add(tid)
    return list(groups.values())


def build_events(
    log: pd.DataFrame, min_frames: int, max_gap: int, merge_iou: float
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Collapse the (already sorted) log into events; returns (events, stats).

    Deterministic: sources alphabetical, events by first frame, event ids
    sequential from 1.
    """
    events: list[dict[str, object]] = []
    dropped_short = 0
    for source in sorted(log["source"].unique()):
        source_log = log[log["source"] == source]
        for members in merge_fragments(source_log, max_gap, merge_iou):
            detections = source_log[source_log["track_id"].isin(members)]
            first_frame = int(detections["frame"].min())
            if detections["frame"].nunique() < min_frames:
                dropped_short += 1
                continue
            # The largest detection anchors the event (ties -> earliest frame):
            # it is also the natural severity anchor for severity.py.
            largest = (
                detections.assign(area=(detections["x2"] - detections["x1"]) * (detections["y2"] - detections["y1"]))
                .sort_values(["area", "frame"], ascending=[False, True])
                .iloc[0]
            )
            events.append(
                {
                    "event_id": 0,  # assigned below, after all sources are ordered
                    "frame": int(largest["frame"]),
                    "x1": int(largest["x1"]),
                    "y1": int(largest["y1"]),
                    "x2": int(largest["x2"]),
                    "y2": int(largest["y2"]),
                    "conf": float(largest["conf"]),
                    "source": source,
                    "first_frame": first_frame,
                    "last_frame": int(detections["frame"].max()),
                }
            )

    result = pd.DataFrame(events)
    if not result.empty:
        result["event_id"] = range(1, len(result) + 1)
    result = result.reindex(columns=EVENT_COLUMNS)
    stats = {
        "tracks_in": int(log.groupby("source")["track_id"].nunique().sum()),
        "dropped_short": dropped_short,
        "events_out": len(result),
    }
    return result, stats


def main() -> None:
    """CLI entry: python -m src.tracker <track_log.csv> [--out PATH] [threshold flags]"""
    parser = argparse.ArgumentParser(description="Collapse a raw track log into unique pothole events")
    parser.add_argument("input", type=Path, help="raw track log CSV from detect.py")
    parser.add_argument("--out", type=Path, default=config.OUTPUTS_DIR / "pothole_events.csv")
    parser.add_argument(
        "--min-frames",
        type=int,
        default=config.MIN_TRACK_FRAMES,
        help="drop merged tracks appearing in fewer distinct frames than this (config: MIN_TRACK_FRAMES)",
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=config.MERGE_MAX_GAP_FRAMES,
        help="merge gap in ORIGINAL-video frame indices (config: MERGE_MAX_GAP_FRAMES)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=config.MERGE_IOU,
        help="min IoU for merging two fragments (config: MERGE_IOU)",
    )
    args = parser.parse_args()

    log = pd.read_csv(args.input)
    missing = REQUIRED_COLUMNS - set(log.columns)
    if missing:
        raise RuntimeError(f"track log missing columns: {sorted(missing)}")
    log = log.sort_values(["source", "frame", "track_id"]).reset_index(drop=True)

    events, stats = build_events(log, args.min_frames, args.max_gap, args.iou)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(args.out, index=False)
    print(f"[tracker] tracks in: {stats['tracks_in']}")
    print(f"[tracker] dropped short (< {args.min_frames} distinct frames): {stats['dropped_short']}")
    print(f"[tracker] events out: {stats['events_out']} -> {args.out}")


if __name__ == "__main__":
    main()
