"""Interactive Folium map from geotagged pothole events.

Input: the *_geo.csv produced by geotag.py (needs latitude/longitude).
Pure CSV -> map: no model, no torch, no video access.

Two layers: severity-colored markers (green/yellow/red) and an optional
heatmap behind them; LayerControl toggles between them. A legend overlay
explains the colors; popups trace each pothole back to the exact video
moment (source + frame + bbox).

CLI (repo root): python -m src.map_builder <events_geo.csv> [--out PATH] [--heatmap]
"""

import argparse
from pathlib import Path

import folium
import pandas as pd
from folium.plugins import HeatMap

from src import config

MARKER_RADIUS = {"small": 8, "medium": 12, "large": 16}


def popup_html(row: pd.Series) -> str:
    """Popup traces a marker back to the exact video moment; tolerates CSVs
    without bbox/event_id columns."""
    lines = [f"<b>Pothole #{row.get('event_id', '?')}</b>"]
    if "severity" in row.index and pd.notna(row["severity"]):
        lines.append(f"severity: {row['severity']}")
    lines.append(f"{row['source']} @ frame {int(row['frame'])}")
    if {"x1", "y1", "x2", "y2"} <= set(row.index):
        lines.append(
            f"bbox: ({int(row['x1'])},{int(row['y1'])})-({int(row['x2'])},{int(row['y2'])})"
        )
    return "<br>".join(lines)


def build_map(events: pd.DataFrame, heatmap: bool = False) -> folium.Map:
    """Severity-colored markers + optional heatmap layer; blank coordinates
    are dropped and counted, never silently lost."""
    missing_cols = {"latitude", "longitude"} - set(events.columns)
    if missing_cols:
        raise RuntimeError(f"CSV missing {sorted(missing_cols)} — run geotag.py first")

    tagged = events.dropna(subset=["latitude", "longitude"])
    dropped = len(events) - len(tagged)
    if dropped:
        print(f"[map] dropped {dropped} untagged rows (no lat/lon)")
    if tagged.empty:
        print("[map] WARNING: nothing geotagged — centering on the default location")

    center = (
        (float(tagged["latitude"].mean()), float(tagged["longitude"].mean()))
        if len(tagged)
        else config.MAP_DEFAULT_CENTER
    )
    fmap = folium.Map(location=list(center), zoom_start=15, tiles="OpenStreetMap")

    markers = folium.FeatureGroup(name="Potholes by severity")
    for _, row in tagged.iterrows():
        severity = row.get("severity")
        color = config.SEVERITY_COLORS.get(severity, "#888888")
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=MARKER_RADIUS.get(severity, 8),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            popup=folium.Popup(popup_html(row), max_width=250),
        ).add_to(markers)
    markers.add_to(fmap)

    if heatmap and len(tagged):
        heat = folium.FeatureGroup(name="Heatmap")
        HeatMap(
            [[lat, lon] for lat, lon in zip(tagged["latitude"], tagged["longitude"])],
            radius=14,
        ).add_to(heat)
        heat.add_to(fmap)

    folium.LayerControl().add_to(fmap)

    # Legend + title overlay (plain HTML injected into the map page)
    legend_items = "".join(
        f'<div><span style="background:{color};display:inline-block;width:10px;'
        f'height:10px;border-radius:5px;margin-right:4px;"></span>{name}</div>'
        for name, color in config.SEVERITY_COLORS.items()
    )
    overlay = (
        '<div style="position:fixed;top:10px;left:10px;z-index:9999;background:white;'
        'padding:8px;border:1px solid #ccc;font-family:sans-serif;font-size:13px;">'
        f'<div style="font-weight:bold;margin-bottom:4px;">Potholes mapped: {len(tagged)}</div>'
        f"{legend_items}</div>"
    )
    fmap.get_root().html.add_child(folium.Element(overlay))
    return fmap


def main() -> None:
    """CLI entry: python -m src.map_builder <events_geo.csv> [--out PATH] [--heatmap]"""
    parser = argparse.ArgumentParser(description="Folium map from geotagged pothole events")
    parser.add_argument("input", type=Path, help="geotagged events CSV (*_geo.csv)")
    parser.add_argument("--out", type=Path, default=config.OUTPUTS_DIR / "map.html")
    parser.add_argument("--heatmap", action="store_true", help="add the heatmap layer")
    args = parser.parse_args()

    events = pd.read_csv(args.input)
    fmap = build_map(events, heatmap=args.heatmap)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(args.out))
    print(f"[map] saved {len(events)} events -> {args.out}")


if __name__ == "__main__":
    main()
