"""Render per-frame tracking states without implying trajectory accuracy."""

import argparse
import csv
import json
from pathlib import Path
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("fullrate", type=Path)
parser.add_argument("stride5", type=Path)
parser.add_argument("fusionlab", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--fusionlab-fullrate", type=Path)
a = parser.parse_args()
panels = 4 if a.fusionlab_fullrate else 3
figure = Figure(figsize=(12, panels * 2))
spans = []
for index, (name, directory) in enumerate(
    [("ORB-SLAM3: all frames", a.fullrate), ("ORB-SLAM3: stride 5", a.stride5)]
):
    with (directory / "states.csv").open() as f:
        rows = list(csv.DictReader(f))
    origin = int(rows[0]["source_time_ns"])
    times = [(int(r["source_time_ns"]) - origin) / 1e9 for r in rows]
    spans.append(times[-1])
    axes = figure.add_subplot(panels, 1, index + 1)
    axes.step(times, [int(r["state"]) for r in rows], where="post", linewidth=1)
    axes.set_yticks(
        [0, 1, 2, 3, 4],
        ["No image/reset", "Initializing", "Tracking", "Recently lost", "Lost"],
    )
    axes.set_title(name, loc="left")
    axes.grid(alpha=0.2)
    changes = [
        i for i in range(1, len(rows)) if rows[i]["map_id"] != rows[i - 1]["map_id"]
    ]
    for i in changes:
        axes.axvline(times[i], color="red", alpha=0.3, linewidth=1)
fusion_runs = [a.fusionlab] + ([a.fusionlab_fullrate] if a.fusionlab_fullrate else [])
for index, directory in enumerate(fusion_runs):
    rows = [
        json.loads(line)
        for line in (directory / "poses.jsonl").read_text().splitlines()
    ]
    stride = json.loads((directory / "summary.json").read_text())["stride"]
    origin = rows[0]["source_time_ns"]
    axes = figure.add_subplot(panels, 1, index + 3)
    axes.step(
        [(r["source_time_ns"] - origin) / 1e9 for r in rows],
        [
            {"initializing": 1, "tracking": 2, "tracking_lost": 3}[r["status"]]
            for r in rows
        ],
        where="post",
        linewidth=1,
    )
    axes.set_yticks([1, 2, 3], ["Initializing", "Tracking", "Lost"])
    axes.set_title(
        f"FusionLab: landmark refinement + local BA, stride {stride}", loc="left"
    )
    spans.append((rows[-1]["source_time_ns"] - origin) / 1e9)
    axes.grid(alpha=0.2)
axes.set(xlabel="Source time from first processed frame (s)")
for panel in figure.axes:
    panel.set_xlim(0, max(spans) + 0.5)
axes.grid(alpha=0.2)
figure.suptitle("Walking capture: tracking continuity (red lines: ORB map-ID changes)")
figure.tight_layout()
a.output.parent.mkdir(parents=True, exist_ok=True)
FigureCanvasAgg(figure).print_png(str(a.output))
