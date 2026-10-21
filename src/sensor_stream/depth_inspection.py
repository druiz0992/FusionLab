"""Raw depth diagnostics: no distance-unit or invalid-sentinel assumptions."""

from array import array
from collections import Counter
from math import isfinite
from pathlib import Path
from statistics import median
import sys

from .camera_inspection import image_path


def depth_values(capture: Path, event: dict) -> array:
    code = {"16uc1": "H", "32fc1": "f"}.get(event["encoding"].lower())
    if code is None:
        raise ValueError(f"Unsupported depth encoding: {event['encoding']}")
    values = array(code)
    values.frombytes(image_path(capture, event["image_path"]).read_bytes())
    # array reads in native byte order; recordings are interpreted as little-endian.
    # Swap on big-endian hosts to recover the same numeric pixel values.
    if sys.byteorder != "little":
        values.byteswap()
    if len(values) != event["width"] * event["height"]:
        raise ValueError("Depth dimensions/payload mismatch")
    return values


def check_depth(capture: Path, event: dict) -> None:
    if event["file_check"] != "ok":
        return
    try:
        values = depth_values(capture, event)
    except ValueError as exc:
        event["depth_check"] = str(exc)
        return
    finite = [v for v in values if isfinite(v)]
    positive = [v for v in finite if v > 0]
    w, h = event["width"], event["height"]
    # Sample a clipped 5x5 patch around the image centre in row-major order.
    # Its positive-value median is a raw diagnostic, without metric conversion.
    center = [
        values[y * w + x]
        for y in range(max(0, h // 2 - 2), min(h, h // 2 + 3))
        for x in range(max(0, w // 2 - 2), min(w, w // 2 + 3))
    ]
    center = [v for v in center if isfinite(v) and v > 0]
    event["depth_check"] = "ok"
    event["depth_raw_stats"] = {
        "pixels": len(values),
        "zero_count": sum(v == 0 for v in finite),
        "nonfinite_count": len(values) - len(finite),
        "negative_count": sum(v < 0 for v in finite),
        "positive_min": min(positive) if positive else None,
        "positive_max": max(positive) if positive else None,
        "center_positive_median": median(center) if center else None,
    }


def depth_summary(events: list[dict]) -> list[dict]:
    groups = {}
    for e in events:
        if e["type"] == "DepthSample":
            groups.setdefault((e["session_id"], e["stream_id"]), []).append(e)
    results = []
    for (session, stream), rows in groups.items():
        times = [e["source_time_ns"] for e in rows]
        dt = [(b - a) / 1e9 for a, b in zip(times, times[1:])]
        span = (times[-1] - times[0]) / 1e9
        centers = [
            e["depth_raw_stats"]["center_positive_median"]
            for e in rows
            if e.get("depth_raw_stats", {}).get("center_positive_median") is not None
        ]
        results.append(
            {
                "session_id": session,
                "stream_id": stream,
                "frames": len(rows),
                "dimensions": sorted({(e["width"], e["height"]) for e in rows}),
                "encodings": sorted({e["encoding"] for e in rows}),
                "source_span_s": span,
                "average_rate_hz": (len(rows) - 1) / span if span > 0 else None,
                "interval_s": (
                    {"min": min(dt), "median": median(dt), "max": max(dt)}
                    if dt
                    else None
                ),
                "timestamp_regressions": sum(v < 0 for v in dt),
                "file_checks": dict(
                    Counter(e.get("file_check", "unchecked") for e in rows)
                ),
                "depth_checks": dict(
                    Counter(e.get("depth_check", "unchecked") for e in rows)
                ),
                "center_positive_median_range_raw": (
                    [min(centers), max(centers)] if centers else None
                ),
                "zero_pixels": sum(
                    e.get("depth_raw_stats", {}).get("zero_count", 0) for e in rows
                ),
                "nonfinite_pixels": sum(
                    e.get("depth_raw_stats", {}).get("nonfinite_count", 0) for e in rows
                ),
                "units": "raw; little-endian interpretation assumed; zero/nonfinite values counted, not asserted to be invalid codes",
            }
        )
    return results


def plot_depth(events: list[dict], capture: Path, output: Path) -> None:
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        import numpy as np
    except ImportError as exc:
        raise ValueError('Depth plots require pip install -e ".[plot]"') from exc
    rows = [e for e in events if e["type"] == "DepthSample"]
    figure = Figure(figsize=(12, 8))
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 3).flat
    for axis in axes:
        axis.set_axis_off()
    indexes = sorted({round(i * (len(rows) - 1) / 5) for i in range(6)})
    for axis, index in zip(figure.axes, indexes):
        e = rows[index]
        try:
            values = np.asarray(depth_values(capture, e), dtype=float).reshape(
                e["height"], e["width"]
            )
            values = np.ma.masked_where(~np.isfinite(values) | (values <= 0), values)
            image = axis.imshow(values, cmap="viridis")
            figure.colorbar(image, ax=axis, label="Raw depth value")
            axis.set_title(f"Frame {index}: {e['width']}×{e['height']}")
        except (ValueError, OSError) as exc:
            axis.text(0.1, 0.5, str(exc), transform=axis.transAxes)
    figure.suptitle(
        "Depth frames — per-frame colour scale; zero/nonpositive/nonfinite hidden for display only"
    )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
