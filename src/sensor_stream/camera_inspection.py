"""Offline camera payload integrity and thumbnail diagnostics."""

from collections import Counter
from hashlib import sha256
from pathlib import Path
from statistics import median
from typing import Any


def image_path(capture: Path, relative: str) -> Path:
    # Resolve symlinks and parent components before checking containment so
    # recorded asset paths cannot lead readers outside the capture directory.
    target = (capture / relative).resolve()
    if not target.is_relative_to(capture.resolve()):
        raise ValueError("Camera image path escapes capture folder")
    return target


def check_frame(capture: Path, event: dict[str, Any]) -> None:
    for key in ("width", "height", "image_size_bytes"):
        if type(event[key]) is not int or event[key] <= 0:
            raise ValueError(f"Camera {key} must be positive integer")
    if not isinstance(event["encoding"], str) or not isinstance(
        event["image_path"], str
    ):
        raise ValueError("Camera encoding/image_path must be strings")
    path = image_path(capture, event["image_path"])
    if not path.is_file():
        event["file_check"] = "missing"
    elif path.stat().st_size != event["image_size_bytes"]:
        event["file_check"] = "size_mismatch"
    else:
        digest = sha256(path.read_bytes()).hexdigest()
        event["file_check"] = (
            "ok" if digest == event.get("image_sha256") else "hash_mismatch"
        )


def camera_summary(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = {}
    for e in events:
        if e["type"] == "CameraSample":
            groups.setdefault((e["session_id"], e["stream_id"]), []).append(e)
    result = []
    for (session, stream), rows in groups.items():
        dt = [
            (b["source_time_ns"] - a["source_time_ns"]) / 1e9
            for a, b in zip(rows, rows[1:])
        ]
        span = (rows[-1]["source_time_ns"] - rows[0]["source_time_ns"]) / 1e9
        result.append(
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
                "repeated_timestamps": sum(v == 0 for v in dt),
                "payload_bytes": sum(e["image_size_bytes"] for e in rows),
                "file_checks": dict(
                    Counter(e.get("file_check", "unchecked") for e in rows)
                ),
            }
        )
    return result


def plot_camera(events: list[dict[str, Any]], capture: Path, output: Path) -> None:
    rows = [e for e in events if e["type"] == "CameraSample"]
    if not rows:
        raise ValueError("No camera frames to plot")
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise ValueError('Camera thumbnails require: pip install -e ".[plot]"') from exc
    figure = Figure(figsize=(12, 8))
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 3).flat
    # Spread up to six thumbnails across the capture; deduplicate indexes
    # when a short recording contains fewer than six frames.
    indexes = sorted({round(i * (len(rows) - 1) / 5) for i in range(6)})
    for axis in axes:
        axis.set_axis_off()
    # Recreate axes iterator after disabling all axes.
    for axis, index in zip(figure.axes, indexes):
        row = rows[index]
        try:
            with Image.open(image_path(capture, row["image_path"])) as image:
                actual = image.size
                image.thumbnail((640, 480))
                axis.imshow(image.convert("RGB"))
            mismatch = (
                " DIMENSION MISMATCH" if actual != (row["width"], row["height"]) else ""
            )
            axis.set_title(f"Frame {index}: {actual[0]}×{actual[1]}{mismatch}")
        except (OSError, ValueError) as exc:
            axis.text(
                0.05,
                0.5,
                f"Frame {index}: unreadable\n{type(exc).__name__}",
                transform=axis.transAxes,
            )
    figure.suptitle(
        "Recorded camera frames — stored pixels, no orientation/mirroring correction"
    )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
