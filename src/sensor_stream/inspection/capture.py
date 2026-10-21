"""Offline inspection of decoded captures. No network access required."""

import json
from collections import Counter
from math import acos, degrees, isfinite, sqrt
from pathlib import Path
from statistics import mean, median
from typing import Any
from .gps import validate_gps, gps_summary, plot_gps
from .depth import check_depth, depth_summary, plot_depth
from .camera import check_frame, camera_summary, plot_camera


def load_capture(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = json.loads((path / "metadata.json").read_text())
    if not isinstance(metadata, dict):
        raise ValueError("metadata.json must contain an object")
    events = []
    with (path / "events.jsonl").open() as file:
        for number, line in enumerate(file, 1):
            try:
                event = json.loads(line)
                if not isinstance(event, dict) or not isinstance(
                    event.get("type"), str
                ):
                    raise ValueError("Missing event type")
                if event["type"] == "GpsSample":
                    validate_gps(event)
                # Asset checks annotate frame records for summaries and plots; structural
                # validation errors below are reported with their JSONL line number.
                if event["type"] in ("CameraSample", "DepthSample"):
                    check_frame(path, event)
                    if event["type"] == "DepthSample":
                        check_depth(path, event)
                if event["type"] in (
                    "ImuSample",
                    "GpsSample",
                    "CameraSample",
                    "DepthSample",
                ):
                    for key in ("source_time_ns", "received_monotonic_ns"):
                        if type(event[key]) is not int:
                            raise ValueError(f"{key} must be an integer")
                    for key in ("session_id", "stream_id"):
                        if not isinstance(event[key], str):
                            raise ValueError(f"{key} must be a string")
                if event["type"] == "ImuSample":
                    for key in ("acceleration", "angular_velocity"):
                        _vector(event[key], 3)
                    if event.get("orientation") is not None:
                        _vector(event["orientation"], 4)
                    for key in ("source_time_ns", "received_monotonic_ns"):
                        if type(event[key]) is not int:
                            raise ValueError(f"{key} must be an integer")
                    for key in ("session_id", "stream_id"):
                        if not isinstance(event[key], str):
                            raise ValueError(f"{key} must be a string")
                events.append(event)
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"events.jsonl line {number}: {exc}") from exc
    return metadata, events


def _vector(value: Any, size: int) -> None:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"Expected a vector of length {size}")
    if not all(type(v) in (float, int) and isfinite(v) for v in value):
        raise ValueError("Vector values must be finite numbers")


def _groups(
    events: list[dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for event in events:
        if event["type"] == "ImuSample":
            groups.setdefault((event["session_id"], event["stream_id"]), []).append(
                event
            )
    return groups


def summarize(metadata: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(event["type"] for event in events)
    total = counts["ImuSample"]
    gps_total = counts["GpsSample"]
    camera_total = counts["CameraSample"]
    depth_total = counts["DepthSample"]
    all_total = total + gps_total + camera_total + depth_total
    warnings = []
    for label, value in (
        ("metadata.samples_written", metadata.get("samples_written")),
        (
            "final_stats.decoded_samples",
            metadata.get("final_stats", {}).get("decoded_samples"),
        ),
    ):
        if value is not None and value != all_total:
            warnings.append(
                f"{label}={value}, but file contains {all_total} sensor samples"
            )
    for row in camera_summary(events) + depth_summary(events):
        if any(key not in ("ok", "unchecked") for key in row["file_checks"]):
            warnings.append(f"Camera payload integrity issues: {row['file_checks']}")
    streams = []
    # Compute timing within each session/stream so reconnects or other sensors
    # do not create spurious sample intervals.
    for (session, stream), samples in _groups(events).items():
        times = [e["source_time_ns"] for e in samples]
        intervals = [(b - a) / 1e6 for a, b in zip(times, times[1:])]
        span = (times[-1] - times[0]) / 1e9
        item: dict[str, Any] = {
            "session_id": session,
            "stream_id": stream,
            "samples": len(samples),
            "source_span_s": span,
            "average_rate_hz": (len(samples) - 1) / span if span > 0 else None,
            "interval_ms": (
                {
                    "min": min(intervals),
                    "median": median(intervals),
                    "max": max(intervals),
                }
                if intervals
                else None
            ),
            "gaps_over_30ms": sum(dt > 30 for dt in intervals),
            "timestamp_regressions": sum(dt < 0 for dt in intervals),
        }
        for key in ("acceleration", "angular_velocity"):
            vectors = [s[key] for s in samples]
            item[key] = {
                "axis_mean": [mean(v[i] for v in vectors) for i in range(3)],
                "rms_magnitude": sqrt(mean(sum(x * x for x in v) for v in vectors)),
                "max_magnitude": max(sqrt(sum(x * x for x in v)) for v in vectors),
            }
        norms = [
            sqrt(sum(x * x for x in s["orientation"]))
            for s in samples
            if s.get("orientation") is not None
        ]
        item["quaternion_norm_range"] = [min(norms), max(norms)] if norms else None
        streams.append(item)
    return {
        "status": metadata.get("status"),
        "imu_samples": total,
        "gps_samples": gps_total,
        "gps_streams": gps_summary(events),
        "camera_samples": camera_total,
        "camera_streams": camera_summary(events),
        "depth_samples": depth_total,
        "depth_streams": depth_summary(events),
        "event_counts": dict(counts),
        "final_stats": metadata.get("final_stats"),
        "streams": streams,
        "warnings": warnings,
        "units": "wire values; acceleration/gyro units and frame conventions unverified",
        "timing": "source-clock rate includes pauses; gaps above 30 ms are diagnostics, not proven loss",
    }


def _plot_non_depth(
    events: list[dict[str, Any]], output: Path, *, capture: Path | None = None
) -> None:
    has_camera = any(e["type"] == "CameraSample" for e in events)
    if has_camera and capture is None:
        raise ValueError("Camera plots require capture folder")
    groups = _groups(events)
    if not groups:
        if any(e["type"] == "GpsSample" for e in events):
            plot_gps(events, output)
            if has_camera:
                plot_camera(
                    events,
                    capture,
                    output.with_name(output.stem + "_camera" + output.suffix),
                )
            return
        if has_camera:
            plot_camera(events, capture, output)
            return
        raise ValueError("Cannot plot a capture without IMU samples")
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError as exc:
        raise ValueError('Plotting requires: pip install -e ".[plot]"') from exc
    figure = Figure(figsize=(12, 10))
    FigureCanvasAgg(figure)
    axes = figure.subplots(4, 1, sharex=True)
    origin = next(iter(groups.values()))[0]["received_monotonic_ns"]
    colors = ("tab:blue", "tab:orange", "tab:green")
    for group_number, samples in enumerate(groups.values()):
        # Local monotonic time keeps reconnects on one timeline without assuming
        # source-clock continuity. Separate lines avoid bridging reconnects.
        times = [(s["received_monotonic_ns"] - origin) / 1e9 for s in samples]
        for i, name in enumerate("xyz"):
            for axis, key in ((axes[0], "acceleration"), (axes[1], "angular_velocity")):
                axis.plot(
                    times,
                    [s[key][i] for s in samples],
                    color=colors[i],
                    label=name if group_number == 0 else None,
                    linewidth=0.8,
                )
        reference = next(
            (
                s["orientation"]
                for s in samples
                if s.get("orientation") and sum(v * v for v in s["orientation"]) > 0
            ),
            None,
        )
        angles = []
        for sample in samples:
            q = sample.get("orientation")
            norm = sqrt(sum(v * v for v in q)) if q else 0
            if reference is None or norm == 0:
                angles.append(float("nan"))
            else:
                refnorm = sqrt(sum(v * v for v in reference))
                dot = abs(sum(a * b for a, b in zip(q, reference)) / (norm * refnorm))
                angles.append(degrees(2 * acos(min(1.0, dot))))
        axes[2].plot(times, angles, linewidth=0.8)
        source_dt = [
            (b["source_time_ns"] - a["source_time_ns"]) / 1e6
            for a, b in zip(samples, samples[1:])
        ]
        receive_dt = [
            (b["received_monotonic_ns"] - a["received_monotonic_ns"]) / 1e6
            for a, b in zip(samples, samples[1:])
        ]
        for values, color, label in (
            (source_dt, "tab:blue", "source"),
            (receive_dt, "tab:orange", "receive"),
        ):
            axes[3].plot(
                times[1:],
                values,
                color=color,
                alpha=0.7,
                label=label if group_number == 0 else None,
                linewidth=0.8,
            )
    for axis, label in zip(
        axes,
        (
            "Acceleration\n(wire units)",
            "Angular velocity\n(wire units)",
            "Orientation change\nfrom stream start (degrees)",
            "Sample interval (ms)",
        ),
    ):
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)
    for i in (0, 1, 3):
        axes[i].legend()
    axes[3].set_xlabel("Laptop receive time from first sample (s)")
    figure.suptitle(
        "SensorStream IMU capture — orientation change is unsigned (0–180°)"
    )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    if any(e["type"] == "GpsSample" for e in events):
        plot_gps(events, output.with_name(output.stem + "_gps" + output.suffix))
    if has_camera:
        plot_camera(
            events, capture, output.with_name(output.stem + "_camera" + output.suffix)
        )


def plot_capture(
    events: list[dict[str, Any]], output: Path, *, capture: Path | None = None
) -> None:
    depth = [e for e in events if e["type"] == "DepthSample"]
    other = [e for e in events if e["type"] != "DepthSample"]
    has_other = any(
        e["type"] in ("ImuSample", "GpsSample", "CameraSample") for e in other
    )
    if has_other or not depth:
        _plot_non_depth(other, output, capture=capture)
    if depth:
        if capture is None:
            raise ValueError("Depth plots require capture folder")
        target = (
            output.with_name(output.stem + "_depth" + output.suffix)
            if has_other
            else output
        )
        plot_depth(depth, capture, target)
