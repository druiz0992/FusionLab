"""GPS diagnostics; receiver scatter is not an absolute accuracy measurement."""

from collections import Counter
from math import cos, isfinite, radians, sqrt
from pathlib import Path
from statistics import mean, median
from typing import Any

# Method caveats saved with every report (what the numbers do not show).
GPS_SCATTER_NOTE = (
    "Scatter includes real motion and is not absolute accuracy. Altitude datum and "
    "covariance mapping need producer validation."
)


def validate_gps(event: dict[str, Any]) -> None:
    for key in ("latitude", "longitude", "altitude"):
        value = event[key]
        if value is not None and (
            type(value) not in (int, float) or not isfinite(value)
        ):
            raise ValueError(f"GPS {key} must be finite or null")
    for key in ("status", "service"):
        if event[key] is not None and type(event[key]) is not int:
            raise ValueError(f"GPS {key} must be integer or null")
    if type(event["position_covariance_type"]) is not int:
        raise ValueError("GPS covariance type must be integer")
    covariance = event["position_covariance"]
    if not isinstance(covariance, list) or not all(
        v is None or (type(v) in (float, int) and isfinite(v)) for v in covariance
    ):
        raise ValueError("GPS covariance must be a list of finite values or nulls")


def valid_fix(sample: dict[str, Any]) -> bool:
    lat, lon = sample["latitude"], sample["longitude"]
    return (
        sample["status"] in (0, 1, 2)
        and lat is not None
        and lon is not None
        and -90 <= lat <= 90
        and -180 <= lon <= 180
    )


def covariance_sigma(sample: dict[str, Any]) -> tuple[float, float, float] | None:
    cov = sample["position_covariance"]
    if sample["position_covariance_type"] not in (1, 2, 3) or len(cov) != 9:
        return None
    # For a row-major 3x3 covariance, these diagonal entries are axis variances.
    # Square roots give standard-deviation scales under the assumed ENU convention.
    diagonal = [cov[i] for i in (0, 4, 8)]
    if any(v is None or v < 0 for v in diagonal):
        return None
    return tuple(sqrt(v) for v in diagonal)


def local_xy(sample: dict[str, Any], origin: dict[str, Any]) -> tuple[float, float]:
    """Approximate short-range east/north offsets from degrees; not map projection."""
    # Wrap longitude differences at the date line before computing a local offset.
    dlon = (sample["longitude"] - origin["longitude"] + 180) % 360 - 180
    return (
        6378137 * radians(dlon) * cos(radians(origin["latitude"])),
        6378137 * radians(sample["latitude"] - origin["latitude"]),
    )


def gps_summary(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for event in events:
        if event["type"] == "GpsSample":
            groups.setdefault((event["session_id"], event["stream_id"]), []).append(
                event
            )
    result = []
    for (session, stream), samples in groups.items():
        good = [s for s in samples if valid_fix(s)]
        times = [s["source_time_ns"] for s in samples]
        dt = [(b - a) / 1e9 for a, b in zip(times, times[1:])]
        span = (times[-1] - times[0]) / 1e9
        xy = [local_xy(s, good[0]) for s in good] if good else []
        centre = tuple(mean(p[i] for p in xy) for i in (0, 1)) if xy else (0.0, 0.0)
        scatter = (
            sqrt(mean(sum((p[i] - centre[i]) ** 2 for i in (0, 1)) for p in xy))
            if xy
            else None
        )
        alt = [s["altitude"] for s in good if s["altitude"] is not None]
        result.append(
            {
                "session_id": session,
                "stream_id": stream,
                "samples": len(samples),
                "valid_fix_samples": len(good),
                "status_counts": dict(Counter(str(s["status"]) for s in samples)),
                "service_values": sorted(
                    {s["service"] for s in samples if s["service"] is not None}
                ),
                "source_span_s": span,
                "average_rate_hz": (len(samples) - 1) / span if span > 0 else None,
                "interval_s": (
                    {"min": min(dt), "median": median(dt), "max": max(dt)}
                    if dt
                    else None
                ),
                "timestamp_regressions": sum(v < 0 for v in dt),
                "repeated_timestamps": sum(v == 0 for v in dt),
                "latitude_range_deg": (
                    [min(s["latitude"] for s in good), max(s["latitude"] for s in good)]
                    if good
                    else None
                ),
                "longitude_range_deg": (
                    [
                        min(s["longitude"] for s in good),
                        max(s["longitude"] for s in good),
                    ]
                    if good
                    else None
                ),
                "altitude_range_m": [min(alt), max(alt)] if alt else None,
                "horizontal_scatter_rms_m": scatter,
                "interpretable_covariance_samples": sum(
                    covariance_sigma(s) is not None for s in samples
                ),
                "note": GPS_SCATTER_NOTE,
            }
        )
    return result


def plot_gps(events: list[dict[str, Any]], output: Path) -> None:
    samples = [e for e in events if e["type"] == "GpsSample"]
    if not samples:
        raise ValueError("No GPS samples to plot")
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError as exc:
        raise ValueError('Plotting requires: pip install -e ".[plot]"') from exc
    figure = Figure(figsize=(12, 10))
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    good = [s for s in samples if valid_fix(s)]
    origin = good[0] if good else None
    time_origin = samples[0]["received_monotonic_ns"]
    groups = {}
    for s in samples:
        groups.setdefault((s["session_id"], s["stream_id"]), []).append(s)
    for index, rows in enumerate(groups.values()):
        times = [(s["received_monotonic_ns"] - time_origin) / 1e9 for s in rows]
        xy = [
            (
                local_xy(s, origin)
                if origin is not None and valid_fix(s)
                else (float("nan"), float("nan"))
            )
            for s in rows
        ]
        axes[0, 0].plot(
            [p[0] for p in xy], [p[1] for p in xy], ".-", label=f"stream {index+1}"
        )
        axes[0, 1].plot(
            times,
            [
                (
                    s["altitude"]
                    if valid_fix(s) and s["altitude"] is not None
                    else float("nan")
                )
                for s in rows
            ],
            ".-",
        )
        sigma = [covariance_sigma(s) if valid_fix(s) else None for s in rows]
        for i, label in enumerate(("east", "north", "up")):
            axes[1, 0].plot(
                times,
                [v[i] if v else float("nan") for v in sigma],
                ".-",
                label=label if index == 0 else None,
            )
        dt = [
            (b["source_time_ns"] - a["source_time_ns"]) / 1e9
            for a, b in zip(rows, rows[1:])
        ]
        axes[1, 1].plot(times[1:], dt, ".-")
    axes[0, 0].set(xlabel="Approx. east offset (m)", ylabel="Approx. north offset (m)")
    axes[0, 0].set_aspect("equal", adjustable="datalim")
    axes[0, 0].legend()
    axes[0, 1].set(
        xlabel="Laptop receive time (s)", ylabel="Altitude (m; datum unverified)"
    )
    axes[1, 0].set(
        xlabel="Laptop receive time (s)",
        ylabel="sqrt(covariance diagonal) (m; assumed ENU)",
    )
    axes[1, 0].legend()
    axes[1, 1].set(
        xlabel="Laptop receive time (s)", ylabel="Source sample interval (s)"
    )
    for axis in axes.flat:
        axis.grid(alpha=0.3)
    figure.suptitle("GPS capture — covariance-derived scales are not app H/V accuracy")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
