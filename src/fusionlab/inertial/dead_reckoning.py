"""Exploratory integration of gravity-compensated acceleration, not SLAM."""

from dataclasses import dataclass
from math import isfinite, sqrt
from pathlib import Path
from statistics import mean
from typing import Iterable

from sensor_stream import ImuSample
from sensor_stream.models import Vector3
from fusionlab.inertial.measurements import MeasurementConvention, convert_imu
from fusionlab.inertial.orientation import conjugate, multiply, normalize

ZERO: Vector3 = (0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class PositionEstimate:
    session_id: str
    stream_id: str
    segment: int
    source_time_ns: int
    received_monotonic_ns: int
    acceleration_reference_m_s2: Vector3
    velocity_m_s: Vector3
    position_m: Vector3
    stationary: bool = False
    pre_update_velocity_m_s: Vector3 = ZERO


@dataclass(frozen=True, slots=True)
class DeadReckoningResult:
    estimates: list[PositionEstimate]
    bias_body_m_s2: Vector3
    bias_samples: int
    acceleration_unit: str
    bias_interval_s: tuple[float, float] | None
    stationary_intervals_s: tuple[tuple[float, float], ...] = ()


def integrate_position(
    samples: Iterable[ImuSample],
    convention: MeasurementConvention,
    *,
    bias_interval_s: tuple[float, float] | None = None,
    max_gap_s: float = 0.1,
    stationary_intervals_s: tuple[tuple[float, float], ...] = (),
) -> DeadReckoningResult:
    """Integrate phone linear acceleration using its body-to-reference attitude.

    Each continuous stream segment starts at zero position and velocity. Bias
    subtraction is in body coordinates; integration is in reference coordinates.
    Stationary intervals are explicit constraints supplied by the caller, not
    automatic detections. No route fitting or gravity subtraction is performed.
    """
    if convention.acceleration_semantics != "gravity_compensated":
        raise ValueError(
            "Position experiment requires gravity-compensated acceleration"
        )
    if not isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError("max_gap_s must be finite and positive")
    rows = list(samples)
    # Convert acceleration to SI once, before estimating bias or integrating motion.
    vectors = [convert_imu(row, convention).acceleration_m_s2 for row in rows]
    if stationary_intervals_s:
        if not rows or len({(r.session_id, r.stream_id) for r in rows}) != 1:
            raise ValueError(
                "Stationary intervals currently require exactly one stream/session"
            )
        previous_end = -1.0
        for start, end in stationary_intervals_s:
            if (
                not all(isfinite(x) for x in (start, end))
                or not 0 <= start < end
                or start < previous_end
            ):
                raise ValueError(
                    "Stationary intervals must be ordered, nonoverlapping, finite and positive in duration"
                )
            count = sum(
                start <= (r.source_time_ns - rows[0].source_time_ns) / 1e9 < end
                for r in rows
            )
            if count < 2:
                raise ValueError(
                    "Each stationary interval requires at least two samples"
                )
            previous_end = end
    bias = ZERO
    selected = []
    # A selected stationary interval estimates a constant body-frame offset.
    if bias_interval_s is not None:
        start, end = bias_interval_s
        if not all(isfinite(v) for v in (start, end)) or not 0 <= start < end:
            raise ValueError("Bias interval must satisfy 0 <= start < end")
        if not rows:
            raise ValueError("Cannot estimate bias from an empty capture")
        first = rows[0]
        selected = [
            v
            for row, v in zip(rows, vectors)
            if (row.session_id, row.stream_id) == (first.session_id, first.stream_id)
            and start <= (row.source_time_ns - first.source_time_ns) / 1e9 < end
        ]
        if len(selected) < 2:
            raise ValueError(
                "Bias interval requires at least two samples from the first stream"
            )
        bias = tuple(mean(v[i] for v in selected) for i in range(3))
    states = {}
    estimates = []
    segment = 0
    for row, vector in zip(rows, vectors):
        elapsed = (row.source_time_ns - rows[0].source_time_ns) / 1e9
        stationary = any(
            start <= elapsed < end for start, end in stationary_intervals_s
        )
        if row.orientation is None:
            raise ValueError(
                "Position experiment requires phone orientation on every sample"
            )
        q = normalize(row.orientation)
        body = tuple(a - b for a, b in zip(vector, bias))
        # Embed acceleration as a pure quaternion: q * (a, 0) * q^-1.
        # The input is already gravity compensated under this convention.
        rotated = multiply(multiply(q, (*body, 0.0)), conjugate(q))[:3]
        key = (row.session_id, row.stream_id)
        previous = states.get(key)
        dt = (row.source_time_ns - previous.source_time_ns) / 1e9 if previous else 0
        # Unknown motion across a gap cannot be reconstructed; restart position
        # and velocity rather than carrying an estimate through missing time.
        if previous is None or dt <= 0 or dt > max_gap_s:
            segment += 1
            current_segment, velocity, position = segment, ZERO, ZERO
        else:
            current_segment = previous.segment
            # Trapezoidal integration uses the mean of the two endpoint accelerations.
            velocity = tuple(
                v + 0.5 * (a + b) * dt
                for v, a, b in zip(
                    previous.velocity_m_s, previous.acceleration_reference_m_s2, rotated
                )
            )
            # Constant linearly changing acceleration over the interval.
            position = tuple(
                p + v * dt + (a / 3 + b / 6) * dt * dt
                for p, v, a, b in zip(
                    previous.position_m,
                    previous.velocity_m_s,
                    previous.acceleration_reference_m_s2,
                    rotated,
                )
            )
        # Save the unconstrained velocity for stop-entry diagnostics before applying
        # the caller-supplied zero-velocity constraint.
        pre_update_velocity = velocity
        if stationary:
            velocity = ZERO
            # Hold position through consecutive stationary samples, but retain
            # displacement accumulated while approaching the first stop sample.
            if (
                previous is not None
                and previous.stationary
                and previous.segment == current_segment
            ):
                position = previous.position_m
        estimate = PositionEstimate(
            row.session_id,
            row.stream_id,
            current_segment,
            row.source_time_ns,
            row.received_monotonic_ns,
            rotated,
            velocity,
            position,
            stationary,
            pre_update_velocity,
        )
        estimates.append(estimate)
        states[key] = estimate
    return DeadReckoningResult(
        estimates,
        bias,
        len(selected),
        convention.acceleration_unit,
        bias_interval_s,
        stationary_intervals_s,
    )


def summarize(result: DeadReckoningResult) -> dict:
    """Report per-segment drift and explicit constraints, without judging accuracy."""
    # Overwriting by segment leaves its final estimate for endpoint drift reporting.
    endpoints = {}
    for e in result.estimates:
        endpoints[e.segment] = e
    stationary = []
    transitions = []
    last_stop = None
    origin = result.estimates[0].source_time_ns if result.estimates else 0
    for start, end in result.stationary_intervals_s:
        rows = [
            e
            for e in result.estimates
            if start <= (e.source_time_ns - origin) / 1e9 < end
        ]
        stationary.append(
            {
                "interval_s": (start, end),
                "samples": len(rows),
                "entry_speed_before_update_m_s": sqrt(
                    sum(v * v for v in rows[0].pre_update_velocity_m_s)
                ),
                "mean_reference_acceleration_m_s2": tuple(
                    mean(e.acceleration_reference_m_s2[i] for e in rows)
                    for i in range(3)
                ),
            }
        )
        if last_stop is not None and last_stop.segment == rows[0].segment:
            delta = tuple(
                b - a for a, b in zip(last_stop.position_m, rows[0].position_m)
            )
            transitions.append(
                {
                    "between_intervals_s": (previous_interval, (start, end)),
                    "displacement_vector_m": delta,
                    "displacement_m": sqrt(sum(v * v for v in delta)),
                }
            )
        last_stop = rows[-1]
        previous_interval = (start, end)
    return {
        "samples": len(result.estimates),
        "bias_body_m_s2": result.bias_body_m_s2,
        "bias_samples": result.bias_samples,
        "bias_interval_s": result.bias_interval_s,
        "assumed_acceleration_unit": result.acceleration_unit,
        "stationary_intervals": stationary,
        "between_stops": transitions,
        "segments": [
            {
                "segment": e.segment,
                "session_id": e.session_id,
                "stream_id": e.stream_id,
                "final_velocity_m_s": e.velocity_m_s,
                "final_speed_m_s": sqrt(sum(v * v for v in e.velocity_m_s)),
                "final_position_m": e.position_m,
                "displacement_m": sqrt(sum(v * v for v in e.position_m)),
                "horizontal_endpoint_displacement_m": sqrt(
                    sum(v * v for v in e.position_m[:2])
                ),
                "horizontal_path_length_m": sum(
                    sqrt(
                        sum(
                            (b - a) ** 2
                            for a, b in zip(p.position_m[:2], q.position_m[:2])
                        )
                    )
                    for p, q in zip(result.estimates, result.estimates[1:])
                    if p.segment == q.segment == e.segment
                ),
            }
            for e in endpoints.values()
        ],
        "assumptions": "Gravity-compensated body acceleration; body-to-reference phone quaternion; zero initial velocity per segment. Explicit stationary intervals hard-reset velocity and hold position; no retrospective smoothing or distance fitting. No position ground truth.",
    }


def plot(
    result: DeadReckoningResult, uncorrected: DeadReckoningResult, output: Path
) -> None:
    if not result.estimates:
        raise ValueError("No samples to plot")
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError as exc:
        raise ValueError('Install plotting support: pip install -e ".[plot]"') from exc
    figure = Figure(figsize=(12, 11))
    FigureCanvasAgg(figure)
    axes = figure.subplots(4, 1)
    origin = result.estimates[0].received_monotonic_ns
    for data, style, prefix in (
        (uncorrected, ":", "uncorrected"),
        (result, "-", "selected"),
    ):
        for seg in sorted({e.segment for e in data.estimates}):
            rows = [e for e in data.estimates if e.segment == seg]
            t = [(e.received_monotonic_ns - origin) / 1e9 for e in rows]
            for axis, key in zip(
                axes[:3], ("acceleration_reference_m_s2", "velocity_m_s", "position_m")
            ):
                for i, (name, color) in enumerate(
                    zip("xyz", ("tab:blue", "tab:orange", "tab:green"))
                ):
                    axis.plot(
                        t,
                        [getattr(e, key)[i] for e in rows],
                        style,
                        color=color,
                        label=f"{prefix} {name}" if seg == 1 else None,
                        linewidth=0.9,
                    )
            axes[3].plot(
                [e.position_m[0] for e in rows],
                [e.position_m[1] for e in rows],
                style,
                label=f"{prefix}, segment {seg}",
            )
    for axis, label in zip(
        axes[:3], ("Reference acceleration (m/s²)", "Velocity (m/s)", "Position (m)")
    ):
        axis.set_ylabel(label)
        axis.set_xlabel("Recorded laptop receive time (s)")
    axes[3].set_xlabel("Reference X (m)")
    axes[3].set_ylabel("Reference Y (m)")
    axes[3].set_aspect("equal", adjustable="datalim")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend(fontsize="small", ncol=3)
    figure.suptitle(
        "Exploratory dead reckoning — selected constraints vs unconstrained baseline"
    )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
