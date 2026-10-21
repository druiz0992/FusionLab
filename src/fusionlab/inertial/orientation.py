"""Gyroscope integration using body-frame increments and Hamilton quaternions."""

from dataclasses import asdict, dataclass
from math import acos, cos, degrees, isfinite, sin, sqrt
from pathlib import Path
from statistics import mean
from typing import Iterable

from sensor_stream import ImuSample
from sensor_stream.models import Quaternion

IDENTITY: Quaternion = (0.0, 0.0, 0.0, 1.0)


def normalize(q: Quaternion) -> Quaternion:
    """Return a unit quaternion in (x, y, z, w) component order."""
    norm = sqrt(sum(x * x for x in q))
    if not isfinite(norm) or norm == 0:
        raise ValueError("Quaternion must be finite and nonzero")
    return tuple(x / norm for x in q)  # type: ignore[return-value]


def conjugate(q: Quaternion) -> Quaternion:
    """Conjugate a quaternion; for unit quaternions this inverts the rotation."""
    x, y, z, w = q
    return (-x, -y, -z, w)


def multiply(a: Quaternion, b: Quaternion) -> Quaternion:
    """Hamilton product a ⊗ b: apply rotation b first, then rotation a."""
    x, y, z, w = a
    X, Y, Z, W = b
    return (
        w * X + x * W + y * Z - z * Y,
        w * Y - x * Z + y * W + z * X,
        w * Z + x * Y - y * X + z * W,
        w * W - x * X - y * Y - z * Z,
    )


def angle(a: Quaternion, b: Quaternion) -> float:
    # q and -q represent the same rotation, so use the absolute dot product.
    dot = abs(sum(x * y for x, y in zip(normalize(a), normalize(b))))
    # Clamp roundoff above 1 before acos; the quaternion angle is half the rotation.
    return degrees(2 * acos(min(1.0, dot)))


@dataclass(frozen=True)
class OrientationEstimate:
    session_id: str
    stream_id: str
    segment: int
    source_time_ns: int
    received_monotonic_ns: int
    orientation: Quaternion
    reference_orientation: Quaternion | None
    error_deg: float | None


def integrate(
    samples: Iterable[ImuSample],
    *,
    max_gap_s: float = 0.1,
    inverse_reference: bool = False,
) -> list[OrientationEstimate]:
    """Midpoint gyro integration in rad/s; compare relative rotations.

    The integrated quaternion maps body to initial body frame. Reference is assumed body-to-world unless
    inverse_reference is selected. A gap/regression/reconnect starts a fresh
    segment; do not bridge unknown motion. No bias estimation is applied.
    """
    if not isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError("Maximum gap must be finite and positive")
    # Each session/stream keeps its own previous sample, integrated attitude,
    # initial phone attitude and segment ID; interleaved streams stay independent.
    states: dict[
        tuple[str, str], tuple[ImuSample, Quaternion, Quaternion | None, int]
    ] = {}
    results = []
    segment_count = 0
    for sample in samples:
        if not all(isfinite(v) for v in sample.angular_velocity):
            raise ValueError("Gyroscope values must be finite")
        key = (sample.session_id, sample.stream_id)
        state = states.get(key)
        reference = (
            normalize(sample.orientation) if sample.orientation is not None else None
        )
        if reference is not None and inverse_reference:
            reference = conjugate(reference)
        dt_s = (sample.source_time_ns - state[0].source_time_ns) / 1e9 if state else 0
        # Unknown motion across gaps cannot be integrated safely. Restart the
        # relative comparison rather than joining disconnected trajectory pieces.
        if state is None or dt_s <= 0 or dt_s > max_gap_s:
            segment_count += 1
            orientation, initial_reference, segment = IDENTITY, reference, segment_count
        else:
            previous, orientation, initial_reference, segment = state
            # Midpoint angular velocity gives a body-frame rotation vector.
            rotation = tuple(
                (a + b) * 0.5 * dt_s
                for a, b in zip(previous.angular_velocity, sample.angular_velocity)
            )
            rotation_angle = sqrt(sum(x * x for x in rotation))
            # Quaternion exponential; sin(theta/2)/theta tends to 1/2 at zero.
            quaternion_scale = (
                sin(rotation_angle / 2) / rotation_angle
                if rotation_angle > 1e-12
                else 0.5
            )
            delta_orientation = (
                rotation[0] * quaternion_scale,
                rotation[1] * quaternion_scale,
                rotation[2] * quaternion_scale,
                cos(rotation_angle / 2),
            )
            # Right multiplication composes a body-frame increment.
            orientation = normalize(multiply(orientation, delta_orientation))
        # Remove the initial reference attitude so both curves start at identity.
        relative = (
            normalize(multiply(conjugate(initial_reference), reference))
            if initial_reference is not None and reference is not None
            else None
        )
        results.append(
            OrientationEstimate(
                sample.session_id,
                sample.stream_id,
                segment,
                sample.source_time_ns,
                sample.received_monotonic_ns,
                orientation,
                relative,
                angle(orientation, relative) if relative is not None else None,
            )
        )
        # Carry this state into the next sample from the same session and stream.
        states[key] = (sample, orientation, initial_reference, segment)
    return results


def summary(estimates: list[OrientationEstimate]) -> dict:
    # Only samples with a usable phone reference contribute to error statistics.
    errors = [e.error_deg for e in estimates if e.error_deg is not None]
    return {
        "samples": len(estimates),
        "segments": len({e.segment for e in estimates}),
        "compared_samples": len(errors),
        "rms_error_deg": sqrt(mean(e * e for e in errors)) if errors else None,
        "max_error_deg": max(errors) if errors else None,
        "last_compared_error_deg": errors[-1] if errors else None,
        "note": "Phone orientation is a comparison estimate, not ground truth; gyro assumed rad/s.",
    }


def plot(estimates: list[OrientationEstimate], output: Path) -> None:
    if not estimates:
        raise ValueError("No IMU samples to plot")
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError as exc:
        raise ValueError('Install plotting support: pip install -e ".[plot]"') from exc
    figure = Figure(figsize=(11, 7))
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 1, sharex=True)
    origin = estimates[0].received_monotonic_ns
    # Draw segments separately so plots never connect across integration resets.
    for segment in sorted({e.segment for e in estimates}):
        rows = [e for e in estimates if e.segment == segment]
        t = [(e.received_monotonic_ns - origin) / 1e9 for e in rows]
        axes[0].plot(
            t,
            [angle(IDENTITY, e.orientation) for e in rows],
            color="tab:blue",
            label="gyro" if segment == 1 else None,
        )
        axes[0].plot(
            t,
            [
                (
                    angle(IDENTITY, e.reference_orientation)
                    if e.reference_orientation is not None
                    else float("nan")
                )
                for e in rows
            ],
            color="tab:orange",
            label="phone" if segment == 1 else None,
        )
        axes[1].plot(
            t,
            [e.error_deg if e.error_deg is not None else float("nan") for e in rows],
            color="tab:red",
        )
    axes[0].set_ylabel("Change from segment start (°)\nunsigned, 0–180°")
    axes[1].set_ylabel("Relative orientation difference (°)")
    axes[1].set_xlabel("Recorded laptop receive time (s)")
    axes[0].legend()
    for axis in axes:
        axis.grid(alpha=0.3)
    figure.suptitle("Gyroscope integration vs device-provided orientation")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
