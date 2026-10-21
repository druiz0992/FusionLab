"""Explicit processing interpretations; recorded wire measurements stay intact."""

from dataclasses import dataclass
from math import isfinite
from typing import Literal

from sensor_stream import ImuSample
from sensor_stream.models import Vector3

STANDARD_GRAVITY_M_S2 = 9.80665


@dataclass(frozen=True, slots=True)
class MeasurementConvention:
    acceleration_unit: Literal["g", "m/s2"]
    angular_velocity_unit: Literal["rad/s"] = "rad/s"
    acceleration_semantics: Literal["gravity_compensated", "unknown"] = (
        "gravity_compensated"
    )
    evidence: str = "Experimental interpretation; producer implementation unverified"

    def __post_init__(self) -> None:
        if self.acceleration_unit not in ("g", "m/s2"):
            raise ValueError("Acceleration unit must be g or m/s2")
        if self.angular_velocity_unit != "rad/s":
            raise ValueError("Only rad/s angular velocity is supported")
        if self.acceleration_semantics not in ("gravity_compensated", "unknown"):
            raise ValueError("Unsupported acceleration semantics")


@dataclass(frozen=True, slots=True)
class ProcessedImuSample:
    raw: ImuSample
    convention: MeasurementConvention
    acceleration_m_s2: Vector3
    angular_velocity_rad_s: Vector3


def convert_imu(
    sample: ImuSample, convention: MeasurementConvention
) -> ProcessedImuSample:
    """Scale values only: no frame transform, bias correction or gravity removal."""
    if not all(isfinite(v) for v in (*sample.acceleration, *sample.angular_velocity)):
        raise ValueError("IMU vectors must be finite")
    # Values expressed in g need standard gravity as a scale factor.
    # Keep the raw sample alongside the result so the interpretation is traceable.
    scale = STANDARD_GRAVITY_M_S2 if convention.acceleration_unit == "g" else 1.0
    x, y, z = sample.acceleration
    return ProcessedImuSample(
        sample, convention, (x * scale, y * scale, z * scale), sample.angular_velocity
    )
