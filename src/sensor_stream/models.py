"""Wire-valued measurements: units, axes and gravity semantics are unverified."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]  # x, y, z, w


@dataclass(frozen=True, slots=True)
class StreamDescriptor:
    stream_id: str
    session_id: str
    frame_id: str
    kind: Literal["imu"] = "imu"
    acceleration_units: str = "unverified"
    angular_velocity_units: str = "unverified"
    measurement_semantics: str = "device-provided; gravity and axes unverified"
    source_clock_id: str | None = None


@dataclass(frozen=True, slots=True)
class StreamStarted:
    descriptor: (
        StreamDescriptor
        | GpsStreamDescriptor
        | CameraStreamDescriptor
        | DepthStreamDescriptor
    )


@dataclass(frozen=True, slots=True)
class ImuSample:
    stream_id: str
    session_id: str
    source_time_ns: int  # Driver interprets wire integer as ns; epoch unknown.
    received_monotonic_ns: int  # Time final bytes were read, before decoding.
    received_unix_ns: int
    acceleration: Vector3
    angular_velocity: Vector3
    orientation: Quaternion | None
    source_sequence: int | None = None


@dataclass(frozen=True, slots=True)
class SourceDiscontinuity:
    session_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class GpsStreamDescriptor:
    stream_id: str
    session_id: str
    frame_id: str
    kind: Literal["gps"] = "gps"
    coordinate_units: str = "degrees (protocol)"
    altitude_units: str = "metres (protocol)"
    altitude_reference: str = (
        "WGS84 ellipsoid declared by protocol; producer unverified"
    )
    covariance_units: str = (
        "m^2 (protocol); ENU convention assumed, producer unverified"
    )


@dataclass(frozen=True, slots=True)
class GpsSample:
    stream_id: str
    session_id: str
    source_time_ns: int
    received_monotonic_ns: int
    received_unix_ns: int
    latitude: float | None
    longitude: float | None
    altitude: float | None
    status: int | None
    service: int | None
    position_covariance: tuple[float | None, ...]
    position_covariance_type: int
    source_sequence: int | None = None


@dataclass(frozen=True, slots=True)
class CameraStreamDescriptor:
    stream_id: str
    session_id: str
    frame_id: str
    kind: Literal["camera"] = "camera"
    timing_semantics: str = "source ns per companion driver; exposure time unverified"
    calibration: str = "intrinsics, distortion and IMU extrinsics absent from protocol"


@dataclass(frozen=True, slots=True)
class CameraSample:
    stream_id: str
    session_id: str
    source_time_ns: int
    received_monotonic_ns: int
    received_unix_ns: int
    width: int
    height: int
    encoding: str
    image_data: bytes
    source_sequence: int | None = None


@dataclass(frozen=True, slots=True)
class DepthStreamDescriptor:
    stream_id: str
    session_id: str
    frame_id: str
    kind: Literal["depth"] = "depth"
    value_units: str = "unverified raw values; no metric conversion applied"
    byte_order: str = "little-endian assumed from companion driver; producer unverified"
    calibration: str = "intrinsics, distortion and RGB alignment absent from protocol"


@dataclass(frozen=True, slots=True)
class DepthSample:
    stream_id: str
    session_id: str
    source_time_ns: int
    received_monotonic_ns: int
    received_unix_ns: int
    width: int
    height: int
    encoding: str
    depth_data: bytes
    source_sequence: int | None = None


# The event stream includes lifecycle markers as well as measurements, so
# recorders and processors can recognize stream starts and disconnects.
SensorEvent = (
    StreamStarted
    | ImuSample
    | GpsSample
    | CameraSample
    | DepthSample
    | SourceDiscontinuity
)
