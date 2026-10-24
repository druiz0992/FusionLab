"""Renderer-independent v1 contracts. No FusionLab or SensorStream imports."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class RunSpec:
    id: str
    label: str


@dataclass(frozen=True, slots=True)
class FrameSpec:
    id: str
    axes: str
    units: str = "m"


@dataclass(frozen=True, slots=True)
# A segment marks continuity; clock_id identifies the clock used by ticks_ns.
class SampleTime:
    segment_id: str
    clock_id: str
    ticks_ns: int


@dataclass(frozen=True, slots=True, kw_only=True)
# Every payload shares an entity path and timestamp through this base class.
class Update:
    entity_id: str
    time: SampleTime


@dataclass(frozen=True, slots=True, kw_only=True)
class ScalarSample(Update):
    value: float
    units: str


@dataclass(frozen=True, slots=True, kw_only=True)
class OrientationSample(Update):
    frame_id: str
    quaternion_xyzw: tuple[float, float, float, float]
    role: Literal["measurement", "estimate", "reference", "ground_truth"] = "estimate"


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageSample(Update):
    width: int
    height: int
    encoding: str
    data: bytes


@dataclass(frozen=True, slots=True, kw_only=True)
class DepthSample(Update):
    width: int
    height: int
    encoding: Literal["16UC1", "32FC1"]
    data: bytes
    byte_order: Literal["little", "big"]
    # None preserves raw depth values when no metric scale has been established.
    metres_per_unit: float | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class SegmentBoundary(Update):
    reason: str


# Sensor-level messages; the 2D estimate/map messages are added below.
_SensorMessage = (
    ScalarSample | OrientationSample | ImageSample | DepthSample | SegmentBoundary
)


@dataclass(frozen=True, slots=True, kw_only=True)
class Map2D(Update):
    """Wall segments in a declared metric XY frame; each segment has two endpoints."""

    frame_id: str
    walls: tuple[tuple[tuple[float, float], tuple[float, float]], ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class Pose2D(Update):
    frame_id: str
    xy: tuple[float, float]
    heading_rad: float
    role: Literal["estimate", "reference", "ground_truth"] = "estimate"


@dataclass(frozen=True, slots=True, kw_only=True)
class Trajectory2D(Update):
    """Full path snapshot, replacing the previous snapshot for this entity."""

    frame_id: str
    points: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class Particles2D(Update):
    """Full weighted population; headings are radians CCW from +X about +Z."""

    frame_id: str
    poses: tuple[tuple[float, float, float], ...]
    weights: tuple[float, ...]


# Every message type a viewer client accepts.
Message = _SensorMessage | Map2D | Pose2D | Trajectory2D | Particles2D
