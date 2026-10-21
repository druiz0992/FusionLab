# Public viewer API: producers construct messages and submit them through
# ViewerClient; the chosen sink handles persistence or interactive rendering.
"""Standalone public viewer API; importing it needs no rendering SDK."""

from .api import ViewerClient, Sink
from .models import (
    RunSpec,
    FrameSpec,
    SampleTime,
    ScalarSample,
    OrientationSample,
    ImageSample,
    DepthSample,
    SegmentBoundary,
    Message,
    Map2D,
    Pose2D,
    Trajectory2D,
    Particles2D,
)

__all__ = [
    "ViewerClient",
    "Sink",
    "RunSpec",
    "FrameSpec",
    "SampleTime",
    "ScalarSample",
    "OrientationSample",
    "ImageSample",
    "DepthSample",
    "SegmentBoundary",
    "Message",
    "Map2D",
    "Pose2D",
    "Trajectory2D",
    "Particles2D",
]
