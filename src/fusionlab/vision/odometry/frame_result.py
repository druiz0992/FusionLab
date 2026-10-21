"""Typed per-frame visual odometry results (`poses.jsonl`).

Each processed tracking frame yields one `FrameResult`. The fields every consumer
relies on (timing, segment, status, pose) are typed attributes. Everything else
the estimator reports (PnP support, triangulation, keyframe and bundle-adjustment
diagnostics) stays in the ordered `diagnostics` dict, because those details
change with each experimental variant. In JSON they are stored flat, exactly as
the original dict rows were, so existing `poses.jsonl` files still load.

Frames: `world_to_camera` is the 3×4 matrix [R | t] with X_camera = R X_world + t.
The world frame is the segment's first camera; units are arbitrary and
independent for each segment, so poses from different segments must not be
compared or joined.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class TrackingStatus(StrEnum):
    """Estimator state after a frame. Compares equal to the saved strings."""

    INITIALIZING = "initializing"  # waiting for or attempting bootstrap
    TRACKING = "tracking"  # pose estimated against the current map
    TRACKING_LOST = "tracking_lost"  # PnP support failed; the map was cleared


@dataclass(frozen=True)
class FrameResult:
    """Outcome of processing one tracking frame."""

    source_time_ns: int  # phone clock of the camera sample
    received_monotonic_ns: int  # laptop monotonic clock at reception
    segment: int  # VO segment; each has its own world frame and scale
    status: TrackingStatus
    # Estimator details, stored flat in JSON in the order they were produced.
    diagnostics: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None  # why the frame is not tracking, when known
    failure_category: str | None = None  # gate that failed on tracking loss
    camera_center_arbitrary: list[float] | None = None  # -R.T t, segment units
    world_to_camera: list[list[float]] | None = None  # [R | t], 3×4
    frame_index: int | None = None  # row in tracks.jsonl, set by the runner

    # Keys stored as attributes; every other JSON key is a diagnostic.
    _CORE_KEYS = (
        "source_time_ns",
        "received_monotonic_ns",
        "segment",
        "status",
        "reason",
        "failure_category",
        "camera_center_arbitrary",
        "world_to_camera",
        "frame_index",
    )

    @property
    def is_tracking(self) -> bool:
        return self.status == TrackingStatus.TRACKING

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> FrameResult:
        """Parse one `poses.jsonl` row (or the estimator's flat result dict)."""
        return cls(
            source_time_ns=int(data["source_time_ns"]),
            received_monotonic_ns=int(data["received_monotonic_ns"]),
            segment=int(data["segment"]),
            status=TrackingStatus(data["status"]),
            diagnostics={k: v for k, v in data.items() if k not in cls._CORE_KEYS},
            reason=data.get("reason"),
            failure_category=data.get("failure_category"),
            camera_center_arbitrary=data.get("camera_center_arbitrary"),
            world_to_camera=data.get("world_to_camera"),
            frame_index=data.get("frame_index"),
        )

    def to_json(self) -> dict[str, Any]:
        """Serialize flat, in the historical `poses.jsonl` key order."""
        data: dict[str, Any] = {
            "source_time_ns": self.source_time_ns,
            "received_monotonic_ns": self.received_monotonic_ns,
            "segment": self.segment,
            "status": self.status.value,
            **self.diagnostics,
        }
        optional = (
            ("reason", self.reason),
            ("failure_category", self.failure_category),
            ("camera_center_arbitrary", self.camera_center_arbitrary),
            ("world_to_camera", self.world_to_camera),
            ("frame_index", self.frame_index),
        )
        for key, value in optional:
            if value is not None:
                data[key] = value
        return data


def read_frame_results(path: Path) -> list[FrameResult]:
    """Read every row of a `poses.jsonl` file."""
    with Path(path).open() as file:
        return [
            FrameResult.from_json(json.loads(line)) for line in file if line.strip()
        ]
