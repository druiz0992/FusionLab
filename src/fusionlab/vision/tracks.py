"""Typed records for feature-tracking output (`tracks.jsonl`).

`feature_tracking.track_capture` writes one `TrackFrame` per camera frame;
two-view geometry, visual odometry and diagnostics read them back. The JSON
layout is unchanged from the original dict rows, so existing files still load.

Pixel coordinates are in the resized, **unrotated** tracking image: +X right,
+Y down, origin at the top-left pixel center. `pixel_scale_xy` converts original
stored pixels to tracking pixels per axis. Display rotation only affects the
overlay JPEG named by `overlay_path`, never the coordinates.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Track:
    """One tracked image patch in one frame.

    `id` follows the same patch across frames within a run and is never reused;
    it is not a recognized 3D landmark.
    """

    id: int
    xy: tuple[float, float]  # tracking pixels
    age_frames: int | None = None  # frames survived, 1 when newly detected

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Track:
        x, y = data["xy"]
        return cls(
            id=int(data["id"]),
            xy=(float(x), float(y)),
            age_frames=data.get("age_frames"),
        )

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {"id": self.id, "xy": list(self.xy)}
        if self.age_frames is not None:
            data["age_frames"] = self.age_frames
        return data


@dataclass(frozen=True)
class TrackFrame:
    """All feature tracks for one camera frame, plus tracker diagnostics."""

    # Timing and identity of the source camera sample.
    source_time_ns: int  # phone clock
    received_monotonic_ns: int  # laptop monotonic clock at reception
    stream_id: str
    # Tracker segment: increments when the tracker resets (gap, stream change...).
    segment: int
    # Image geometry as (width, height) pixels.
    original_size: tuple[int, int]
    tracking_size: tuple[int, int]
    pixel_scale_xy: tuple[float, float]  # tracking / original pixels
    tracks: tuple[Track, ...]
    # Tracker diagnostics; absent in minimal hand-written rows.
    reset_reason: str | None = None
    previous_features: int = 0
    tracked: int = 0
    rejected_flow: int = 0
    rejected_geometry: int = 0
    geometry_status: str | None = None
    added: int = 0
    # Presentation: older track files predate this field.
    display_rotation_clockwise_deg: int | None = None
    overlay_path: str | None = None  # relative to the tracks.jsonl directory

    @property
    def geometry_key(self) -> tuple[str, int, tuple[int, int]]:
        """Frames are geometrically comparable only if this key matches."""
        return (self.stream_id, self.segment, self.tracking_size)

    def positions(self) -> dict[int, tuple[float, float]]:
        """Map track ID to tracking-pixel position."""
        return {track.id: track.xy for track in self.tracks}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> TrackFrame:
        """Parse one `tracks.jsonl` row; reject rows missing required fields."""
        required = (
            "source_time_ns",
            "received_monotonic_ns",
            "stream_id",
            "segment",
            "original_size",
            "tracking_size",
            "pixel_scale_xy",
            "tracks",
        )
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Tracking row missing fields: {', '.join(missing)}")
        return cls(
            source_time_ns=int(data["source_time_ns"]),
            received_monotonic_ns=int(data["received_monotonic_ns"]),
            stream_id=data["stream_id"],
            segment=int(data["segment"]),
            original_size=_pair(data["original_size"], int),
            tracking_size=_pair(data["tracking_size"], int),
            pixel_scale_xy=_pair(data["pixel_scale_xy"], float),
            tracks=tuple(Track.from_json(track) for track in data["tracks"]),
            reset_reason=data.get("reset_reason"),
            previous_features=data.get("previous_features", 0),
            tracked=data.get("tracked", 0),
            rejected_flow=data.get("rejected_flow", 0),
            rejected_geometry=data.get("rejected_geometry", 0),
            geometry_status=data.get("geometry_status"),
            added=data.get("added", 0),
            display_rotation_clockwise_deg=data.get("display_rotation_clockwise_deg"),
            overlay_path=data.get("overlay_path"),
        )

    def to_json(self) -> dict[str, Any]:
        """Serialize in the historical `tracks.jsonl` key order."""
        data: dict[str, Any] = {
            "segment": self.segment,
            "reset_reason": self.reset_reason,
            "previous_features": self.previous_features,
            "tracked": self.tracked,
            "rejected_flow": self.rejected_flow,
            "rejected_geometry": self.rejected_geometry,
            "geometry_status": self.geometry_status,
            "added": self.added,
            "tracks": [track.to_json() for track in self.tracks],
            "source_time_ns": self.source_time_ns,
            "received_monotonic_ns": self.received_monotonic_ns,
            "stream_id": self.stream_id,
            "original_size": list(self.original_size),
            "tracking_size": list(self.tracking_size),
            "pixel_scale_xy": list(self.pixel_scale_xy),
        }
        if self.display_rotation_clockwise_deg is not None:
            data["display_rotation_clockwise_deg"] = self.display_rotation_clockwise_deg
        if self.overlay_path is not None:
            data["overlay_path"] = self.overlay_path
        return data


def _pair(values: Any, kind: type) -> tuple[Any, Any]:
    first, second = values
    return (kind(first), kind(second))


def read_track_frames(path: Path) -> Iterator[tuple[int, TrackFrame]]:
    """Yield `(frame_index, frame)` for every row of a `tracks.jsonl` file.

    The frame index is the zero-based row number, which saved VO results use.
    """
    with Path(path).open() as file:
        for index, line in enumerate(file):
            yield index, TrackFrame.from_json(json.loads(line))


def read_track_frame_pair(
    path: Path, frame_a: int, frame_b: int
) -> tuple[TrackFrame, TrackFrame]:
    """Read two rows by index, stopping once the later one has been read."""
    if frame_a < 0 or frame_b <= frame_a:
        raise ValueError("Frame indices must satisfy 0 <= frame-a < frame-b")
    selected: dict[int, TrackFrame] = {}
    with Path(path).open() as file:
        for index, line in enumerate(file):
            if index in (frame_a, frame_b):
                selected[index] = TrackFrame.from_json(json.loads(line))
            if index >= frame_b:
                break
    if len(selected) != 2:
        raise ValueError("Frame index exceeds tracking output")
    return selected[frame_a], selected[frame_b]
