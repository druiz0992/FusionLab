from math import isfinite
import re
from typing import Protocol
from .models import (
    RunSpec,
    FrameSpec,
    ScalarSample,
    OrientationSample,
    ImageSample,
    DepthSample,
    SegmentBoundary,
    Map2D,
    Pose2D,
    Trajectory2D,
    Particles2D,
    Message,
)


class Sink(Protocol):
    """Backend boundary shared by portable logs and interactive renderers."""

    def register_run(self, run: RunSpec) -> None: ...
    def register_frame(self, run_id: str, frame: FrameSpec) -> None: ...
    def publish(self, run_id: str, message: Message) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...


# IDs may contain slash-separated path components, but no empty parts or dots.
def identifier(value: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*", value
    ):
        raise ValueError(f"Invalid identifier: {value!r}")


class ViewerClient:
    """Synchronous v1 client: publish completes sink submission, not rendering.

    No background queue or remote connection is promised by this first version.
    Sink selection keeps renderer dependencies out of producer message types.
    """

    def __init__(self, sink: Sink):
        self.sink = sink
        # Keep declarations locally so messages can be checked before reaching a backend.
        # Frame and entity keys include the run ID to keep different runs independent.
        self.runs = {}
        self.frames = {}
        self.entities = {}
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _open(self):
        if self.closed:
            raise RuntimeError("Viewer client is closed")

    def register_run(self, run: RunSpec) -> str:
        self._open()
        identifier(run.id)
        # Repeating an identical declaration is safe; reusing its ID with new data is not.
        if run.id in self.runs:
            if self.runs[run.id] != run:
                raise ValueError("Conflicting run ID")
            return run.id
        self.sink.register_run(run)
        self.runs[run.id] = run
        return run.id

    def register_frame(self, run: str, frame: FrameSpec) -> None:
        self._open()
        identifier(frame.id)
        if run not in self.runs:
            raise ValueError("Unknown run")
        key = (run, frame.id)
        if key in self.frames and self.frames[key] != frame:
            raise ValueError("Conflicting frame ID")
        if frame.units not in ("m", "arbitrary") or not frame.axes:
            raise ValueError("Declare frame axes and supported units explicitly")
        self.sink.register_frame(run, frame)
        self.frames[key] = frame

    def publish(self, run: str, message: Message) -> None:
        """Validate a message and submit it without changing units or frames."""
        self._open()
        if run not in self.runs:
            raise ValueError("Unknown run")
        if not isinstance(
            message,
            (
                ScalarSample,
                OrientationSample,
                ImageSample,
                DepthSample,
                SegmentBoundary,
                Map2D,
                Pose2D,
                Trajectory2D,
                Particles2D,
            ),
        ):
            raise ValueError("Unsupported message type")
        # Validate shared identity and timing first, then the payload for each message type.
        identifier(message.entity_id)
        identifier(message.time.segment_id)
        identifier(message.time.clock_id)
        if (
            type(message.time.ticks_ns) is not int
            or not -(2**63) <= message.time.ticks_ns < 2**63
        ):
            raise ValueError("Timestamp must be signed int64 nanoseconds")
        if isinstance(message, ScalarSample) and (
            not isfinite(message.value) or not message.units
        ):
            raise ValueError("Scalar needs finite value and units")
        # Orientations require a declared frame and an already normalized quaternion.
        # Reject invalid data here rather than silently altering a producer estimate.
        if isinstance(message, OrientationSample):
            if (run, message.frame_id) not in self.frames:
                raise ValueError("Unknown spatial frame")
            q = message.quaternion_xyzw
            if (
                len(q) != 4
                or not all(isfinite(v) for v in q)
                or abs(sum(v * v for v in q) - 1) > 0.001
            ):
                raise ValueError("Quaternion must have unit norm")
            if message.role not in (
                "measurement",
                "estimate",
                "reference",
                "ground_truth",
            ):
                raise ValueError("Invalid role")
        if isinstance(message, (ImageSample, DepthSample)):
            if (
                message.width <= 0
                or message.height <= 0
                or not isinstance(message.data, bytes)
                or not message.data
            ):
                raise ValueError("Image needs dimensions and immutable bytes")
            if isinstance(message, DepthSample):
                # Depth is a raw pixel array, so dimensions and bytes per pixel fix its size.
                sizes = {"16UC1": 2, "32FC1": 4}
                if message.encoding not in sizes or message.byte_order not in (
                    "little",
                    "big",
                ):
                    raise ValueError("Unsupported depth representation")
                if (
                    len(message.data)
                    != message.width * message.height * sizes[message.encoding]
                ):
                    raise ValueError("Depth size mismatch")
                if message.metres_per_unit is not None and (
                    not isfinite(message.metres_per_unit)
                    or message.metres_per_unit <= 0
                ):
                    raise ValueError("Depth scale must be finite and positive")
        if isinstance(message, (Map2D, Pose2D, Trajectory2D, Particles2D)):
            if (run, message.frame_id) not in self.frames:
                raise ValueError("Unknown spatial frame")

            # Use the same shape/finite-value rules for walls, paths and particle poses.
            # Tuples keep published coordinate collections immutable.
            def vectors(values, size):
                if not isinstance(values, tuple) or any(
                    not isinstance(v, tuple)
                    or len(v) != size
                    or not all(isfinite(x) for x in v)
                    for v in values
                ):
                    raise ValueError(
                        "Coordinates must be immutable finite tuples with correct dimensions"
                    )

            if isinstance(message, Map2D):
                if not isinstance(message.walls, tuple):
                    raise ValueError("Walls must be immutable")
                for wall in message.walls:
                    vectors(wall, 2)
                    if len(wall) != 2:
                        raise ValueError("Wall needs two endpoints")
            elif isinstance(message, Pose2D):
                vectors((message.xy,), 2)
                if not isfinite(message.heading_rad):
                    raise ValueError("Heading must be finite")
                if message.role not in ("estimate", "reference", "ground_truth"):
                    raise ValueError("Invalid pose role")
            elif isinstance(message, Trajectory2D):
                vectors(message.points, 2)
            else:
                vectors(message.poses, 3)
                if not isinstance(message.weights, tuple) or len(
                    message.weights
                ) != len(message.poses):
                    raise ValueError("One immutable weight per particle required")
                if any(not isfinite(w) or w < 0 for w in message.weights):
                    raise ValueError("Weights must be finite and nonnegative")
                if message.weights and (
                    not isfinite(sum(message.weights)) or sum(message.weights) <= 0
                ):
                    raise ValueError("Nonempty population needs positive total weight")
            # A spatial entity keeps its original frame throughout the run; a new frame
            # needs a new entity ID so old and new coordinates cannot be mixed.
            key_frame = (run, message.entity_id, "frame")
            if (
                key_frame in self.entities
                and self.entities[key_frame] != message.frame_id
            ):
                raise ValueError("Entity spatial frame cannot change")
        key = (run, message.entity_id)
        if key in self.entities and self.entities[key] != type(message):
            raise ValueError("Entity type cannot change")
        # Record entity metadata only after the sink accepts the message.
        # A failed submission must not reserve an entity type or spatial frame.
        self.sink.publish(run, message)
        self.entities[key] = type(message)
        if isinstance(message, (Map2D, Pose2D, Trajectory2D, Particles2D)):
            self.entities[(run, message.entity_id, "frame")] = message.frame_id

    def flush(self):
        self._open()
        self.sink.flush()

    def close(self):
        if not self.closed:
            try:
                self.sink.flush()
            # Close the sink even if flushing fails, and prevent further client use.
            finally:
                self.closed = True
                self.sink.close()
