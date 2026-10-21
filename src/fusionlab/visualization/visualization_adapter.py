"""SensorStream capture translation; the viewer package knows no sensor types."""

import json
from math import sqrt, isfinite
from pathlib import Path
from hashlib import sha256
from fusion_viewer import (
    RunSpec,
    FrameSpec,
    SampleTime,
    ScalarSample,
    ImageSample,
    DepthSample,
    OrientationSample,
    SegmentBoundary,
)


def export_capture(capture: Path, viewer, *, depth_metres_per_unit=None) -> int:
    run = viewer.register_run(
        RunSpec("capture", f"FusionLab recording: {capture.name}")
    )
    origin = None
    frames = set()
    last = {}
    count = 0
    # Stream event metadata and read one asset at a time. Do not load a capture's
    # decoded images or run sensor-specific analysis on the display path.
    with (capture / "events.jsonl").open() as file:
        for line in file:
            event = json.loads(line)
            kind = event["type"]
            if kind == "StreamStarted":
                continue
            if kind == "SourceDiscontinuity":
                if origin is not None:
                    stamp = SampleTime(
                        event["session_id"], "receive", last.get(event["session_id"], 0)
                    )
                    viewer.publish(
                        run,
                        SegmentBoundary(
                            entity_id="events", time=stamp, reason=event["reason"]
                        ),
                    )
                continue
            if "received_monotonic_ns" not in event:
                continue
            if origin is None:
                origin = event["received_monotonic_ns"]
            # Use laptop receive time for shared playback across phone reconnects;
            # source timestamps are published separately as diagnostics.
            stamp = SampleTime(
                event["session_id"], "receive", event["received_monotonic_ns"] - origin
            )
            last[event["session_id"]] = stamp.ticks_ns

            def scalar(name, value, unit):
                if value is not None and isfinite(value):
                    viewer.publish(
                        run,
                        ScalarSample(
                            entity_id=name, time=stamp, value=value, units=unit
                        ),
                    )

            scalar(
                f"diagnostics/source_time/{kind}",
                event["source_time_ns"] / 1e9,
                "source seconds; epoch unverified",
            )
            if kind == "ImuSample":
                for field, unit in (
                    ("acceleration", "g (app label)"),
                    ("angular_velocity", "rad/s (app label)"),
                ):
                    for axis, value in zip("xyz", event[field]):
                        scalar(f"signals/{field}/{axis}", value, unit)
                q = event.get("orientation")
                if q is not None:
                    norm = sqrt(sum(v * v for v in q))
                    if norm == 0:
                        continue
                    frame = f"reference_{event['session_id']}"
                    if frame not in frames:
                        viewer.register_frame(
                            run,
                            FrameSpec(
                                frame,
                                "Device reference axes unverified; orientation only; each session separate",
                            ),
                        )
                        frames.add(frame)
                    viewer.publish(
                        run,
                        OrientationSample(
                            entity_id="orientation/phone",
                            time=stamp,
                            frame_id=frame,
                            quaternion_xyzw=tuple(v / norm for v in q),
                            role="reference",
                        ),
                    )
            elif kind in ("CameraSample", "DepthSample"):
                path = (capture / event["image_path"]).resolve()
                if not path.is_relative_to(capture.resolve()):
                    raise ValueError("Asset path escapes capture")
                data = path.read_bytes()
                if (
                    len(data) != event["image_size_bytes"]
                    or sha256(data).hexdigest() != event["image_sha256"]
                ):
                    raise ValueError("Asset integrity failure")
                common = dict(
                    time=stamp, width=event["width"], height=event["height"], data=data
                )
                if kind == "CameraSample":
                    encoding = event["encoding"].lower()
                    if encoding not in ("jpeg", "jpg", "image/jpeg"):
                        raise ValueError(f"Unsupported image encoding {encoding}")
                    viewer.publish(
                        run,
                        ImageSample(
                            entity_id="rgb/frame", encoding="image/jpeg", **common
                        ),
                    )
                else:
                    viewer.publish(
                        run,
                        DepthSample(
                            entity_id="depth/frame",
                            encoding=event["encoding"],
                            byte_order="little",
                            metres_per_unit=depth_metres_per_unit,
                            **common,
                        ),
                    )
            elif kind == "GpsSample":
                for key, unit in (
                    ("latitude", "degrees"),
                    ("longitude", "degrees"),
                    ("altitude", "m; datum unverified"),
                    ("status", "code"),
                ):
                    scalar(f"diagnostics/gps/{key}", event.get(key), unit)
            count += 1
    return count


def export_orientation(estimates, viewer) -> int:
    """Compare relative gyro/phone rotations in each integration segment's frame.

    Both start at identity, equivalent to aligning the gyro to the first phone
    quaternion and expressing both in that initial body's coordinate system.
    """
    if not estimates:
        raise ValueError("No IMU estimates to visualize")
    run = viewer.register_run(
        RunSpec(
            "orientation_baseline",
            "Gyro integration vs phone orientation; phone is not ground truth",
        )
    )
    origin = estimates[0].received_monotonic_ns
    frames = set()
    for estimate in estimates:
        frame = f"segment_{estimate.segment}"
        if frame not in frames:
            viewer.register_frame(
                run,
                FrameSpec(
                    frame,
                    "Initial body axes for this segment; relative body-to-start rotation",
                ),
            )
            frames.add(frame)
        stamp = SampleTime(frame, "receive", estimate.received_monotonic_ns - origin)
        viewer.publish(
            run,
            OrientationSample(
                entity_id="orientation/gyro_integrated",
                time=stamp,
                frame_id=frame,
                quaternion_xyzw=estimate.orientation,
                role="estimate",
            ),
        )
        if estimate.reference_orientation is not None:
            viewer.publish(
                run,
                OrientationSample(
                    entity_id="orientation/phone_relative",
                    time=stamp,
                    frame_id=frame,
                    quaternion_xyzw=estimate.reference_orientation,
                    role="reference",
                ),
            )
        if estimate.error_deg is not None:
            viewer.publish(
                run,
                ScalarSample(
                    entity_id="signals/orientation_disagreement_deg",
                    time=stamp,
                    value=estimate.error_deg,
                    units="deg",
                ),
            )
    return len(estimates)


def export_position(result, viewer, *, route_path=None) -> int:
    """Render XY dead reckoning and an optional, separately framed route sketch."""
    from fusion_viewer import Trajectory2D

    if not result.estimates:
        raise ValueError("No position estimates to visualize")
    run = viewer.register_run(
        RunSpec(
            "position_baseline",
            "Experimental IMU position; no endpoint fitting or route alignment",
        )
    )
    origin = result.estimates[0].received_monotonic_ns
    if route_path is not None:
        route = json.loads(Path(route_path).read_text())
        points = tuple(tuple(p) for p in route["waypoints_m"])
        viewer.register_frame(
            run,
            FrameSpec(
                "route_sketch",
                "User route: +X right, +Y up; not aligned to phone reference",
            ),
        )
        viewer.publish(
            run,
            Trajectory2D(
                entity_id="reported_route",
                time=SampleTime("route", "receive", 0),
                frame_id="route_sketch",
                points=points,
            ),
        )
    # Keep a separate trajectory for every integration segment because each
    # segment resets its origin and does not establish continuity with the next.
    paths = {}
    for i, estimate in enumerate(result.estimates):
        frame = f"phone_reference_segment_{estimate.segment}"
        if frame not in paths:
            paths[frame] = []
            viewer.register_frame(
                run,
                FrameSpec(
                    frame,
                    "Phone quaternion reference XY projection; origin reset to zero; heading unaligned to route",
                ),
            )
        stamp = SampleTime(
            f"segment_{estimate.segment}",
            "receive",
            estimate.received_monotonic_ns - origin,
        )
        paths[frame].append(estimate.position_m[:2])
        # Full snapshots at ~5 Hz for 50 Hz IMU, preserving every integrated point.
        last_segment = (
            i == len(result.estimates) - 1
            or result.estimates[i + 1].segment != estimate.segment
        )
        if len(paths[frame]) % 10 == 1 or last_segment:
            viewer.publish(
                run,
                Trajectory2D(
                    entity_id=f"path_segment_{estimate.segment}",
                    time=stamp,
                    frame_id=frame,
                    points=tuple(paths[frame]),
                ),
            )
        for axis, value in zip("xyz", estimate.position_m):
            viewer.publish(
                run,
                ScalarSample(
                    entity_id=f"signals/position_m/{axis}",
                    time=stamp,
                    value=value,
                    units="m",
                ),
            )
        speed = sqrt(sum(v * v for v in estimate.velocity_m_s))
        viewer.publish(
            run,
            ScalarSample(
                entity_id="signals/speed_m_s", time=stamp, value=speed, units="m/s"
            ),
        )
    return len(result.estimates)
