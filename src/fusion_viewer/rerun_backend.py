"""Optional Rerun 0.39 backend. All SDK imports stay behind this boundary."""

from pathlib import Path
from .models import (
    ScalarSample,
    OrientationSample,
    ImageSample,
    DepthSample,
    SegmentBoundary,
    Map2D,
    Pose2D,
    Trajectory2D,
    Particles2D,
)


class RerunSink:
    """Translate validated viewer messages into Rerun entities and timelines."""

    def __init__(
        self,
        *,
        output: Path | None = None,
        spawn: bool = False,
        layout_profile: str = "default",
    ):
        try:
            import rerun as rr
            import rerun.blueprint as rb
            import numpy as np
        except ImportError as exc:
            raise ValueError(
                'Install viewer support: pip install -e ".[viewer]"'
            ) from exc
        if layout_profile not in ("default", "odometry"):
            raise ValueError("Unknown viewer layout profile")
        self.layout_profile = layout_profile
        self.rr, self.np = rr, np
        self.stream = rr.RecordingStream("fusion-viewer")
        self.last = {}
        self.segments = {}
        self.rb = rb
        self.runs = []
        self.localization_frames = set()
        self.frame_units = {}
        self.orientation_entities = set()
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            self.stream.save(output)
        if spawn:
            self.stream.spawn(memory_limit="1GB")
        if not output and not spawn:
            raise ValueError("Select an output recording or viewer window")

    def _layout(self):
        """Build explicit entity queries and isolate unrelated spatial frames."""
        rb = self.rb

        # Rerun queries select a literal subtree with a trailing /**.
        # Expand run IDs explicitly rather than using a wildcard mid-path.
        def contents(group):
            return [f"/runs/{run}/{group}/**" for run in self.runs]

        localization = [
            rb.Spatial2DView(
                name=f"Localization: {run}/{frame} ({self.frame_units.get((run, frame), 'm')})",
                origin=f"/runs/{run}/localization/{frame}",
                contents="$origin/**",
            )
            for run, frame in sorted(self.localization_frames)
        ]
        if self.layout_profile == "odometry":
            # One display panel whose contents reset to the active segment.
            return rb.Blueprint(
                rb.Vertical(
                    rb.Horizontal(
                        rb.Spatial2DView(
                            name="Camera / tracking status", contents=contents("rgb")
                        ),
                        rb.Spatial2DView(
                            name="Active segment XZ — arbitrary units",
                            contents=contents("localization"),
                        ),
                    ),
                    rb.TimeSeriesView(
                        name="Tracking: 0 initializing, 1 tracking, 2 lost",
                        contents=contents("signals"),
                    ),
                ),
                rb.TimePanel(timeline="playback"),
                auto_layout=False,
                auto_views=False,
            )
        orientations = [
            rb.Spatial3DView(
                name=f"{run}: {entity}", contents=f"/runs/{run}/{entity}/**"
            )
            for run, entity in sorted(self.orientation_entities)
        ]
        if not orientations:
            orientations = [
                rb.Spatial3DView(
                    name="Phone orientation (no translation)",
                    contents=contents("orientation"),
                )
            ]
        return rb.Blueprint(
            rb.Vertical(
                *localization,
                rb.Horizontal(
                    rb.Spatial2DView(name="RGB", contents=contents("rgb")),
                    rb.Spatial2DView(
                        name="Depth (raw unless scale supplied)",
                        contents=contents("depth"),
                    ),
                    *orientations,
                ),
                rb.Horizontal(
                    rb.TimeSeriesView(
                        name="Acceleration / gyro", contents=contents("signals")
                    ),
                    rb.TimeSeriesView(
                        name="Timing / GPS", contents=contents("diagnostics")
                    ),
                ),
            ),
            rb.TimePanel(timeline="playback"),
            auto_layout=False,
            auto_views=False,
        )

    def register_run(self, run):
        self.runs.append(run.id)
        self.stream.send_blueprint(self._layout(), make_active=True, make_default=True)
        self.stream.log(
            f"metadata/{run.id}",
            self.rr.TextDocument(
                run.label
                + "\nOrientation is a device estimate; no calibrated camera frustum or depth point cloud."
            ),
            static=True,
        )

    def register_frame(self, run, frame):
        self.frame_units[(run, frame.id)] = frame.units
        self.stream.log(
            f"metadata/{run}/{frame.id}", self.rr.TextDocument(frame.axes), static=True
        )

    def _time(self, time):
        self.stream.set_time(
            "playback", duration=self.np.timedelta64(time.ticks_ns, "ns")
        )

    def publish(self, run, message):
        rr = self.rr
        time = message.time
        if isinstance(message, OrientationSample):
            entity = (run, message.entity_id)
            if entity not in self.orientation_entities:
                self.orientation_entities.add(entity)
                self.stream.send_blueprint(
                    self._layout(), make_active=True, make_default=True
                )
        # Clear stale geometry across reconnects instead of implying continuity.
        previous_segment = self.segments.get(run)
        self._time(time)
        if previous_segment is not None and previous_segment != time.segment_id:
            self.stream.log(f"runs/{run}", rr.Clear(recursive=True))
        self.segments[run] = time.segment_id
        path = f"runs/{run}/{message.entity_id}"
        spatial = isinstance(message, (Map2D, Pose2D, Trajectory2D, Particles2D))
        # Put each metric frame in its own subtree so unrelated XY coordinates
        # appear in separate views instead of sharing a misleading origin.
        if spatial:
            path = f"runs/{run}/localization/{message.frame_id}/{message.entity_id}"
            frame = (run, message.frame_id)
            if frame not in self.localization_frames:
                self.localization_frames.add(frame)
                self.stream.send_blueprint(
                    self._layout(), make_active=True, make_default=True
                )
        key = (run, message.entity_id)
        previous = self.last.get(key)
        # A common display timeline requires the producer to align clocks first.
        if previous and time.clock_id != previous.clock_id:
            raise ValueError("Explicit clock alignment required")
        if previous and time.ticks_ns < previous.ticks_ns:
            raise ValueError("Backend expects monotonically ordered entity timestamps")
        # Expire a dynamic entity after a half-second gap; otherwise Rerun would
        # keep showing its last value until the next sample arrives.
        if (
            previous
            and not isinstance(message, Map2D)
            and time.ticks_ns - previous.ticks_ns > 500_000_000
        ):
            self.stream.set_time(
                "playback",
                duration=self.np.timedelta64(previous.ticks_ns + 500_000_000, "ns"),
            )
            self.stream.log(path, rr.Clear(recursive=True))
            self._time(time)
        self.last[key] = time
        if isinstance(message, Map2D):
            if message.walls:
                self.stream.log(
                    path, rr.LineStrips2D(message.walls, colors=[180, 180, 180])
                )
            else:
                self.stream.log(path, rr.Clear(recursive=True))
        elif isinstance(message, Pose2D):
            from math import cos, sin

            color = {
                "estimate": [255, 170, 30],
                "reference": [70, 180, 255],
                "ground_truth": [80, 230, 100],
            }[message.role]
            self.stream.log(path, rr.Points2D([message.xy], colors=color, radii=0.08))
            self.stream.log(
                path + "/heading",
                rr.Arrows2D(
                    origins=[message.xy],
                    vectors=[
                        [0.4 * cos(message.heading_rad), 0.4 * sin(message.heading_rad)]
                    ],
                    colors=color,
                ),
            )
        elif isinstance(message, Trajectory2D):
            if len(message.points) >= 2:
                self.stream.log(
                    path, rr.LineStrips2D([message.points], colors=[255, 170, 30])
                )
            else:
                self.stream.log(path, rr.Clear(recursive=True))
        elif isinstance(message, Particles2D):
            from math import cos, sin

            if not message.poses:
                self.stream.log(path, rr.Clear(recursive=True))
            else:
                weights = self.np.asarray(message.weights)
                # Relative visual emphasis only; this does not renormalize the
                # producer's probability distribution or change its estimates.
                weights = weights / weights.max()
                colors = self.np.array(
                    [
                        [100, 100 + int(155 * w), 255, 80 + int(175 * w)]
                        for w in weights
                    ],
                    dtype="uint8",
                )
                xy = [p[:2] for p in message.poses]
                self.stream.log(
                    path, rr.Points2D(xy, colors=colors, radii=0.025 + 0.04 * weights)
                )
                self.stream.log(
                    path + "/headings",
                    rr.Arrows2D(
                        origins=xy,
                        vectors=[
                            [0.15 * cos(p[2]), 0.15 * sin(p[2])] for p in message.poses
                        ],
                        colors=colors,
                    ),
                )
        elif isinstance(message, ScalarSample):
            self.stream.log(path, rr.Scalars(message.value))
        elif isinstance(message, ImageSample):
            self.stream.log(
                path,
                rr.EncodedImage(contents=message.data, media_type=message.encoding),
            )
        elif isinstance(message, DepthSample):
            # Decode the declared byte order and pixel type before reshaping to rows.
            # The float copy allows invalid pixels to be hidden with NaN.
            dtype = ("<" if message.byte_order == "little" else ">") + (
                "u2" if message.encoding == "16UC1" else "f4"
            )
            image = (
                self.np.frombuffer(message.data, dtype=dtype)
                .astype("float32")
                .reshape(message.height, message.width)
            )
            image[~self.np.isfinite(image) | (image <= 0)] = self.np.nan
            self.stream.log(
                path,
                rr.DepthImage(
                    image,
                    # Rerun expects raw units per metre; our public API stores metres per unit.
                    meter=(
                        1 / message.metres_per_unit if message.metres_per_unit else None
                    ),
                ),
            )
        elif isinstance(message, OrientationSample):
            self.stream.log(
                path,
                rr.Transform3D(quaternion=rr.Quaternion(xyzw=message.quaternion_xyzw)),
            )
            self.stream.log(
                path + "/axes",
                rr.Arrows3D(
                    origins=[[0, 0, 0]] * 3,
                    vectors=[[0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2]],
                    colors=[[255, 60, 60], [60, 255, 60], [60, 100, 255]],
                    labels=["X", "Y", "Z"],
                ),
            )
        elif isinstance(message, SegmentBoundary):
            self.stream.log(f"runs/{run}", rr.Clear(recursive=True))
            self.stream.log("events", rr.TextLog(message.reason))

    def flush(self):
        self.stream.flush()

    def close(self):
        self.stream.disconnect()
