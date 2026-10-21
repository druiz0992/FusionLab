"""Offline Shi–Tomasi / pyramidal Lucas–Kanade tracking in image pixels.

`FeatureTracker` follows corners through consecutive grayscale frames and gives
each a persistent ID. `track_capture` replays one camera stream of a recording,
writes `tracks.jsonl` (typed `TrackFrame` rows) plus overlay JPEGs, and can show
them in the viewer. Coordinates are resized, unrotated tracking pixels (+X right,
+Y down); display rotation only affects the rendered overlays.
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any

from fusionlab.vision.tracks import TrackFrame

# Method caveats saved with every report (what the numbers do not show).
TRACKING_SCOPE = (
    "Pixel tracks only; no pose, metric motion or calibrated geometry. RANSAC is "
    "heuristic and can be degenerate."
)

# Pyramidal Lucas–Kanade search window (px) and pyramid depth.
LK_WINDOW_PX = (21, 21)
LK_PYRAMID_LEVELS = 3
# Fundamental-matrix RANSAC needs eight correspondences and uses this confidence.
MIN_FUNDAMENTAL_MATCHES = 8
FUNDAMENTAL_CONFIDENCE = 0.99
# Shi–Tomasi detection: relative corner quality and minimum spacing (px).
CORNER_QUALITY = 0.01
MIN_CORNER_SPACING_PX = 10
DISPLAY_ROTATIONS = (0, 90, 180, 270)
# Overlay colors (BGR): surviving tracks/motion lines, new detections, text.
SURVIVING_COLOR = (0, 220, 0)
NEW_COLOR = (0, 160, 255)
TEXT_COLOR = (255, 255, 255)


@dataclass(frozen=True)
class TrackingConfig:
    """Tracker settings; distances are tracking pixels, time in seconds."""

    max_features: int = 400
    max_width: int = 640  # wider frames are downscaled to this width
    fb_threshold_px: float = 1.0  # forward/backward round-trip limit
    ransac_threshold_px: float = 1.5  # fundamental-matrix RANSAC threshold
    max_gap_s: float = 0.25  # larger source-time gaps reset the tracker

    def validate(self) -> None:
        """Reject settings that make detection or rejection thresholds invalid."""
        if type(self.max_features) is not int or self.max_features < 8:
            raise ValueError("Need at least eight features")
        if type(self.max_width) is not int or self.max_width < 32:
            raise ValueError("Image width must be at least 32")
        for value in (self.fb_threshold_px, self.ransac_threshold_px, self.max_gap_s):
            if not isfinite(value) or value <= 0:
                raise ValueError("Thresholds must be finite and positive")


class FeatureTracker:
    """Sequential grayscale frames; IDs never reused, even across resets.

    Tracks pass LK status, image bounds and forward/backward checks, followed
    by a fundamental-matrix RANSAC when at least eight correspondences exist.
    Geometric filtering can be degenerate for planar or pure-rotation scenes;
    diagnostics explicitly report when it could not be estimated.

    State: `points` is an OpenCV (N, 1, 2) float32 array (None when empty);
    `ids` and `ages` are parallel lists, filtered together with `points`.
    """

    def __init__(self, config: TrackingConfig = TrackingConfig()):
        config.validate()
        try:
            import cv2  # noqa: F401  (checked here; imported where used)
            import numpy  # noqa: F401
        except ImportError as exc:
            raise ValueError(
                'Install tracking support: pip install -e ".[tracking]"'
            ) from exc
        self.config = config
        self.previous = None  # last grayscale frame
        self.points = None
        self.ids: list[int] = []
        self.ages: list[int] = []
        self.next_id = 0
        self.key = None  # stream ID of the last frame
        self.time: int | None = None  # source time of the last frame, ns
        self.segment = 0

    def process(self, gray, *, stream_id: str, source_time_ns: int):
        """Advance one frame; return track diagnostics and accepted motion lines.

        Track coordinates use resized, unrotated pixels. OpenCV stores point
        arrays as (N, 1, 2); the extra dimension is retained for optical flow.
        Returns (row, trails): `row` holds counts and `tracks`; `trails` are
        ((x0, y0), (x1, y1)) motions of tracks that survived this frame.
        """
        import numpy as np

        if gray.ndim != 2 or gray.dtype != np.uint8:
            raise ValueError("Expected uint8 grayscale image")
        reason = self._reset_reason(gray, stream_id, source_time_ns)
        previous_feature_count = 0 if reason else len(self.ids)
        rejected_flow = rejected_geometry = 0
        geometry_status = "not_attempted"
        trails: list[tuple[tuple[float, float], tuple[float, float]]] = []
        if reason:
            self.segment += 1
            self.points = None
            self.ids = []
            self.ages = []
        elif self.points is not None and len(self.points):
            rejected_flow, rejected_geometry, geometry_status, trails = (
                self._track_existing(gray)
            )
        tracked = len(self.ids)
        added = self._replenish(gray, tracked)
        self.previous = gray.copy()
        self.key = stream_id
        self.time = source_time_ns
        return {
            "segment": self.segment,
            "reset_reason": reason,
            "previous_features": previous_feature_count,
            "tracked": tracked,
            "rejected_flow": rejected_flow,
            "rejected_geometry": rejected_geometry,
            "geometry_status": geometry_status,
            "added": added,
            "tracks": self._track_rows(),
        }, trails

    def _reset_reason(self, gray, stream_id: str, source_time_ns: int) -> str | None:
        """Why correspondence history cannot continue into this frame, if so."""
        if self.previous is None:
            return "first_frame"
        if stream_id != self.key:
            return "stream_changed"
        if gray.shape != self.previous.shape:
            return "dimensions_changed"
        if source_time_ns <= self.time:
            return "timestamp_regression_or_repeat"
        if (source_time_ns - self.time) / 1e9 > self.config.max_gap_s:
            return "timestamp_gap"
        return None

    def _track_existing(self, gray):
        """Follow current tracks into `gray`; drop failures, keep identities.

        Returns (rejected_flow, rejected_geometry, geometry_status, trails).
        """
        import numpy as np

        previous_points = self.points
        current_points, accepted_mask = self._flow_survivors(previous_points, gray)
        rejected_flow = int((~accepted_mask).sum())
        rejected_geometry, geometry_status = self._geometric_filter(
            previous_points, current_points, accepted_mask
        )
        accepted_indexes = np.flatnonzero(accepted_mask)
        trails = [
            (
                tuple(map(float, previous_points[j, 0])),
                tuple(map(float, current_points[j, 0])),
            )
            for j in accepted_indexes
        ]
        self.points = (
            current_points[accepted_mask].reshape(-1, 1, 2)
            if len(accepted_indexes)
            else None
        )
        # Filter point coordinates, IDs and ages together so track identity
        # survives rejection without shifting onto another feature.
        self.ids = [self.ids[j] for j in accepted_indexes]
        self.ages = [self.ages[j] + 1 for j in accepted_indexes]
        return rejected_flow, rejected_geometry, geometry_status, trails

    def _flow_survivors(self, previous_points, gray):
        """LK flow forward and back; keep valid, in-bounds round trips.

        Returns (current_points, accepted_mask). A reliable match should return
        close to its original location when tracked backward; the round-trip
        error is measured in tracking pixels.
        """
        import cv2 as cv
        import numpy as np

        current_points, status, _ = cv.calcOpticalFlowPyrLK(
            self.previous,
            gray,
            previous_points,
            None,
            winSize=LK_WINDOW_PX,
            maxLevel=LK_PYRAMID_LEVELS,
        )
        accepted_mask = np.zeros(len(previous_points), dtype=bool)
        if current_points is None or status is None:
            return current_points, accepted_mask
        # Invalid forward results still need safe coordinates for the backward
        # call; their status mask prevents accepting them.
        backward_candidates = current_points.copy()
        backward_candidates[~np.isfinite(backward_candidates)] = 0
        backward_points, backward_status, _ = cv.calcOpticalFlowPyrLK(
            gray,
            self.previous,
            backward_candidates,
            None,
            winSize=LK_WINDOW_PX,
            maxLevel=LK_PYRAMID_LEVELS,
        )
        if backward_points is None or backward_status is None:
            return current_points, accepted_mask
        xy = current_points.reshape(-1, 2)
        accepted_mask = (
            (status.ravel() != 0)
            & (backward_status.ravel() != 0)
            & np.isfinite(xy).all(axis=1)
        )
        round_trip_px = np.linalg.norm(
            (backward_points - previous_points).reshape(-1, 2), axis=1
        )
        accepted_mask &= round_trip_px <= self.config.fb_threshold_px
        accepted_mask &= (
            (xy[:, 0] >= 0)
            & (xy[:, 0] < gray.shape[1])
            & (xy[:, 1] >= 0)
            & (xy[:, 1] < gray.shape[0])
        )
        return current_points, accepted_mask

    def _geometric_filter(self, previous_points, current_points, accepted_mask):
        """Fundamental-matrix RANSAC on flow survivors; updates `accepted_mask`.

        Geometry is a second rejection stage, after round-trip checks. A failed
        model is reported, not treated as proof of consistency. Returns
        (rejected_geometry, geometry_status).
        """
        import cv2 as cv
        import numpy as np

        accepted_indexes = np.flatnonzero(accepted_mask)
        if len(accepted_indexes) < MIN_FUNDAMENTAL_MATCHES:
            return 0, "insufficient_correspondences"
        fundamental, mask = cv.findFundamentalMat(
            previous_points[accepted_indexes],
            current_points[accepted_indexes],
            cv.FM_RANSAC,
            self.config.ransac_threshold_px,
            FUNDAMENTAL_CONFIDENCE,
        )
        if (
            fundamental is None
            or not np.isfinite(fundamental).all()
            or mask is None
            or mask.size != len(accepted_indexes)
        ):
            return 0, "estimation_failed"
        outlier_indexes = accepted_indexes[mask.ravel() == 0]
        accepted_mask[outlier_indexes] = False
        return len(outlier_indexes), "ransac_applied"

    def _replenish(self, gray, tracked: int) -> int:
        """Detect new corners away from existing tracks; returns how many."""
        import cv2 as cv
        import numpy as np

        if tracked >= self.config.max_features:
            return 0
        # Mask existing tracks so replenishment does not duplicate nearby points.
        mask = np.full(gray.shape, 255, dtype=np.uint8)
        if self.points is not None:
            for x, y in self.points.reshape(-1, 2):
                cv.circle(
                    mask,
                    (round(float(x)), round(float(y))),
                    MIN_CORNER_SPACING_PX,
                    0,
                    -1,
                )
        detected = cv.goodFeaturesToTrack(
            gray,
            maxCorners=self.config.max_features - tracked,
            qualityLevel=CORNER_QUALITY,
            minDistance=MIN_CORNER_SPACING_PX,
            mask=mask,
        )
        added = 0 if detected is None else len(detected)
        if added:
            self.points = (
                detected
                if self.points is None
                else np.concatenate((self.points, detected))
            )
            self.ids.extend(range(self.next_id, self.next_id + added))
            self.next_id += added
            self.ages.extend([1] * added)
        return added

    def _track_rows(self) -> list[dict[str, Any]]:
        """Current tracks as `tracks.jsonl` entries (ID, xy, age in frames)."""
        points = self.points.reshape(-1, 2) if self.points is not None else []
        return [
            {"id": identifier, "xy": list(map(float, point)), "age_frames": age}
            for identifier, point, age in zip(self.ids, points, self.ages)
        ]


def track_capture(
    capture: Path,
    output: Path,
    *,
    config: TrackingConfig = TrackingConfig(),
    max_frames: int | None = None,
    stream_id: str | None = None,
    viewer=None,
    viewer_factory=None,
    display_rotation: int = 90,
) -> dict[str, Any]:
    """Replay one camera stream into a new directory of tracks and overlays.

    The optional viewer receives the same overlays and feature-count diagnostics.
    Capture assets are read and checked without modifying the source recording.
    With `stream_id` None, the first camera stream in the recording is used.
    """
    if viewer is not None and viewer_factory is not None:
        raise ValueError("Supply a viewer or viewer factory, not both")
    if display_rotation not in DISPLAY_ROTATIONS:
        raise ValueError("Display rotation must be 0, 90, 180 or 270")
    tracker = FeatureTracker(config)
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")
    output.mkdir(parents=True, exist_ok=False)
    (output / "overlays").mkdir()
    # Reserve the analysis directory before a file-based viewer can create it.
    # ExitStack also closes factory-owned viewers if processing raises an error.
    with ExitStack() as resources:
        if viewer_factory is not None:
            viewer = resources.enter_context(viewer_factory())
        publisher = _TrackingPublisher(viewer) if viewer else None
        selection = _StreamSelection(stream_id)
        count = tracked_total = rejected_total = 0
        with (
            (output / "tracks.jsonl").open("x") as out,
            (capture / "events.jsonl").open() as events,
        ):
            for line in events:
                event = json.loads(line)
                if not selection.accepts(event):
                    continue
                row, overlay_jpeg, overlay_size = _track_event(
                    tracker, capture, event, config, display_rotation, count
                )
                overlay = f"overlays/{count:06d}.jpg"
                (output / overlay).write_bytes(overlay_jpeg)
                row["overlay_path"] = overlay
                # Parsing into the typed record validates the row before writing.
                frame = TrackFrame.from_json(row)
                out.write(json.dumps(frame.to_json(), allow_nan=False) + "\n")
                if publisher:
                    publisher.publish(row, event, overlay_jpeg, overlay_size)
                count += 1
                tracked_total += row["tracked"]
                rejected_total += row["rejected_flow"] + row["rejected_geometry"]
                if max_frames is not None and count >= max_frames:
                    break
        if not count:
            raise ValueError("No camera frames for selected stream")
        report = {
            "frames": count,
            "stream_id": selection.stream_id,
            "segments": tracker.segment,
            "mean_tracked": tracked_total / count,
            "rejected_correspondences": rejected_total,
            "config": asdict(config),
            "display_rotation_clockwise_deg": display_rotation,
            "status": "complete",
            "scope": TRACKING_SCOPE,
        }
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        return report


class _StreamSelection:
    """Accept camera events of one stream: the given one, or the first seen."""

    def __init__(self, stream_id: str | None):
        self.stream_id = stream_id

    def accepts(self, event: dict[str, Any]) -> bool:
        if event["type"] != "CameraSample":
            return False
        if self.stream_id is None:
            self.stream_id = event["stream_id"]
        return event["stream_id"] == self.stream_id


def _track_event(tracker, capture, event, config, display_rotation, count):
    """Track one camera event; return (row, overlay JPEG bytes, overlay (w, h))."""
    import cv2 as cv

    image = _read_verified_image(capture, event)
    height, width = image.shape[:2]
    image = _resize_for_tracking(image, config.max_width)
    row, trails = tracker.process(
        cv.cvtColor(image, cv.COLOR_BGR2GRAY),
        stream_id=event["stream_id"],
        source_time_ns=event["source_time_ns"],
    )
    # Record pixel scales so tracks can be related to stored pixels.
    row.update(
        source_time_ns=event["source_time_ns"],
        received_monotonic_ns=event["received_monotonic_ns"],
        stream_id=event["stream_id"],
        original_size=[width, height],
        tracking_size=[image.shape[1], image.shape[0]],
        pixel_scale_xy=[image.shape[1] / width, image.shape[0] / height],
    )
    row["display_rotation_clockwise_deg"] = display_rotation
    overlay = _draw_overlay(image, trails, row, display_rotation, count)
    ok, encoded = cv.imencode(".jpg", overlay)
    if not ok:
        raise ValueError("Overlay encoding failed")
    return row, encoded.tobytes(), (overlay.shape[1], overlay.shape[0])


def _read_verified_image(capture: Path, event: dict[str, Any]):
    """Decode a camera asset after checking containment, size and SHA-256."""
    import cv2 as cv
    import numpy as np

    target = (capture / event["image_path"]).resolve()
    if not target.is_relative_to(capture.resolve()):
        raise ValueError("Image path escapes capture")
    data = target.read_bytes()
    if (
        len(data) != event["image_size_bytes"]
        or sha256(data).hexdigest() != event["image_sha256"]
    ):
        raise ValueError("Camera asset integrity failure")
    image = cv.imdecode(np.frombuffer(data, dtype=np.uint8), cv.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot decode camera image")
    return image


def _resize_for_tracking(image, max_width: int):
    """Shrink wide images to limit cost; never upscale small frames."""
    import cv2 as cv

    height, width = image.shape[:2]
    scale = min(1.0, max_width / width)
    return cv.resize(
        image,
        (round(width * scale), round(height * scale)),
        interpolation=cv.INTER_AREA,
    )


def _draw_overlay(image, trails, row, display_rotation: int, count: int):
    """Draw motions and tracks on `image`, rotate for display, add a caption.

    Rotation is applied only to the rendered overlay. Serialized coordinates
    remain in the tracking frame, independent of presentation settings.
    """
    import cv2 as cv

    for a, b in trails:
        cv.line(
            image,
            tuple(round(v) for v in a),
            tuple(round(v) for v in b),
            SURVIVING_COLOR,
            1,
        )
    for track in row["tracks"]:
        xy = tuple(round(v) for v in track["xy"])
        color = SURVIVING_COLOR if track["age_frames"] > 1 else NEW_COLOR
        cv.circle(image, xy, 2, color, -1)
    if display_rotation:
        rotations = {
            90: cv.ROTATE_90_CLOCKWISE,
            180: cv.ROTATE_180,
            270: cv.ROTATE_90_COUNTERCLOCKWISE,
        }
        image = cv.rotate(image, rotations[display_rotation])
    cv.putText(
        image,
        f"frame {count} tracked {row['tracked']} new {row['added']}",
        (8, 20),
        cv.FONT_HERSHEY_SIMPLEX,
        0.45,
        TEXT_COLOR,
        1,
    )
    return image


class _TrackingPublisher:
    """Send overlays and per-frame track counts to a viewer on a receive clock."""

    COUNTS = ("tracked", "added", "rejected_flow", "rejected_geometry")

    def __init__(self, viewer):
        from fusion_viewer import RunSpec

        self.viewer = viewer
        self.origin: int | None = None
        self.run = viewer.register_run(
            RunSpec(
                "feature_tracking", "Image feature tracking; no camera pose estimate"
            )
        )

    def publish(self, row, event, overlay_jpeg: bytes, overlay_size) -> None:
        from fusion_viewer import ImageSample, SampleTime, ScalarSample

        if self.origin is None:
            self.origin = event["received_monotonic_ns"]
        stamp = SampleTime(
            f"segment_{row['segment']}",
            "receive",
            event["received_monotonic_ns"] - self.origin,
        )
        width, height = overlay_size
        self.viewer.publish(
            self.run,
            ImageSample(
                entity_id="rgb/tracks",
                time=stamp,
                width=width,
                height=height,
                encoding="image/jpeg",
                data=overlay_jpeg,
            ),
        )
        for key in self.COUNTS:
            self.viewer.publish(
                self.run,
                ScalarSample(
                    entity_id=f"signals/tracking/{key}",
                    time=stamp,
                    value=row[key],
                    units="features",
                ),
            )
