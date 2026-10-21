"""Offline Shi–Tomasi / pyramidal Lucas–Kanade tracking in image pixels."""

import json
from contextlib import ExitStack
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path

from fusionlab.vision.tracks import TrackFrame


@dataclass(frozen=True)
class TrackingConfig:
    max_features: int = 400
    max_width: int = 640
    fb_threshold_px: float = 1.0
    ransac_threshold_px: float = 1.5
    max_gap_s: float = 0.25

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
    """

    def __init__(self, config=TrackingConfig()):
        config.validate()
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise ValueError(
                'Install tracking support: pip install -e ".[tracking]"'
            ) from exc
        self.cv, self.np, self.config = cv2, np, config
        self.previous = None
        self.points = None
        self.ids = []
        self.ages = []
        self.next_id = 0
        self.key = None
        self.time = None
        self.segment = 0

    def process(self, gray, *, stream_id, source_time_ns):
        """Advance one frame; return track diagnostics and accepted motion lines.

        Track coordinates use resized, unrotated pixels. OpenCV stores point
        arrays as (N, 1, 2); the extra dimension is retained for optical flow.
        """
        cv, np, config = self.cv, self.np, self.config
        if gray.ndim != 2 or gray.dtype != np.uint8:
            raise ValueError("Expected uint8 grayscale image")
        # Reset correspondence history when frames are no longer contiguous.
        reason = None
        if self.previous is None:
            reason = "first_frame"
        elif stream_id != self.key:
            reason = "stream_changed"
        elif gray.shape != self.previous.shape:
            reason = "dimensions_changed"
        elif source_time_ns <= self.time:
            reason = "timestamp_regression_or_repeat"
        elif (source_time_ns - self.time) / 1e9 > config.max_gap_s:
            reason = "timestamp_gap"
        previous_feature_count = 0 if reason else len(self.ids)
        rejected_flow = rejected_geometry = 0
        geometry_status = "not_attempted"
        trails = []
        if reason:
            self.segment += 1
            self.points = None
            self.ids = []
            self.ages = []
        elif self.points is not None and len(self.points):
            previous_points = self.points
            # Predict each existing corner in the new frame using pyramidal optical flow.
            current_points, status, _ = cv.calcOpticalFlowPyrLK(
                self.previous, gray, previous_points, None, winSize=(21, 21), maxLevel=3
            )
            accepted_mask = np.zeros(len(previous_points), dtype=bool)
            if current_points is not None and status is not None:
                # Invalid forward results still need safe coordinates for the
                # backward call; their status mask prevents accepting them.
                backward_candidates = current_points.copy()
                backward_candidates[~np.isfinite(backward_candidates)] = 0
                backward_points, backward_status, _ = cv.calcOpticalFlowPyrLK(
                    gray,
                    self.previous,
                    backward_candidates,
                    None,
                    winSize=(21, 21),
                    maxLevel=3,
                )
                if backward_points is not None and backward_status is not None:
                    xy = current_points.reshape(-1, 2)
                    accepted_mask = (
                        (status.ravel() != 0)
                        & (backward_status.ravel() != 0)
                        & np.isfinite(xy).all(axis=1)
                    )
                    # A reliable match should return close to its original location when
                    # tracked backward; measure that round-trip error in tracking pixels.
                    accepted_mask &= (
                        np.linalg.norm(
                            (backward_points - previous_points).reshape(-1, 2), axis=1
                        )
                        <= config.fb_threshold_px
                    )
                    accepted_mask &= (
                        (xy[:, 0] >= 0)
                        & (xy[:, 0] < gray.shape[1])
                        & (xy[:, 1] >= 0)
                        & (xy[:, 1] < gray.shape[0])
                    )
            rejected_flow = int((~accepted_mask).sum())
            accepted_indexes = np.flatnonzero(accepted_mask)
            # Geometry is a second rejection stage, after round-trip checks.
            # A failed model is reported, not treated as proof of consistency.
            if len(accepted_indexes) >= 8:
                fundamental, mask = cv.findFundamentalMat(
                    previous_points[accepted_indexes],
                    current_points[accepted_indexes],
                    cv.FM_RANSAC,
                    config.ransac_threshold_px,
                    0.99,
                )
                if (
                    fundamental is not None
                    and np.isfinite(fundamental).all()
                    and mask is not None
                    and mask.size == len(accepted_indexes)
                ):
                    outlier_indexes = accepted_indexes[mask.ravel() == 0]
                    accepted_mask[outlier_indexes] = False
                    rejected_geometry = len(outlier_indexes)
                    geometry_status = "ransac_applied"
                else:
                    geometry_status = "estimation_failed"
            else:
                geometry_status = "insufficient_correspondences"
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
            # Filter point coordinates, IDs and ages together so track identity survives
            # rejection without shifting onto another feature.
            self.ids = [self.ids[j] for j in accepted_indexes]
            self.ages = [self.ages[j] + 1 for j in accepted_indexes]
        tracked = len(self.ids)
        # Mask existing tracks so replenishment does not duplicate nearby points.
        mask = np.full(gray.shape, 255, dtype=np.uint8)
        if self.points is not None:
            for x, y in self.points.reshape(-1, 2):
                cv.circle(mask, (round(float(x)), round(float(y))), 10, 0, -1)
        detected = (
            cv.goodFeaturesToTrack(
                gray,
                maxCorners=config.max_features - tracked,
                qualityLevel=0.01,
                minDistance=10,
                mask=mask,
            )
            if tracked < config.max_features
            else None
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
        self.previous = gray.copy()
        self.key = stream_id
        self.time = source_time_ns
        tracks = [
            {"id": identifier, "xy": list(map(float, point)), "age_frames": age}
            for identifier, point, age in zip(
                self.ids,
                self.points.reshape(-1, 2) if self.points is not None else [],
                self.ages,
            )
        ]
        return {
            "segment": self.segment,
            "reset_reason": reason,
            "previous_features": previous_feature_count,
            "tracked": tracked,
            "rejected_flow": rejected_flow,
            "rejected_geometry": rejected_geometry,
            "geometry_status": geometry_status,
            "added": added,
            "tracks": tracks,
        }, trails


def track_capture(
    capture: Path,
    output: Path,
    *,
    config=TrackingConfig(),
    max_frames=None,
    stream_id=None,
    viewer=None,
    viewer_factory=None,
    display_rotation=90,
):
    """Replay one camera stream into a new directory of tracks and overlays.

    The optional viewer receives the same overlays and feature-count diagnostics.
    Capture assets are read and checked without modifying the source recording.
    """
    if viewer is not None and viewer_factory is not None:
        raise ValueError("Supply a viewer or viewer factory, not both")
    if display_rotation not in (0, 90, 180, 270):
        raise ValueError("Display rotation must be 0, 90, 180 or 270")
    tracker = FeatureTracker(config)
    cv, np = tracker.cv, tracker.np
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")
    output.mkdir(parents=True, exist_ok=False)
    (output / "overlays").mkdir()
    # Reserve the analysis directory before a file-based viewer can create it.
    # ExitStack also closes factory-owned viewers if processing raises an error.
    with ExitStack() as resources:
        if viewer_factory is not None:
            viewer = resources.enter_context(viewer_factory())
        count = tracked_total = rejected_total = 0
        selected = stream_id
        origin = None
        if viewer:
            from fusion_viewer import RunSpec, SampleTime, ImageSample, ScalarSample

            run = viewer.register_run(
                RunSpec(
                    "feature_tracking",
                    "Image feature tracking; no camera pose estimate",
                )
            )
        with (
            (output / "tracks.jsonl").open("x") as out,
            (capture / "events.jsonl").open() as events,
        ):
            for line in events:
                event = json.loads(line)
                if event["type"] != "CameraSample":
                    continue
                if selected is None:
                    selected = event["stream_id"]
                if event["stream_id"] != selected:
                    continue
                target = (capture / event["image_path"]).resolve()
                if not target.is_relative_to(capture.resolve()):
                    raise ValueError("Image path escapes capture")
                data = target.read_bytes()
                if (
                    len(data) != event["image_size_bytes"]
                    or sha256(data).hexdigest() != event["image_sha256"]
                ):
                    raise ValueError("Camera asset integrity failure")
                image = cv.imdecode(
                    np.frombuffer(data, dtype=np.uint8), cv.IMREAD_COLOR
                )
                if image is None:
                    raise ValueError("Cannot decode camera image")
                height, width = image.shape[:2]
                # Limit processing cost by shrinking wide images; never upscale small frames.
                # Record the resulting pixel scales so tracks can be related to stored pixels.
                scale = min(1.0, config.max_width / width)
                image = cv.resize(
                    image,
                    (round(width * scale), round(height * scale)),
                    interpolation=cv.INTER_AREA,
                )
                row, trails = tracker.process(
                    cv.cvtColor(image, cv.COLOR_BGR2GRAY),
                    stream_id=event["stream_id"],
                    source_time_ns=event["source_time_ns"],
                )
                row.update(
                    source_time_ns=event["source_time_ns"],
                    received_monotonic_ns=event["received_monotonic_ns"],
                    stream_id=event["stream_id"],
                    original_size=[width, height],
                    tracking_size=[image.shape[1], image.shape[0]],
                    pixel_scale_xy=[image.shape[1] / width, image.shape[0] / height],
                )
                for a, b in trails:
                    cv.line(
                        image,
                        tuple(round(v) for v in a),
                        tuple(round(v) for v in b),
                        (0, 220, 0),
                        1,
                    )
                for track in row["tracks"]:
                    xy = tuple(round(v) for v in track["xy"])
                    cv.circle(
                        image,
                        xy,
                        2,
                        (0, 220, 0) if track["age_frames"] > 1 else (0, 160, 255),
                        -1,
                    )
                # Rotate only the rendered overlay. Serialized coordinates remain
                # in the tracking frame, independent of presentation settings.
                row["display_rotation_clockwise_deg"] = display_rotation
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
                    (255, 255, 255),
                    1,
                )
                ok, encoded = cv.imencode(".jpg", image)
                if not ok:
                    raise ValueError("Overlay encoding failed")
                overlay = f"overlays/{count:06d}.jpg"
                (output / overlay).write_bytes(encoded.tobytes())
                row["overlay_path"] = overlay
                # Parsing into the typed record validates the row before writing.
                frame = TrackFrame.from_json(row)
                out.write(json.dumps(frame.to_json(), allow_nan=False) + "\n")
                if viewer:
                    if origin is None:
                        origin = event["received_monotonic_ns"]
                    stamp = SampleTime(
                        f"segment_{row['segment']}",
                        "receive",
                        event["received_monotonic_ns"] - origin,
                    )
                    viewer.publish(
                        run,
                        ImageSample(
                            entity_id="rgb/tracks",
                            time=stamp,
                            width=image.shape[1],
                            height=image.shape[0],
                            encoding="image/jpeg",
                            data=encoded.tobytes(),
                        ),
                    )
                    for key in (
                        "tracked",
                        "added",
                        "rejected_flow",
                        "rejected_geometry",
                    ):
                        viewer.publish(
                            run,
                            ScalarSample(
                                entity_id=f"signals/tracking/{key}",
                                time=stamp,
                                value=row[key],
                                units="features",
                            ),
                        )
                count += 1
                tracked_total += row["tracked"]
                rejected_total += row["rejected_flow"] + row["rejected_geometry"]
                if max_frames is not None and count >= max_frames:
                    break
        if not count:
            raise ValueError("No camera frames for selected stream")
        from dataclasses import asdict

        report = {
            "frames": count,
            "stream_id": selected,
            "segments": tracker.segment,
            "mean_tracked": tracked_total / count,
            "rejected_correspondences": rejected_total,
            "config": asdict(config),
            "display_rotation_clockwise_deg": display_rotation,
            "status": "complete",
            "scope": "Pixel tracks only; no pose, metric motion or calibrated geometry. RANSAC is heuristic and can be degenerate.",
        }
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
