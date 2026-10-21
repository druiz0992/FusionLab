"""Viewer replay of saved VO results: video, status signals and segment path.

Presentation only: nothing here feeds back into estimation.
"""

from __future__ import annotations

from pathlib import Path

from fusionlab.vision.odometry.frame_result import FrameResult, TrackingStatus
from fusionlab.vision.tracks import TrackFrame

# Numeric codes for the status signal plotted in the viewer.
STATUS_CODES = {
    TrackingStatus.INITIALIZING: 0,
    TrackingStatus.TRACKING: 1,
    TrackingStatus.TRACKING_LOST: 2,
}
# Map counts published every frame (0 when absent): (key, units).
COUNT_SIGNALS = (
    ("landmarks", "landmarks"),
    ("pnp_inliers", "inliers"),
    ("visible_landmarks", "landmarks"),
    ("landmarks_added", "landmarks"),
    ("landmarks_no_longer_visible", "landmarks"),
)
# Pixel errors published only when measured: zero would imply a perfect fit.
ERROR_SIGNALS = (
    "pnp_median_candidate_error_px",
    "pnp_median_inlier_error_px",
    "refined_median_inlier_error_px",
)
BANNER_HEIGHT_PX = 90
# Banner text colors (BGR).
TRACKING_COLOR = (80, 220, 80)
NOT_TRACKING_COLOR = (80, 150, 255)
TEXT_COLOR = (255, 255, 255)


class OdometryReplay:
    """Synchronized video, status, and a resettable segment-local path display.

    The display frame is deliberately a container for the currently active
    segment, not a shared physical frame linking the independent trajectories.
    The path is the camera center's X and Z (segment world frame, arbitrary
    units), starting at the segment's first camera.
    """

    def __init__(self, viewer, tracks: Path):
        from fusion_viewer import FrameSpec, RunSpec

        self.viewer = viewer
        self.tracks_root = tracks.parent.resolve()
        self.origin: int | None = None  # first receive time, ns
        self.active_segment: int | None = None
        self.path: list[tuple[float, float]] = []
        self.run = viewer.register_run(
            RunSpec(
                "monocular_vo",
                "Experimental monocular VO; independent arbitrary scale per segment",
            )
        )
        viewer.register_frame(
            self.run,
            FrameSpec(
                "active_segment_xz",
                "Display container: current segment initial-camera XZ; cleared on "
                "loss/reset; no alignment between segments",
                units="arbitrary",
            ),
        )

    def publish(self, row: TrackFrame, result: FrameResult, index: int) -> None:
        """Publish one frame: path, signals and the overlay with a status banner."""
        stamp = self._stamp(row)
        self._update_path(result)
        self._publish_path(stamp)
        self._publish_signals(stamp, result)
        image = self._with_banner(self._load_overlay(row), result, index)
        self._publish_image(stamp, image)

    def _stamp(self, row: TrackFrame):
        """Replay time: laptop receive time relative to the first frame."""
        from fusion_viewer import SampleTime

        if self.origin is None:
            self.origin = row.received_monotonic_ns
        return SampleTime(
            "vo_replay", "receive", row.received_monotonic_ns - self.origin
        )

    def _update_path(self, result: FrameResult) -> None:
        """Extend the active segment's XZ path, or clear it when not tracking."""
        if result.status == TrackingStatus.TRACKING:
            if result.segment != self.active_segment:
                self.path = [(0.0, 0.0)]
                self.active_segment = result.segment
            center = result.camera_center_arbitrary
            self.path.append((center[0], center[2]))
        else:
            # Do not leave the last successful pose looking current during loss.
            self.path = []
            self.active_segment = None

    def _publish_path(self, stamp) -> None:
        from fusion_viewer import Trajectory2D

        self.viewer.publish(
            self.run,
            Trajectory2D(
                entity_id="active_path",
                time=stamp,
                frame_id="active_segment_xz",
                points=tuple(self.path),
            ),
        )

    def _publish_signals(self, stamp, result: FrameResult) -> None:
        """Status, segment, map counts, and pixel errors when measured."""
        diagnostics = result.diagnostics
        signals = [
            ("status", STATUS_CODES[result.status], "state"),
            ("segment", result.segment, "segment"),
        ]
        signals += [
            (key, diagnostics.get(key, 0), units) for key, units in COUNT_SIGNALS
        ]
        signals += [
            (key, diagnostics[key], "px")
            for key in ERROR_SIGNALS
            if diagnostics.get(key) is not None
        ]
        for key, value, units in signals:
            self._publish_scalar(stamp, key, value, units)

    def _publish_scalar(self, stamp, key: str, value, units: str) -> None:
        from fusion_viewer import ScalarSample

        self.viewer.publish(
            self.run,
            ScalarSample(
                entity_id=f"signals/vo/{key}", time=stamp, value=value, units=units
            ),
        )

    def _load_overlay(self, row: TrackFrame):
        """The tracking overlay JPEG for this row, kept inside the tracks folder."""
        import cv2

        overlay = row.overlay_path
        if not overlay:
            raise ValueError(
                "Tracking row has no overlay_path; regenerate feature tracking overlays"
            )
        path = (self.tracks_root / overlay).resolve()
        if not path.is_relative_to(self.tracks_root):
            raise ValueError("Overlay path escapes tracking directory")
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Missing or unreadable tracking overlay: {path}")
        return image

    def _with_banner(self, image, result: FrameResult, index: int):
        """Stack a status banner above the overlay.

        The tracking overlay already has the selected display rotation applied;
        a separate banner keeps state and segment readable at every frame.
        """
        import cv2
        import numpy as np

        banner = np.zeros((BANNER_HEIGHT_PX, image.shape[1], 3), dtype=np.uint8)
        tracking = result.status == TrackingStatus.TRACKING
        lines = (
            (
                f"Frame {index} | {result.status}",
                22,
                0.45,
                TRACKING_COLOR if tracking else NOT_TRACKING_COLOR,
            ),
            (
                f"Segment {result.segment} | "
                f"{result.diagnostics.get('landmarks', 0)} landmarks",
                44,
                0.45,
                TEXT_COLOR,
            ),
            ("Path resets on loss; arbitrary scale", 66, 0.43, TEXT_COLOR),
        )
        for text, baseline_px, font_scale, color in lines:
            cv2.putText(
                banner,
                text,
                (8, baseline_px),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                1,
            )
        return np.vstack((banner, image))

    def _publish_image(self, stamp, image) -> None:
        import cv2
        from fusion_viewer import ImageSample

        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            raise ValueError("Replay video encoding failed")
        self.viewer.publish(
            self.run,
            ImageSample(
                entity_id="rgb/tracking",
                time=stamp,
                width=image.shape[1],
                height=image.shape[0],
                encoding="image/jpeg",
                data=encoded.tobytes(),
            ),
        )
