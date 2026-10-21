"""Viewer replay of saved VO results: video, status signals and segment path.

Presentation only: nothing here feeds back into estimation.
"""

from fusionlab.vision.odometry.frame_result import FrameResult, TrackingStatus
from fusionlab.vision.tracks import TrackFrame


class OdometryReplay:
    """Synchronized video, status, and a resettable segment-local path display.

    The display frame is deliberately a container for the currently active
    segment, not a shared physical frame linking the independent trajectories.
    """

    def __init__(self, viewer, tracks):
        from fusion_viewer import RunSpec, FrameSpec

        self.viewer = viewer
        self.tracks_root = tracks.parent.resolve()
        self.origin = None
        self.active_segment = None
        self.path = []
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
                "Display container: current segment initial-camera XZ; cleared on loss/reset; no alignment between segments",
                units="arbitrary",
            ),
        )

    def publish(self, row: TrackFrame, result: FrameResult, index):
        import cv2
        import numpy as np
        from fusion_viewer import SampleTime, Trajectory2D, ScalarSample, ImageSample

        if self.origin is None:
            self.origin = row.received_monotonic_ns
        stamp = SampleTime(
            "vo_replay", "receive", row.received_monotonic_ns - self.origin
        )
        status = result.status
        segment = result.segment
        diagnostics = result.diagnostics
        if status == TrackingStatus.TRACKING:
            if segment != self.active_segment:
                self.path = [(0.0, 0.0)]
                self.active_segment = segment
            center = result.camera_center_arbitrary
            self.path.append((center[0], center[2]))
        else:
            # Do not leave the last successful pose looking current during loss.
            self.path = []
            self.active_segment = None
        self.viewer.publish(
            self.run,
            Trajectory2D(
                entity_id="active_path",
                time=stamp,
                frame_id="active_segment_xz",
                points=tuple(self.path),
            ),
        )
        for key, value, units in (
            (
                "status",
                {"initializing": 0, "tracking": 1, "tracking_lost": 2}[status],
                "state",
            ),
            ("segment", segment, "segment"),
            ("landmarks", diagnostics.get("landmarks", 0), "landmarks"),
            ("pnp_inliers", diagnostics.get("pnp_inliers", 0), "inliers"),
            ("visible_landmarks", diagnostics.get("visible_landmarks", 0), "landmarks"),
            ("landmarks_added", diagnostics.get("landmarks_added", 0), "landmarks"),
            (
                "landmarks_no_longer_visible",
                diagnostics.get("landmarks_no_longer_visible", 0),
                "landmarks",
            ),
        ):
            self.viewer.publish(
                self.run,
                ScalarSample(
                    entity_id=f"signals/vo/{key}", time=stamp, value=value, units=units
                ),
            )
        # Missing measurements stay absent; zero would imply a perfect fit.
        for key in (
            "pnp_median_candidate_error_px",
            "pnp_median_inlier_error_px",
            "refined_median_inlier_error_px",
        ):
            value = diagnostics.get(key)
            if value is not None:
                self.viewer.publish(
                    self.run,
                    ScalarSample(
                        entity_id=f"signals/vo/{key}",
                        time=stamp,
                        value=value,
                        units="px",
                    ),
                )
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
        # The tracking overlay already has the selected display rotation applied.
        # Add a separate banner so state and segment are readable at every frame.
        banner = np.zeros((90, image.shape[1], 3), dtype=np.uint8)
        color = (80, 220, 80) if status == "tracking" else (80, 150, 255)
        cv2.putText(
            banner,
            f"Frame {index} | {status}",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )
        cv2.putText(
            banner,
            f"Segment {segment} | {diagnostics.get('landmarks',0)} landmarks",
            (8, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            banner,
            "Path resets on loss; arbitrary scale",
            (8, 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 255),
            1,
        )
        image = np.vstack((banner, image))
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
