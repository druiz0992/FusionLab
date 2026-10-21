"""fusionlab results exported through the fusion_viewer API."""

import math
import unittest
from dataclasses import replace

from fusionlab.inertial.orientation import integrate, angle, IDENTITY
from fusionlab.inertial.dead_reckoning import integrate_position
from fusionlab.inertial.measurements import MeasurementConvention
from fusionlab.vision.odometry.frame_result import FrameResult, TrackingStatus
from fusionlab.vision.tracks import TrackFrame

from support.imu import sample
from support import require


def measurement(t, acceleration=(1.0, 0.0, 0.0), orientation=(0.0, 0.0, 0.0, 1.0)):
    return replace(
        sample(t, gyro=(0.0, 0.0, 0.0), orientation=orientation),
        acceleration=acceleration,
    )


class OrientationViewerTests(unittest.TestCase):
    def test_comparison_export_alignment_and_segments(self):
        import tempfile
        from pathlib import Path
        from fusion_viewer import ViewerClient, OrientationSample, ScalarSample
        from fusion_viewer.recording import LogSink, read_log
        from fusionlab.visualization.visualization_adapter import export_orientation

        initial = (math.sin(0.3), 0.0, 0.0, math.cos(0.3))
        estimates = integrate(
            [
                sample(0, orientation=initial),
                sample(0.02, orientation=initial),
                sample(1, orientation=initial),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison"
            with ViewerClient(LogSink(path)) as viewer:
                self.assertEqual(export_orientation(estimates, viewer), 3)
            values = [v for _, v in read_log(path)]
        orientations = [v for v in values if isinstance(v, OrientationSample)]
        self.assertEqual(len(orientations), 6)
        self.assertLess(angle(orientations[0].quaternion_xyzw, IDENTITY), 1e-5)
        self.assertLess(angle(orientations[1].quaternion_xyzw, IDENTITY), 1e-5)
        self.assertEqual(orientations[-1].time.segment_id, "segment_2")
        self.assertEqual(orientations[-1].time.ticks_ns, 1_000_000_000)
        self.assertEqual(len([v for v in values if isinstance(v, ScalarSample)]), 3)


class PositionViewerTests(unittest.TestCase):
    def test_path_export_and_unaligned_route(self):
        import json
        import tempfile
        from pathlib import Path
        from fusion_viewer import ViewerClient, Trajectory2D, FrameSpec
        from fusion_viewer.recording import LogSink, read_log
        from fusionlab.visualization.visualization_adapter import export_position
        from fusionlab.inertial.dead_reckoning import summarize

        result = integrate_position(
            [measurement(i / 100) for i in range(101)], MeasurementConvention("m/s2")
        )
        report = summarize(result)["segments"][0]
        self.assertAlmostEqual(report["horizontal_path_length_m"], 0.5)
        self.assertAlmostEqual(report["horizontal_endpoint_displacement_m"], 0.5)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            route = root / "route.json"
            route.write_text(json.dumps({"waypoints_m": [[0, 0], [0, 2], [1, 2]]}))
            with ViewerClient(LogSink(root / "viewer")) as viewer:
                self.assertEqual(export_position(result, viewer, route_path=route), 101)
            values = [v for _, v in read_log(root / "viewer")]
        paths = [v for v in values if isinstance(v, Trajectory2D)]
        self.assertEqual(paths[0].frame_id, "route_sketch")
        self.assertNotEqual(paths[-1].frame_id, paths[0].frame_id)
        self.assertEqual(len(paths[-1].points), 101)
        self.assertAlmostEqual(paths[-1].points[-1][0], 0.5)
        self.assertEqual(paths[-1].time.ticks_ns, 1_000_000_000)
        self.assertEqual(len([v for v in values if isinstance(v, FrameSpec)]), 2)


class VisualOdometryTests(unittest.TestCase):
    def setUp(self):
        self.np, self.cv = require("numpy", "cv2")

    def test_replay_video_and_path_reset_share_timeline(self):
        import tempfile
        from pathlib import Path
        from fusionlab.visualization.odometry_replay import OdometryReplay
        from fusion_viewer import ViewerClient, Trajectory2D, ImageSample, ScalarSample
        from fusion_viewer.recording import LogSink, read_log

        np, cv = self.np, self.cv
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cv.imwrite(str(root / "frame.jpg"), np.zeros((160, 120, 3), dtype=np.uint8))
            with ViewerClient(LogSink(root / "log")) as viewer:
                replay = OdometryReplay(viewer, root / "tracks.jsonl")
                for i, (status, segment) in enumerate(
                    (
                        ("tracking", 1),
                        ("tracking", 1),
                        ("tracking_lost", 1),
                        ("tracking", 2),
                    )
                ):
                    row = TrackFrame(
                        source_time_ns=i * 100_000_000,
                        received_monotonic_ns=i * 100_000_000,
                        stream_id="s",
                        segment=segment,
                        original_size=(120, 160),
                        tracking_size=(120, 160),
                        pixel_scale_xy=(1.0, 1.0),
                        tracks=(),
                        overlay_path="frame.jpg",
                    )
                    replay.publish(
                        row,
                        FrameResult(
                            source_time_ns=row.source_time_ns,
                            received_monotonic_ns=row.received_monotonic_ns,
                            segment=segment,
                            status=TrackingStatus(status),
                            diagnostics={"landmarks": 25},
                            camera_center_arbitrary=[float(i + 1), 0.0, 0.0],
                        ),
                        i,
                    )
            values = [value for _, value in read_log(root / "log")]
        paths = [v for v in values if isinstance(v, Trajectory2D)]
        self.assertEqual([len(p.points) for p in paths], [2, 3, 0, 2])
        self.assertEqual(len({p.frame_id for p in paths}), 1)
        images = [v for v in values if isinstance(v, ImageSample)]
        self.assertEqual(len(images), 4)
        self.assertEqual(images[0].height, 250)
        self.assertEqual([p.time for p in paths], [v.time for v in images])
        statuses = [
            v.value
            for v in values
            if isinstance(v, ScalarSample) and v.entity_id.endswith("/status")
        ]
        self.assertEqual(statuses, [1, 1, 2, 1])
