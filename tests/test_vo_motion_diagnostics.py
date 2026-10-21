import unittest

from fusionlab.evaluation.vo_motion_diagnostics import (
    predict_constant_velocity,
    spatial_distribution,
    evaluate_pose,
    gyro_interval,
)


class MotionDiagnosticTests(unittest.TestCase):
    def setUp(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("Optional tracking dependencies unavailable")
        self.cv, self.np = cv2, np

    def test_constant_velocity_uses_intervals_and_rotation(self):
        cv, np = self.cv, self.np

        def pose(time):
            r = cv.Rodrigues(np.array([0.0, 0.1 * time, 0.0]))[0]
            return np.column_stack((r, -r @ np.array([time, 0.0, 0.0])))

        history = [
            {"time": 0, "pose": pose(0.0)},
            {"time": 200_000_000, "pose": pose(0.2)},
        ]
        predicted = predict_constant_velocity(history, 500_000_000)
        np.testing.assert_allclose(predicted, pose(0.5), atol=1e-9)
        self.assertIsNone(predict_constant_velocity(history, 1_000_000_000))
        self.assertIsNone(predict_constant_velocity(history[:1], 500_000_000))
        changed = [{**history[0], "segment": 1}, {**history[1], "segment": 2}]
        self.assertIsNone(predict_constant_velocity(changed, 500_000_000))
        self.assertIsNone(predict_constant_velocity(history, 100_000_000))

    def test_identical_landmarks_compare_pose_support_and_spatial_coverage(self):
        np = self.np
        objects = np.array(
            [[-1.0, -1.0, 5.0], [1.0, -1.0, 5.0], [-1.0, 1.0, 5.0], [1.0, 1.0, 5.0]]
        )
        camera = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1.0]])
        normalized = objects[:, :2] / objects[:, 2, None]
        pixels = normalized * 500 + [320, 240]
        snapshot = {"objects": objects, "normalized": normalized, "camera": camera}
        pose = np.column_stack((np.eye(3), np.zeros(3)))
        correct = evaluate_pose(pose, snapshot, pixels, [640, 480])
        wrong = pose.copy()
        wrong[0, 3] = 0.3
        incorrect = evaluate_pose(wrong, snapshot, pixels, [640, 480])
        self.assertEqual(correct["inliers_3px"], 4)
        self.assertEqual(incorrect["inliers_3px"], 0)
        self.assertEqual(correct["median_error_px"], 0.0)
        line = spatial_distribution([[10, 10], [20, 20], [30, 30]], [640, 480])
        self.assertAlmostEqual(line["pca_minor_major_ratio"], 0.0)
        self.assertGreater(
            correct["inlier_spatial"]["hull_fraction"], line["hull_fraction"]
        )

    def test_gyro_bracketing_and_alignment_convention(self):
        cv, np = self.cv, self.np
        samples = [
            {
                "source_time_ns": i * 10_000_000,
                "angular_velocity": [0.0, 0.0, np.pi / 2],
            }
            for i in range(31)
        ]
        visual = cv.Rodrigues(np.array([0.0, 0.0, -np.pi / 10]))[0]
        result = gyro_interval(samples, 50_000_000, 250_000_000, visual_rotation=visual)
        self.assertAlmostEqual(result["gyro_angle_deg"], 18.0, places=6)
        self.assertAlmostEqual(result["angle_magnitude_difference_deg"], 0.0, places=6)
        self.assertNotIn("axis_aware_rotation_error_deg", result)
        alignment = cv.Rodrigues(np.array([np.pi / 2, 0.0, 0.0]))[0]
        result = gyro_interval(
            samples,
            50_000_000,
            250_000_000,
            visual_rotation=alignment @ visual @ alignment.T,
            camera_from_imu=alignment,
        )
        self.assertLess(result["axis_aware_rotation_error_deg"], 1e-5)
        self.assertEqual(gyro_interval(samples, -1, 100)["status"], "no_bracketing_imu")
        self.assertEqual(
            gyro_interval(samples[::10], 10_000_000, 150_000_000)["status"],
            "imu_gap_or_regression",
        )

    def test_offset_and_endpoint_interpolation_on_varying_rate(self):
        np = self.np
        samples = [
            {"source_time_ns": i * 10_000_000, "angular_velocity": [0.0, 0.0, i * 0.01]}
            for i in range(41)
        ]
        report = gyro_interval(samples, 105_000_000, 205_000_000)
        expected = np.degrees((0.205**2 - 0.105**2) / 2)
        self.assertAlmostEqual(report["gyro_angle_deg"], expected, places=6)
        shifted = gyro_interval(
            samples, 105_000_000, 205_000_000, offset_ns=100_000_000
        )
        self.assertAlmostEqual(
            shifted["gyro_angle_deg"], np.degrees((0.305**2 - 0.205**2) / 2), places=6
        )

    def test_initializing_replay_does_not_fabricate_predictions_or_gyro(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from fusionlab.evaluation.vo_motion_diagnostics import diagnose_run
        from fusionlab.vision.odometry.visual_odometry import MonocularOdometry

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            run.mkdir()
            capture = root / "capture"
            capture.mkdir()
            calibration = {
                "image_size": [640, 480],
                "camera_matrix": [[500.0, 0, 320], [0, 500, 240], [0, 0, 1]],
                "distortion_order": ["k1", "k2", "p1", "p2", "k3"],
                "distortion_coefficients": [0.0] * 5,
            }
            (root / "calibration.json").write_text(json.dumps(calibration))
            tracker = MonocularOdometry(calibration)
            rows = [
                {
                    "source_time_ns": i * 100_000_000,
                    "received_monotonic_ns": i * 100_000_000,
                    "stream_id": "s",
                    "segment": 1,
                    "original_size": [640, 480],
                    "tracking_size": [640, 480],
                    "pixel_scale_xy": [1.0, 1.0],
                    "tracks": [],
                }
                for i in range(3)
            ]
            (root / "tracks.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rows)
            )
            results = [
                {**tracker.process(row), "frame_index": i} for i, row in enumerate(rows)
            ]
            (run / "poses.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in results)
            )
            (run / "summary.json").write_text(
                json.dumps(
                    {
                        "tracks": str(root / "tracks.jsonl"),
                        "calibration": str(root / "calibration.json"),
                    }
                )
            )
            (capture / "events.jsonl").write_text("")
            with patch("fusionlab.evaluation.vo_motion_diagnostics.write_findings"):
                summary = diagnose_run(run, capture, root / "report")
            self.assertTrue(summary["reproduced_saved_run"])
            self.assertEqual(summary["frames_with_cv"], 0)
            self.assertEqual(summary["gyro_intervals"], 0)
            self.assertTrue((root / "report/index.html").exists())
