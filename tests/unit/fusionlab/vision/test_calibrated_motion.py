"""Calibrated two-view motion."""

import unittest

from fusionlab.vision.calibrated_motion import estimate_motion

from support import require


class CalibratedMotionTests(unittest.TestCase):
    def setUp(self):
        self.np, self.cv = require("numpy", "cv2")

    def scene(self, translation):
        np, cv = self.np, self.cv
        points = np.random.default_rng(14).uniform([-2, -1, 4], [2, 1, 8], (150, 3))
        camera = np.array([[500.0, 0, 320], [0, 510, 240], [0, 0, 1]])
        distortion = np.array([0.05, -0.01, 0.001, 0.001, 0])
        rotvec = np.array([0.02, 0.04, 0.01])
        a, _ = cv.projectPoints(points, np.zeros(3), np.zeros(3), camera, distortion)
        b, _ = cv.projectPoints(
            points, rotvec, np.array(translation, dtype=float), camera, distortion
        )
        return (
            a.reshape(-1, 2),
            b.reshape(-1, 2),
            camera,
            distortion,
            cv.Rodrigues(rotvec)[0],
        )

    def test_known_motion_with_distortion_and_outliers(self):
        np = self.np
        a, b, camera, distortion, expected = self.scene([0.4, 0.02, 0.01])
        b[:15] = np.random.default_rng(1).uniform([0, 0], [640, 480], (15, 2))
        report = estimate_motion(a, b, camera, distortion)
        self.assertEqual(report["status"], "accepted_pair")
        np.testing.assert_allclose(report["rotation_b_from_a"], expected, atol=0.01)
        direction = np.array(report["translation_b_from_a_unit"])
        truth = np.array([0.4, 0.02, 0.01])
        truth /= np.linalg.norm(truth)
        self.assertGreater(direction @ truth, 0.99)
        np.testing.assert_allclose(
            report["camera_b_center_in_a_unit"], -expected.T @ direction, atol=0.01
        )

    def test_pure_rotation_rejected(self):
        a, b, k, d, _ = self.scene([0, 0, 0])
        report = estimate_motion(a, b, k, d)
        self.assertEqual(report["status"], "rejected_pair")

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            estimate_motion([], [], self.np.eye(3), [0] * 5)


class AnalyzeMotionTests(unittest.TestCase):
    def setUp(self):
        self.np, _ = require("numpy", "cv2")

    def write_inputs(self, root, calibration):
        import json

        from support.vision import translating_camera_rows

        tracks = root / "tracks.jsonl"
        tracks.write_text(
            "".join(json.dumps(row) + "\n" for row in translating_camera_rows(11))
        )
        calibration_path = root / "calibration.json"
        calibration_path.write_text(json.dumps(calibration))
        return tracks, calibration_path

    def test_known_translation_is_saved_with_tracking_camera(self):
        import json
        import tempfile
        from pathlib import Path

        from fusionlab.vision.calibrated_motion import analyze_motion
        from support.vision import CAMERA_MATRIX, calibration_json

        np = self.np
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tracks, calibration = self.write_inputs(root, calibration_json())
            report = analyze_motion(
                tracks, calibration, root / "motion", frame_a=0, frame_b=10
            )
            saved = json.loads((root / "motion" / "motion.json").read_text())
            self.assertEqual(saved, report)
            with self.assertRaises(FileExistsError):
                analyze_motion(
                    tracks, calibration, root / "motion", frame_a=0, frame_b=10
                )
        self.assertEqual(report["status"], "accepted_pair")
        # The camera moved 2.5 units along +X; only the direction is observable.
        np.testing.assert_allclose(
            report["camera_b_center_in_a_unit"], [1.0, 0.0, 0.0], atol=1e-3
        )
        np.testing.assert_allclose(report["tracking_camera_matrix"], CAMERA_MATRIX)
        self.assertAlmostEqual(report["source_interval_s"], 1.0)
        self.assertEqual(report["calibration_status"], "unknown")

    def test_calibration_for_another_image_size_is_refused(self):
        import tempfile
        from pathlib import Path

        from fusionlab.vision.calibrated_motion import analyze_motion
        from support.vision import calibration_json

        calibration = {**calibration_json(), "image_size": [1280, 960]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tracks, calibration_path = self.write_inputs(root, calibration)
            with self.assertRaisesRegex(ValueError, "image dimensions"):
                analyze_motion(
                    tracks, calibration_path, root / "motion", frame_a=0, frame_b=10
                )
            self.assertFalse((root / "motion").exists())
