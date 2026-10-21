import unittest
from fusionlab.vision.calibrated_motion import estimate_motion


class CalibratedMotionTests(unittest.TestCase):
    def setUp(self):
        try:
            import numpy as np
            import cv2
        except ImportError:
            self.skipTest("Optional OpenCV unavailable")
        self.np, self.cv = np, cv2

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
