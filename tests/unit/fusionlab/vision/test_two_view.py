"""Uncalibrated two-view geometry."""

import unittest

from fusionlab.vision.two_view import estimate_geometry

from support import require


class TwoViewTests(unittest.TestCase):
    def test_projected_scene_and_outliers(self):
        np, cv2 = require("numpy", "cv2")
        rng = np.random.default_rng(12)
        points = rng.uniform([-2, -1, 4], [2, 1, 10], (100, 3))
        camera = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])
        rotation, _ = cv2.Rodrigues(np.array([0.02, 0.08, 0.01]))

        def project(x):
            pixels = x @ camera.T
            return pixels[:, :2] / pixels[:, 2:]

        a = project(points)
        b = project(points @ rotation.T + np.array([0.3, 0.02, 0.05]))
        b[:15] = rng.uniform([0, 0], [640, 480], (15, 2))
        result = estimate_geometry(a, b)
        self.assertGreaterEqual(result["inliers"], 75)
        self.assertLess(result["median_inlier_sampson_px"], 0.1)
        self.assertLess(sum(result["inlier_mask"][:15]), 5)

    def test_invalid_correspondences(self):
        require("cv2")  # availability only
        for a, b in (
            ([], []),
            ([[1, 2]] * 7, [[2, 3]] * 7),
            ([[1, 2]] * 8, [[1, 2]] * 9),
        ):
            with self.assertRaises(ValueError):
                estimate_geometry(a, b)
