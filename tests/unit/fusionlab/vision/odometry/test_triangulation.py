"""Two-view triangulation and candidate quality on known geometry."""

import unittest

from fusionlab.vision.odometry.triangulation import (
    TriangulationCounts,
    camera_center,
    candidate_quality,
    triangulate_rays,
)

from support import require


class TriangulationTests(unittest.TestCase):
    def setUp(self):
        (self.np,) = require("numpy")
        require("cv2")
        np = self.np
        self.pose_a = np.column_stack((np.eye(3), np.zeros(3)))
        # Camera b sits one unit to the right: C_b = -R.T t = (1, 0, 0).
        self.pose_b = np.column_stack((np.eye(3), [-1.0, 0.0, 0.0]))

    def rays(self, points, pose):
        camera = points @ pose[:, :3].T + pose[:, 3]
        return camera[:, :2] / camera[:, 2, None]

    def test_exact_rays_recover_points_and_flag_points_behind(self):
        np = self.np
        points = np.array([[0.0, 0.0, 5.0], [0.5, -0.2, 3.0], [0.0, 0.0, -4.0]])
        triangulated, camera_a, camera_b, valid = triangulate_rays(
            self.rays(points, self.pose_a),
            self.rays(points, self.pose_b),
            self.pose_a,
            self.pose_b,
        )
        np.testing.assert_allclose(triangulated[:2], points[:2], atol=1e-9)
        np.testing.assert_allclose(camera_b[:2], points[:2] - [1.0, 0.0, 0.0])
        np.testing.assert_array_equal(valid, [True, True, False])

    def test_quality_reports_parallax_and_worst_view_error(self):
        np = self.np
        point = np.array([0.5, 0.0, 5.0])
        ray_a = self.rays(point[None], self.pose_a)[0]
        ray_b = self.rays(point[None], self.pose_b)[0] + [0.002, 0.0]
        error_px, parallax = candidate_quality(
            point,
            point,
            point - [1.0, 0.0, 0.0],
            ray_a,
            ray_b,
            camera_center(self.pose_a),
            camera_center(self.pose_b),
            500.0,
        )
        self.assertAlmostEqual(error_px, 1.0)
        # Lines of sight at ±0.5 m over 5 m depth: 2 atan(0.1) ≈ 11.42°.
        self.assertAlmostEqual(parallax, np.degrees(2 * np.arctan(0.1)))

    def test_camera_center_and_counts_defaults(self):
        np = self.np
        np.testing.assert_allclose(camera_center(self.pose_b), [1.0, 0.0, 0.0])
        counts = TriangulationCounts(candidates=3)
        self.assertEqual(counts.accepted, 0)
        self.assertEqual(counts.accepted_parallax_deg, [])
