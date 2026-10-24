"""PnP hypotheses, refinement and positive-depth support on known geometry."""

import unittest

from fusionlab.vision.odometry.pnp import (
    positive_depth_support,
    refine_pose_support,
    solve_pnp_ransac,
)

from support import require


class PnpTests(unittest.TestCase):
    def setUp(self):
        self.np, self.cv = require("numpy", "cv2")

    def scene(self, rotation, translation):
        """100 landmarks in front of the camera and their exact rays."""
        np, cv = self.np, self.cv
        objects = np.random.default_rng(3).uniform([-2, -1, 4], [2, 1, 9], (100, 3))
        rays, _ = cv.projectPoints(objects, rotation, translation, np.eye(3), None)
        return objects, rays.reshape(-1, 2)

    def test_ransac_recovers_known_pose_despite_outliers(self):
        np = self.np
        rotation = np.array([0.02, -0.04, 0.01])
        translation = np.array([0.2, -0.03, 0.01])
        objects, rays = self.scene(rotation, translation)
        rays[:20] += 0.05  # 25 px at f = 500: clear outliers
        candidate = solve_pnp_ransac(objects, rays, 500.0)
        self.assertIsNotNone(candidate)
        np.testing.assert_allclose(candidate.rotation.ravel(), rotation, atol=1e-6)
        np.testing.assert_allclose(
            candidate.translation.ravel(), translation, atol=1e-6
        )
        self.assertTrue(set(candidate.inliers.ravel()).isdisjoint(range(20)))
        self.assertEqual(candidate.world_to_camera().shape, (3, 4))
        errors = candidate.reprojection_errors_px(500.0)
        self.assertLess(float(np.median(errors[20:])), 1e-3)

    def test_support_excludes_points_behind_the_camera(self):
        np = self.np
        rotation, translation = np.zeros(3), np.zeros(3)
        objects, rays = self.scene(rotation, translation)
        # Mirroring a point through the camera center keeps its ray exactly.
        objects[:10] *= -1
        support = positive_depth_support(objects, rays, rotation, translation, 500.0)
        self.assertEqual(support.shape, (90, 1))
        self.assertTrue(set(support.ravel()).isdisjoint(range(10)))

    def test_refinement_needs_six_inliers(self):
        np = self.np
        objects, rays = self.scene(np.zeros(3), np.zeros(3))
        for inliers in (None, np.arange(5).reshape(-1, 1)):
            ok, _, _, support = refine_pose_support(
                objects, rays, np.zeros(3), np.zeros(3), inliers, 500.0
            )
            self.assertFalse(ok)
            self.assertEqual(support.shape, (0, 1))

    def test_refinement_improves_known_pose_and_rejects_random_matches(self):
        np, cv = self.np, self.cv
        rng = np.random.default_rng(55)
        objects = rng.uniform([-2, -1, 4], [2, 1, 9], (100, 3))
        truth_rotation = np.array([0.02, -0.04, 0.01])
        truth_translation = np.array([0.2, -0.03, 0.01])
        pixels, _ = cv.projectPoints(
            objects, truth_rotation, truth_translation, np.eye(3), None
        )
        pixels = pixels.reshape(-1, 2) + rng.normal(0, 0.15 / 500, (100, 2))
        seed_rotation = truth_rotation + np.array([0.01, -0.01, 0.01])
        seed_translation = truth_translation + np.array([0.03, -0.02, 0.02])
        rotation_copy = seed_rotation.copy()
        translation_copy = seed_translation.copy()
        before, _ = cv.projectPoints(
            objects, seed_rotation, seed_translation, np.eye(3), None
        )
        ok, rotation, translation, support = refine_pose_support(
            objects,
            pixels,
            seed_rotation,
            seed_translation,
            np.arange(100).reshape(-1, 1),
            500.0,
        )
        self.assertTrue(ok)
        after, _ = cv.projectPoints(objects, rotation, translation, np.eye(3), None)
        self.assertLess(
            np.mean((after.reshape(-1, 2) - pixels) ** 2),
            np.mean((before.reshape(-1, 2) - pixels) ** 2),
        )
        np.testing.assert_allclose(rotation.ravel(), truth_rotation, atol=0.002)
        np.testing.assert_allclose(translation.ravel(), truth_translation, atol=0.005)
        np.testing.assert_array_equal(seed_rotation, rotation_copy)
        np.testing.assert_array_equal(seed_translation, translation_copy)
        self.assertEqual(len(support), 100)
        random_pixels = rng.uniform(-0.5, 0.5, (100, 2))
        _, _, _, bad_support = refine_pose_support(
            objects,
            random_pixels,
            seed_rotation,
            seed_translation,
            np.arange(100).reshape(-1, 1),
            500.0,
        )
        self.assertLess(len(bad_support), 20)
