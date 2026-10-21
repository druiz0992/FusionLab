import unittest


class LocalBundleTests(unittest.TestCase):
    def setUp(self):
        try:
            import cv2
            import numpy as np
            import scipy
        except ImportError:
            self.skipTest("Optional bundle adjustment dependencies unavailable")
        self.cv, self.np = cv2, np

    def test_joint_refinement_preserves_anchors_and_improves_geometry(self):
        from fusionlab.vision.odometry.local_bundle_adjustment import optimize_window

        cv, np = self.cv, self.np
        rng = np.random.default_rng(22)
        truth_points = rng.uniform([-1, -0.7, 4], [1, 0.7, 7], (30, 3))
        truth_poses = np.array(
            [
                np.column_stack(
                    (
                        cv.Rodrigues(np.array([0.0, 0.015 * i, 0.0]))[0],
                        [-0.3 * i, 0.0, 0.0],
                    )
                )
                for i in range(4)
            ]
        )
        observations = []
        for camera, pose in enumerate(truth_poses):
            transformed = truth_points @ pose[:, :3].T + pose[:, 3]
            for landmark, point in enumerate(transformed):
                observations.append(
                    (
                        camera,
                        landmark,
                        point[:2] / point[2] + rng.normal(0, 0.1 / 500, 2),
                    )
                )
        poses = truth_poses.copy()
        for i in (2, 3):
            poses[i, :, 3] += np.array([0.025, -0.015, 0.01])
        points = truth_points + rng.normal(0, 0.04, truth_points.shape)
        original_poses, original_points = poses.copy(), points.copy()
        optimized_poses, optimized_points, report = optimize_window(
            poses, points, observations, focal=500.0, max_evaluations=60
        )
        self.assertTrue(report["accepted"])
        self.assertLess(report["final_robust_cost"], report["initial_robust_cost"])
        self.assertTrue(report["positive_depth"])
        np.testing.assert_array_equal(optimized_poses[:2], poses[:2])
        np.testing.assert_array_equal(poses, original_poses)
        np.testing.assert_array_equal(points, original_points)
        self.assertLess(
            np.linalg.norm(optimized_points - truth_points),
            np.linalg.norm(points - truth_points),
        )
        self.assertLess(
            np.linalg.norm(optimized_poses[2:, :, 3] - truth_poses[2:, :, 3]),
            np.linalg.norm(poses[2:, :, 3] - truth_poses[2:, :, 3]),
        )

    def test_degenerate_anchor_baseline_is_rejected(self):
        from fusionlab.vision.odometry.local_bundle_adjustment import optimize_window

        np = self.np
        poses = np.array([np.column_stack((np.eye(3), np.zeros(3))) for _ in range(4)])
        with self.assertRaisesRegex(ValueError, "baseline"):
            optimize_window(
                poses,
                np.array([[0.0, 0.0, 5.0]]),
                [(i, 0, [0.0, 0.0]) for i in range(4)],
                focal=500.0,
            )

    def test_odometry_rejects_bundle_that_degrades_current_support(self):
        from unittest.mock import patch
        from fusionlab.vision.odometry.visual_odometry import MonocularOdometry

        np = self.np
        vo = MonocularOdometry({})
        poses = np.array(
            [np.column_stack((np.eye(3), [-0.2 * i, 0.0, 0.0])) for i in range(4)]
        )
        points = np.array([[i * 0.02, 0.0, 5.0] for i in range(25)])

        def observations(pose):
            camera = points @ pose[:, :3].T + pose[:, 3]
            return {i: p[:2] / p[2] for i, p in enumerate(camera)}

        vo.landmarks = {i: p.copy() for i, p in enumerate(points)}
        vo.pose = poses[-1].copy()
        vo.step = 6
        vo.bundle_window = [
            {"pose": pose.copy(), "points": observations(pose), "step": i}
            for i, pose in enumerate(poses[:3])
        ]
        bad = poses.copy()
        bad[-1, 0, 3] += 10.0
        before = vo.pose.copy()
        with patch(
            "fusionlab.vision.odometry.local_bundle_adjustment.optimize_window",
            return_value=(bad, points, {"accepted": True}),
        ):
            report = vo._adjust_bundle(observations(poses[-1]), 500.0)
        self.assertFalse(report["accepted"])
        self.assertEqual(report["reason"], "current_frame_support_degraded")
        np.testing.assert_array_equal(vo.pose, before)
