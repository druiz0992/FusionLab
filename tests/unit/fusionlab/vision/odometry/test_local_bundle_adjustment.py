"""Bounded local bundle adjustment."""

import unittest

from support import require


class LocalBundleTests(unittest.TestCase):
    def setUp(self):
        self.cv, self.np, _ = require("cv2", "numpy", "scipy")

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
        report = optimize_window(
            poses, points, observations, focal=500.0, max_evaluations=60
        )
        optimized_poses, optimized_points = report.poses, report.points
        self.assertTrue(report.accepted)
        self.assertLess(report.final_robust_cost, report.initial_robust_cost)
        self.assertTrue(report.positive_depth)
        self.assertLessEqual(report.evaluations, 60)
        self.assertEqual(report.summary()["fixed_pose_indices"], [0, 1])
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
        from fusionlab.vision.odometry.local_bundle_adjustment import (
            DegenerateAnchorBaseline,
            optimize_window,
        )

        np = self.np
        poses = np.array([np.column_stack((np.eye(3), np.zeros(3))) for _ in range(4)])
        with self.assertRaisesRegex(DegenerateAnchorBaseline, "baseline"):
            optimize_window(
                poses,
                np.array([[0.0, 0.0, 5.0]]),
                [(i, 0, [0.0, 0.0]) for i in range(4)],
                focal=500.0,
            )

    def test_odometry_rejects_bundle_that_degrades_current_support(self):
        from unittest.mock import patch
        from fusionlab.vision.odometry.local_bundle_adjustment import (
            WindowAdjustment,
        )
        from fusionlab.vision.odometry.visual_odometry import (
            MonocularOdometry,
            _BundleFrame,
        )

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
            _BundleFrame(pose.copy(), observations(pose), i)
            for i, pose in enumerate(poses[:3])
        ]
        bad = poses.copy()
        bad[-1, 0, 3] += 10.0
        before = vo.pose.copy()
        # A window step the solver accepted, but which breaks the current frame.
        lowered_cost = WindowAdjustment(
            bad, points, True, 2.0, 1.0, 1.0, 0.5, True, True, 3
        )
        with patch(
            "fusionlab.vision.odometry.local_bundle_adjustment.optimize_window",
            return_value=lowered_cost,
        ):
            report = vo._adjust_bundle(observations(poses[-1]), 500.0)
        self.assertFalse(report["accepted"])
        self.assertEqual(report["reason"], "current_frame_support_degraded")
        np.testing.assert_array_equal(vo.pose, before)

    def test_adopted_bundle_updates_every_stored_copy_of_window_poses(self):
        from fusionlab.vision.odometry.visual_odometry import (
            MonocularOdometry,
            _BundleFrame,
            _Keyframe,
        )

        np = self.np
        vo = MonocularOdometry({})
        poses = np.array(
            [np.column_stack((np.eye(3), [-0.2 * i, 0.0, 0.0])) for i in range(3)]
        )
        corrected = poses.copy()
        corrected[:, 1, 3] += 0.01
        outside = np.column_stack((np.eye(3), [5.0, 0.0, 0.0]))
        ray = np.zeros(2)
        window = [_BundleFrame(pose.copy(), {}, i) for i, pose in enumerate(poses)]
        vo.bundle_window = window
        vo.reference = _Keyframe({}, poses[1].copy())
        vo.landmark_observations = {7: [(poses[0].copy(), ray), (outside, ray)]}
        landmarks = {7: np.array([0.0, 0.0, 5.0])}
        vo._apply_bundle_correction(window, poses, corrected, landmarks)
        np.testing.assert_array_equal(vo.reference.pose, corrected[1])
        history = vo.landmark_observations[7]
        np.testing.assert_array_equal(history[0][0], corrected[0])
        # Poses outside the window are never rewritten.
        np.testing.assert_array_equal(history[1][0], outside)
        np.testing.assert_array_equal(window[2].pose, corrected[2])
        np.testing.assert_array_equal(vo.pose, corrected[2])
        self.assertIs(vo.landmarks, landmarks)
