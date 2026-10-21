import unittest

from fusionlab.vision.odometry.landmark_refinement import (
    refine_landmark,
    reprojection_errors,
)


class LandmarkRefinementTests(unittest.TestCase):
    def setUp(self):
        try:
            import numpy as np
            import cv2
        except ImportError:
            self.skipTest("Optional tracking dependencies unavailable")
        self.np = np

    def test_multiview_refinement_improves_position_and_cost(self):
        np = self.np
        truth = np.array([0.3, -0.1, 5.0])
        rng = np.random.default_rng(9)
        observations = []
        for x in (0.0, 0.3, 0.6, 0.9):
            pose = np.column_stack((np.eye(3), [-x, 0.0, 0.0]))
            camera = pose[:, :3] @ truth + pose[:, 3]
            observations.append(
                (pose, camera[:2] / camera[2] + rng.normal(0, 0.1 / 500, 2))
            )
        initial = truth + [0.05, 0.02, 0.4]
        original = initial.copy()
        refined = refine_landmark(initial, observations)
        self.assertLess(
            np.linalg.norm(refined - truth), np.linalg.norm(initial - truth)
        )
        before = reprojection_errors(initial, observations, 500.0)
        after = reprojection_errors(refined, observations, 500.0)
        self.assertLess(after @ after, before @ before)
        np.testing.assert_array_equal(initial, original)
        with self.assertRaises(ValueError):
            refine_landmark(initial, observations[:2])

    def test_pending_landmark_requires_a_consistent_third_view(self):
        from fusionlab.vision.odometry.visual_odometry import MonocularOdometry

        np = self.np
        vo = MonocularOdometry({}, landmark_policy="multiview")
        truth = np.array([0.2, 0.0, 5.0])
        poses = [np.column_stack((np.eye(3), [-x, 0.0, 0.0])) for x in (0.0, 0.3, 0.6)]

        def project(pose):
            camera = pose[:, :3] @ truth + pose[:, 3]
            return camera[:2] / camera[2]

        histories = [(pose.copy(), project(pose)) for pose in poses[:2]]
        vo.pending_landmarks = {1: truth.copy(), 2: truth.copy()}
        vo.landmark_observations = {1: list(histories), 2: list(histories)}
        vo.pose = poses[2]
        self.assertEqual(vo.landmarks, {})
        report = vo._update_landmarks(
            {1: project(poses[2]), 2: project(poses[2]) + [0.1, 0.1]}, 500.0
        )
        self.assertIn(1, vo.landmarks)
        self.assertNotIn(2, vo.landmarks)
        self.assertEqual(report["confirmed"], 1)
        self.assertEqual(report["rejected_third_view"], 1)
        self.assertFalse(vo.pending_landmarks)

    def test_ablation_modes_separate_admission_from_coordinate_updates(self):
        from fusionlab.vision.odometry.visual_odometry import MonocularOdometry

        np = self.np
        truth = np.array([0.2, 0.0, 5.0])
        point = truth + np.array([0.005, 0.0, 0.02])
        poses = [np.column_stack((np.eye(3), [-x, 0.0, 0.0])) for x in (0.0, 0.3, 0.6)]

        def observe(pose):
            camera = pose[:, :3] @ truth + pose[:, 3]
            return camera[:2] / camera[2]

        history = [(pose.copy(), observe(pose)) for pose in poses[:2]]
        validation = MonocularOdometry({}, landmark_policy="validation")
        validation.pending_landmarks = {1: point.copy()}
        validation.landmark_observations = {1: list(history)}
        validation.pose = poses[2]
        report = validation._update_landmarks({1: observe(poses[2])}, 500.0)
        self.assertIn(1, validation.landmarks)
        np.testing.assert_array_equal(validation.landmarks[1], point)
        self.assertEqual(report["refined"], 0)
        refinement = MonocularOdometry({}, landmark_policy="refinement")
        refinement.landmarks = {1: point.copy(), 2: point.copy()}
        refinement.landmark_observations = {1: list(history), 2: list(history)}
        refinement.pose = poses[2]
        report = refinement._update_landmarks(
            {1: observe(poses[2]), 2: observe(poses[2]) + [0.2, 0.0]}, 500.0
        )
        self.assertLess(
            np.linalg.norm(refinement.landmarks[1] - truth),
            np.linalg.norm(point - truth),
        )
        self.assertIn(2, refinement.landmarks)
        self.assertEqual(report["rejected_third_view"], 0)
        self.assertEqual(report["refined"], 1)
