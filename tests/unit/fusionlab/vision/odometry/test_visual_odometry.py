"""Monocular VO estimator on synthetic scenes."""

import unittest

from fusionlab.vision.tracks import TrackFrame
from fusionlab.vision.odometry.visual_odometry import MonocularOdometry

from support.vision import calibration_json, track_row, translating_camera_rows
from support import require


class VisualOdometryTests(unittest.TestCase):
    def setUp(self):
        self.np, self.cv = require("numpy", "cv2")

    def test_synthetic_trajectory_preserves_scale_and_resets(self):
        np = self.np
        calibration = calibration_json()
        vo = MonocularOdometry(calibration)
        refined_vo = MonocularOdometry(calibration, pose_acceptance="refined")
        keyframe_vo = MonocularOdometry(calibration, keyframe_policy="parallax")
        multiview_vo = MonocularOdometry(calibration, landmark_policy="multiview")
        multiview_results = []
        keyframe_results = []
        refined_results = []
        results = []
        for row in translating_camera_rows(6):
            results.append(vo.process(TrackFrame.from_json(row)))
            refined_results.append(refined_vo.process(TrackFrame.from_json(row)))
            keyframe_results.append(keyframe_vo.process(TrackFrame.from_json(row)))
            multiview_results.append(multiview_vo.process(TrackFrame.from_json(row)))
        self.assertTrue(all(r.status == "tracking" for r in multiview_results[1:]))
        np.testing.assert_allclose(
            multiview_results[-1].camera_center_arbitrary, [5.0, 0.0, 0.0], atol=0.05
        )
        self.assertTrue(all(r.status == "tracking" for r in keyframe_results[1:]))
        np.testing.assert_allclose(
            keyframe_results[-1].camera_center_arbitrary, [5.0, 0.0, 0.0], atol=0.05
        )
        self.assertTrue(all(r.status == "tracking" for r in refined_results[1:]))
        np.testing.assert_allclose(
            refined_results[-1].camera_center_arbitrary, [5.0, 0.0, 0.0], atol=0.05
        )
        self.assertEqual(results[0].status, "initializing")
        self.assertTrue(all(r.status == "tracking" for r in results[1:]))
        np.testing.assert_allclose(
            results[-1].camera_center_arbitrary, [5.0, 0.0, 0.0], atol=0.05
        )
        row["tracks"] = []
        row["source_time_ns"] += 100_000_000
        failure = vo.process(TrackFrame.from_json(row))
        self.assertEqual(failure.status, "tracking_lost")
        self.assertEqual(failure.failure_category, "too_few_visible_landmarks")
        self.assertFalse(failure.diagnostics["pnp_attempted"])
        self.assertEqual(failure.diagnostics["visible_landmarks"], 0)
        self.assertGreater(failure.diagnostics["landmarks_before"], 0)
        import json
        import tempfile
        from pathlib import Path
        from fusionlab.evaluation.vo_reports import write_failure_report

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            from dataclasses import replace

            all_rows = [
                replace(result, frame_index=index)
                for index, result in enumerate(results + [failure])
            ]
            (output / "poses.jsonl").write_text(
                "".join(json.dumps(r.to_json()) + "\n" for r in all_rows)
            )
            report = write_failure_report(output)
            self.assertEqual(report["failure_count"], 1)
            self.assertEqual(
                report["failure_categories"], {"too_few_visible_landmarks": 1}
            )
            self.assertTrue(report["failures"][0]["preceding_frames"])
            from fusionlab.evaluation.vo_reports import write_geometry_report

            audit = write_geometry_report(output)
            self.assertEqual(audit["failure_count"], 1)
            self.assertEqual(audit["counterfactual_same_gate_recoveries"], 0)
            self.assertGreater(audit["triangulated_landmark_observations"], 0)
        self.assertEqual(len(vo.landmarks), 0)

    def test_triangulation_rejects_negative_depth_and_weak_parallax(self):
        np = self.np
        vo = MonocularOdometry({})
        pose_a = np.column_stack((np.eye(3), np.zeros(3)))
        pose_b = np.column_stack((np.eye(3), [-1.0, 0.0, 0.0]))
        points = np.array([[0.0, 0.0, 5.0], [0.0, 0.0, 1000.0], [0.0, 0.0, -5.0]])
        a = {i: p[:2] / p[2] for i, p in enumerate(points)}
        b = {i: (p + [-1, 0, 0])[:2] / p[2] for i, p in enumerate(points)}
        vo._triangulate(a, pose_a, b, pose_b, 500.0)
        self.assertEqual(set(vo.landmarks), {0})
        self.assertEqual(vo.triangulation_diagnostics.accepted, 1)
        self.assertEqual(vo.triangulation_diagnostics.low_parallax, 1)
        self.assertEqual(vo.triangulation_diagnostics.invalid_depth_or_homogeneous, 1)

    def test_keyframe_policy_waits_for_translation_and_preserves_tracks(self):
        from fusionlab.vision.odometry.policies import choose_keyframe

        np, cv = self.np, self.cv
        reference = {i: np.array([-0.3 + i * 0.02, 0.1 * (i % 2)]) for i in range(30)}
        pose = np.column_stack((np.eye(3), np.zeros(3)))
        self.assertFalse(choose_keyframe(reference, pose, reference, pose)["selected"])
        rotation = cv.Rodrigues(np.array([0.0, 0.2, 0.0]))[0]
        current_pose = np.column_stack((rotation, np.zeros(3)))
        rotated = {}
        for key, point in reference.items():
            ray = rotation @ np.array([*point, 1.0])
            rotated[key] = ray[:2] / ray[2]
        decision = choose_keyframe(reference, pose, rotated, current_pose)
        self.assertFalse(decision["selected"])
        shifted = {key: point + [0.08, 0.0] for key, point in reference.items()}
        self.assertEqual(
            choose_keyframe(reference, pose, shifted, pose)["reason"], "parallax"
        )
        reduced = dict(list(reference.items())[:10])
        self.assertEqual(
            choose_keyframe(reference, pose, reduced, pose)["reason"], "track_survival"
        )

    def test_blank_frames_do_not_invent_motion(self):
        vo = MonocularOdometry(calibration_json())
        row = track_row(0)
        self.assertEqual(vo.process(TrackFrame.from_json(row)).status, "initializing")
        row["source_time_ns"] = 100_000_000
        self.assertEqual(vo.process(TrackFrame.from_json(row)).status, "initializing")
        self.assertIsNone(vo.pose)
