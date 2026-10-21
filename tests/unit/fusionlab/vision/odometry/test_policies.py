"""VO policy registries and plugging in a new variant."""

import contextlib
import io
import unittest
from unittest.mock import patch

from fusionlab.vision.odometry.policies import (
    KEYFRAME_POLICIES,
    LANDMARK_POLICIES,
    POSE_ACCEPTANCE,
    EveryFrameKeyframes,
    ParallaxKeyframes,
    lookup,
)

from support import require
from support.vision import calibration_json, translating_camera_rows


class PolicyRegistryTests(unittest.TestCase):
    def test_baseline_is_registered_first_for_every_kind(self):
        for registry in (POSE_ACCEPTANCE, KEYFRAME_POLICIES, LANDMARK_POLICIES):
            self.assertEqual(next(iter(registry)), "baseline")
            for name, policy in registry.items():
                self.assertEqual(policy.name, name)

    def test_landmark_policies_combine_two_independent_switches(self):
        flags = {
            name: (policy.validate_third_view, policy.refine_positions)
            for name, policy in LANDMARK_POLICIES.items()
        }
        self.assertEqual(
            flags,
            {
                "baseline": (False, False),
                "validation": (True, False),
                "refinement": (False, True),
                "multiview": (True, True),
            },
        )
        self.assertFalse(LANDMARK_POLICIES["baseline"].keeps_observations)
        self.assertTrue(LANDMARK_POLICIES["refinement"].keeps_observations)

    def test_pose_acceptance_only_refined_refines_before_gates(self):
        self.assertFalse(POSE_ACCEPTANCE["baseline"].refine_before_gates)
        self.assertTrue(POSE_ACCEPTANCE["refined"].refine_before_gates)

    def test_every_frame_keyframes_always_select_without_recording(self):
        policy = EveryFrameKeyframes()
        decision = policy.decide({}, None, {}, None)
        self.assertEqual(
            decision, {"selected": True, "reason": "baseline_try_every_frame"}
        )
        self.assertFalse(policy.explicit)

    def test_parallax_keyframes_pass_their_thresholds(self):
        (np,) = require("numpy")
        pose = np.column_stack((np.eye(3), np.zeros(3)))
        rays = {key: np.array([0.01 * key, 0.0]) for key in range(30)}
        # Identical rays and pose: no parallax, full survival -> keep waiting.
        decision = ParallaxKeyframes(parallax_deg=1.0).decide(rays, pose, rays, pose)
        self.assertFalse(decision["selected"])
        self.assertEqual(decision["parallax_threshold_deg"], 1.0)

    def test_lookup_lists_known_names(self):
        with self.assertRaisesRegex(ValueError, "baseline, parallax"):
            lookup(KEYFRAME_POLICIES, "sometimes", "keyframe policy")

    def test_cli_offers_exactly_the_registered_names(self):
        from fusionlab.cli import main

        for option, registry in (
            ("--pose-acceptance", POSE_ACCEPTANCE),
            ("--keyframe-policy", KEYFRAME_POLICIES),
            ("--landmark-policy", LANDMARK_POLICIES),
        ):
            with self.subTest(option=option):
                argv = ["fusionlab", "visual-odometry", "--help"]
                with patch("sys.argv", argv):
                    with contextlib.redirect_stdout(io.StringIO()) as stdout:
                        with self.assertRaises(SystemExit):
                            main()
                self.assertIn(f"{option} {{{','.join(registry)}}}", stdout.getvalue())


class NeverKeyframes:
    """A test-only rule: never triangulate after bootstrap."""

    name: str = "never"

    explicit: bool = True

    def decide(self, reference_points, reference_pose, points, pose):
        return {"selected": False, "reason": "test_never"}


class PolicyExtensionTests(unittest.TestCase):
    """A new keyframe rule plugs in by registry entry alone."""

    def setUp(self):
        require("cv2", "numpy")

    def test_registered_keyframe_rule_is_used_by_the_estimator(self):
        from fusionlab.vision.odometry.visual_odometry import MonocularOdometry
        from fusionlab.vision.tracks import TrackFrame

        with patch.dict(KEYFRAME_POLICIES, {"never": NeverKeyframes()}):
            vo = MonocularOdometry(calibration_json(), keyframe_policy="never")
            results = [
                vo.process(TrackFrame.from_json(row))
                for row in translating_camera_rows(5)
            ]
        tracked = [r for r in results if r.is_tracking]
        self.assertGreater(len(tracked), 1)
        # After the bootstrap frame, the rule blocks every triangulation.
        for result in tracked[1:]:
            self.assertEqual(
                result.diagnostics["keyframe_decision"]["reason"], "test_never"
            )
            self.assertEqual(result.diagnostics["landmarks_added"], 0)
        self.assertNotIn("never", KEYFRAME_POLICIES)
