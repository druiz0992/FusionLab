"""Ground-truth association and per-segment Sim(3) scoring."""

import json
import tempfile
import unittest
from pathlib import Path

from support import require


class GroundTruthTests(unittest.TestCase):
    def setUp(self):
        self.np, _ = require("numpy", "scipy")
        require("matplotlib")
        from scipy.spatial.transform import Rotation

        self.Rotation = Rotation

    def reference(self):
        """Body at x = 0, 1, 2, 4 m (t = 0, 10, 20, 100 ms), yawing 10°/sample."""
        from fusionlab.evaluation.ground_truth import ReferenceTrajectory

        np = self.np
        states = []
        for i, (stamp, x) in enumerate([(0, 0.0), (10, 1.0), (20, 2.0), (100, 4.0)]):
            body = np.eye(4)
            body[:3, :3] = self.Rotation.from_euler(
                "z", 10 * i, degrees=True
            ).as_matrix()
            body[0, 3] = x
            states.append(
                dict(source_time_ns=stamp * 1_000_000, world_from_body=body.tolist())
            )
        mounting = np.eye(4)
        mounting[1, 3] = 0.5
        return ReferenceTrajectory.from_json(
            dict(states=states, body_from_camera=mounting.tolist())
        )

    def test_degenerate_alignment_rejected(self):
        from fusionlab.evaluation.ground_truth import similarity_alignment

        with self.assertRaises(ValueError):
            similarity_alignment(
                [[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[0, 0, 0], [2, 0, 0], [4, 0, 0]]
            )

    def test_alignment_recovers_known_similarity(self):
        from fusionlab.evaluation.ground_truth import similarity_alignment

        np = self.np
        estimated = np.random.default_rng(2).normal(size=(10, 3))
        rotation = self.Rotation.from_euler("xyz", [10, -20, 30], degrees=True)
        reference = 3.0 * rotation.apply(estimated) + [1.0, -2.0, 0.5]
        fit = similarity_alignment(estimated, reference)
        self.assertAlmostEqual(fit.scale, 3.0)
        np.testing.assert_allclose(fit.rotation, rotation.as_matrix(), atol=1e-12)
        np.testing.assert_allclose(fit.apply(estimated), reference, atol=1e-12)

    def test_reference_matches_interpolates_and_refuses_gaps(self):
        np = self.np
        reference = self.reference()
        gap_ns = 50e6
        exact = reference.camera_pose_at(10_000_000, gap_ns)
        # Camera = body pose ∘ mounting: the 0.5 m lever arm turns with the body.
        lever = self.Rotation.from_euler("z", 10, degrees=True).apply([0, 0.5, 0])
        np.testing.assert_allclose(exact[:3, 3], [1.0 + lever[0], lever[1], 0])
        middle = reference.camera_pose_at(15_000_000, gap_ns)
        angle = self.Rotation.from_matrix(middle[:3, :3]).as_euler("xyz", degrees=True)
        self.assertAlmostEqual(angle[2], 15.0)
        self.assertIsNone(reference.camera_pose_at(60_000_000, gap_ns))  # 80 ms gap
        self.assertIsNotNone(reference.camera_pose_at(60_000_000, 100e6))
        self.assertIsNone(reference.camera_pose_at(-1, gap_ns))
        self.assertIsNone(reference.camera_pose_at(100_000_001, gap_ns))
        self.assertEqual(reference.camera_positions().shape, (4, 3))

    def test_reference_requires_increasing_timestamps(self):
        from fusionlab.evaluation.ground_truth import ReferenceTrajectory

        state = dict(source_time_ns=5, world_from_body=self.np.eye(4).tolist())
        with self.assertRaisesRegex(ValueError, "increase"):
            ReferenceTrajectory.from_json(
                dict(states=[state, state], body_from_camera=self.np.eye(4).tolist())
            )

    def test_rotating_camera_lever_arm_and_independent_segments(self):
        from fusionlab.evaluation.ground_truth import evaluate_run

        np = self.np
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            run.mkdir()
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = self.Rotation.from_euler(
                "x", 20, degrees=True
            ).as_matrix()
            extrinsic[0, 3] = 0.3
            epoch = 1400000000000000000
            states = []
            for i in range(7):
                body = np.eye(4)
                body[:3, :3] = self.Rotation.from_euler(
                    "z", i * 15, degrees=True
                ).as_matrix()
                states.append(
                    dict(
                        source_time_ns=epoch + i * 10000000,
                        world_from_body=body.tolist(),
                    )
                )
            truth = root / "truth.json"
            truth.write_text(
                json.dumps(dict(states=states, body_from_camera=extrinsic.tolist()))
            )
            records = []
            for i in range(6):
                body = np.eye(4)
                body[:3, :3] = self.Rotation.from_euler(
                    "z", (i + 0.5) * 15, degrees=True
                ).as_matrix()
                camera = body @ extrinsic
                scale = 2 if i < 3 else 5
                # Save poses exactly as VO does: world-to-camera [R | t], 3x4,
                # with t = -R c for the arbitrary-scale camera center c.
                center = camera[:3, 3] / scale
                rotation = camera[:3, :3].T
                world_to_camera = np.column_stack((rotation, -rotation @ center))
                records.append(
                    dict(
                        source_time_ns=epoch + i * 10000000 + 5000000,
                        status="tracking",
                        segment=1 if i < 3 else 2,
                        camera_center_arbitrary=center.tolist(),
                        world_to_camera=world_to_camera.tolist(),
                    )
                )
            (run / "poses.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in records)
            )
            result = evaluate_run(run, truth, root / "score")
            self.assertEqual(len(result["segments"]), 2)
            for segment, scale in zip(result["segments"], [2, 5]):
                self.assertAlmostEqual(segment["scale"], scale)
                self.assertLess(segment["position_rmse_m"], 1e-10)
                self.assertLess(segment["rotation_rmse_deg"], 1e-8)
            self.assertTrue((root / "score/trajectory.svg").exists())

    def test_unalignable_segment_is_reported_with_reason(self):
        from fusionlab.evaluation.ground_truth import evaluate_run

        np = self.np
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "run").mkdir()
            states = [
                dict(source_time_ns=i, world_from_body=np.eye(4).tolist())
                for i in (0, 10)
            ]
            truth = root / "truth.json"
            truth.write_text(
                json.dumps(dict(states=states, body_from_camera=np.eye(4).tolist()))
            )
            row = dict(
                source_time_ns=5,
                status="tracking",
                segment=1,
                camera_center_arbitrary=[0, 0, 0],
                world_to_camera=np.eye(4)[:3].tolist(),
            )
            (root / "run/poses.jsonl").write_text(json.dumps(row) + "\n")
            result = evaluate_run(root / "run", truth, root / "score")
            report = (root / "score/report.md").read_text()
            self.assertFalse((root / "score/trajectory.svg").exists())
        segment = result["segments"][0]
        self.assertEqual(segment["alignment_status"], "unavailable")
        self.assertIn("three", segment["reason"])
        self.assertEqual(result["matched_tracking_fraction_of_reference_frames"], 1.0)
        self.assertIn("| 1 | 1 | 0.000 |", report)
