"""Ground-truth association and per-segment Sim(3) scoring."""

import json
import tempfile
import unittest
from pathlib import Path


class EurocBenchmarkTests(unittest.TestCase):
    def setUp(self):
        try:
            import cv2
            import numpy as np
            import yaml
            from scipy.spatial.transform import Rotation
        except ImportError:
            self.skipTest("benchmark extras unavailable")
        self.np, self.cv, self.yaml, self.Rotation = np, cv2, yaml, Rotation

    def fixture(self, root):
        np = self.np
        mav = root / "source/mav0"
        camera = mav / "cam0"
        (camera / "data").mkdir(parents=True)
        gt = mav / "state_groundtruth_estimate0"
        gt.mkdir()
        transform = np.eye(4)
        transform[0, 3] = 0.2
        sensor = dict(
            camera_model="pinhole",
            distortion_model="radial-tangential",
            resolution=[96, 64],
            intrinsics=[80, 80, 48, 32],
            distortion_coefficients=[0, 0, 0, 0],
            T_BS=dict(data=transform.ravel().tolist()),
        )
        (camera / "sensor.yaml").write_text(self.yaml.safe_dump(sensor))
        epoch = 1400000000000000000
        frames = []
        states = []
        for i in range(8):
            stamp = epoch + i * 10000000
            filename = f"{stamp}.png"
            image = np.random.default_rng(4).integers(0, 255, (64, 96), dtype=np.uint8)
            self.cv.imwrite(str(camera / "data" / filename), image)
            frames.append(f"{stamp},{filename}\n")
            states.append(f"{stamp},{i*0.1},{(i*0.1)**2},0,1,0,0,0,0,0,0\n")
        (camera / "data.csv").write_text("#timestamp,filename\n" + "".join(frames))
        (gt / "data.csv").write_text("#timestamp,p,q\n" + "".join(states))
        return mav, epoch

    def test_degenerate_alignment_rejected(self):
        from fusionlab.evaluation.ground_truth import similarity_alignment

        with self.assertRaises(ValueError):
            similarity_alignment(
                [[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[0, 0, 0], [2, 0, 0], [4, 0, 0]]
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
