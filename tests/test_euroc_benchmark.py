import json
from pathlib import Path
import tempfile
import unittest


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

    def test_import_tracks_and_odometry_keep_source_and_timestamps(self):
        from fusionlab.recordings.euroc import import_sequence
        from fusionlab.vision.feature_tracking import track_capture
        from fusionlab.vision.odometry.runner import run_odometry

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, epoch = self.fixture(root)
            output = root / "imported"
            meta = import_sequence(source, output)
            self.assertEqual(meta["frames"], 8)
            events = [
                json.loads(line)
                for line in (output / "events.jsonl").read_text().splitlines()
            ]
            self.assertEqual(events[0]["source_time_ns"], epoch)
            self.assertEqual(events[1]["received_monotonic_ns"], 10000000)
            filename = f"{epoch}.png"
            self.assertEqual(
                (source / "cam0/data" / filename).read_bytes(),
                (output / "camera" / filename).read_bytes(),
            )
            track_capture(output, root / "tracks", display_rotation=0)
            run_odometry(
                root / "tracks/tracks.jsonl",
                output / "calibration.json",
                root / "run",
                stride=1,
            )
            self.assertEqual(
                len((root / "run/poses.jsonl").read_text().splitlines()), 8
            )
            with self.assertRaises(FileExistsError):
                import_sequence(source, output)

    def test_import_rejects_traversal_before_creating_output(self):
        from fusionlab.recordings.euroc import import_sequence

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, epoch = self.fixture(root)
            (source / "cam0/data.csv").write_text(f"{epoch},../../sensor.yaml\n")
            with self.assertRaises(ValueError):
                import_sequence(source, root / "output")
            self.assertFalse((root / "output").exists())

    def test_interpolated_camera_reference_sim3_and_coverage(self):
        from fusionlab.recordings.euroc import import_sequence
        from fusionlab.evaluation.ground_truth import evaluate_run

        np = self.np
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, epoch = self.fixture(root)
            output = root / "imported"
            import_sequence(source, output)
            run = root / "run"
            run.mkdir()
            rotation = self.Rotation.from_euler("z", 35, degrees=True).as_matrix()
            records = []
            for i in range(6):
                t = (i + 0.5) * 0.01
                # Linear interpolation of body position, then apply camera lever arm.
                a = i * 0.1
                b = (i + 1) * 0.1
                camera = np.array([(a + b) / 2 + 0.2, (a * a + b * b) / 2, 0])
                estimate = rotation.T @ (camera - np.array([1, 2, 3])) / 2.5
                records.append(
                    dict(
                        source_time_ns=epoch + int(round(t * 1e9)),
                        status="tracking",
                        segment=1,
                        camera_center_arbitrary=estimate.tolist(),
                        world_to_camera=rotation.tolist(),
                        refined_median_inlier_error_px=None,
                    )
                )
            records.append(
                dict(source_time_ns=epoch + 65000000, status="tracking_lost", segment=1)
            )
            records.append(
                dict(source_time_ns=epoch + 90000000, status="tracking_lost", segment=1)
            )
            (run / "poses.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in records)
            )
            result = evaluate_run(run, output / "groundtruth.json", root / "score")
            self.assertEqual(result["matched_tracking_frames"], 6)
            self.assertEqual(result["reference_available_frames"], 7)
            self.assertEqual(result["continuity"]["tracking_losses"], 2)
            self.assertAlmostEqual(result["segments"][0]["scale"], 2.5)
            self.assertLess(result["segments"][0]["position_rmse_m"], 1e-10)
            self.assertLess(result["segments"][0]["rotation_rmse_deg"], 1e-8)
            rejected = evaluate_run(
                run, output / "groundtruth.json", root / "gap", max_gap_ms=1
            )
            self.assertEqual(rejected["matched_tracking_frames"], 0)

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

    def test_import_cli_dispatch(self):
        import contextlib
        import io
        from unittest.mock import patch
        from fusionlab.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, _ = self.fixture(root)
            with patch(
                "sys.argv",
                [
                    "fusionlab",
                    "import-euroc",
                    str(source),
                    "--output-dir",
                    str(root / "capture"),
                ],
            ):
                with contextlib.redirect_stdout(io.StringIO()) as stdout:
                    main()
            self.assertEqual(json.loads(stdout.getvalue())["frames"], 8)
