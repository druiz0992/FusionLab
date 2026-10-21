import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "orbslam3_walking",
    Path(__file__).resolve().parents[1] / "benchmarks/orbslam3/walking.py",
)
walking = importlib.util.module_from_spec(spec)
spec.loader.exec_module(walking)


class OrbTrackingSummaryTests(unittest.TestCase):
    def test_tracking_losses_recovery_and_map_continuity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (root / "states.csv").open("w") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["source_time_ns", "state", "map_id", "processing_ms"]
                )
                writer.writeheader()
                for i, (state, map_id) in enumerate(
                    [
                        (1, 0),
                        (2, 0),
                        (2, 0),
                        (3, 0),
                        (2, 0),
                        (2, 1),
                        (4, 1),
                        (1, 2),
                        (2, 2),
                        (2, 2),
                    ]
                ):
                    writer.writerow(
                        dict(
                            source_time_ns=1000000000 * i,
                            state=state,
                            map_id=map_id,
                            processing_ms=10,
                        )
                    )
            result = walking.summarize(root)
            self.assertEqual(result["tracked_frames"], 6)
            self.assertEqual(result["tracking_to_nontracking_transitions"], 2)
            self.assertEqual(result["continuous_tracked_spans"], 4)
            self.assertEqual(result["longest_tracked_span_s"], 1)
            self.assertEqual(result["map_id_transitions"], 2)
            self.assertEqual(result["state_counts"]["3"], 1)
            (root / "frames.tsv").write_text(
                "".join(f"{i*1000000000}\timage.jpg\n" for i in range(11))
            )
            partial = walking.summarize(root)
            self.assertFalse(partial["complete_input"])
            self.assertEqual(partial["input_frames"], 11)

    def test_input_settings_scale_and_selected_frames(self):
        from hashlib import sha256

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture"
            capture.mkdir()
            data = b"raw fixture bytes"
            (capture / "image.jpg").write_bytes(data)
            events = [
                dict(
                    type="CameraSample",
                    stream_id="cam",
                    source_time_ns=i * 100000000,
                    image_path="image.jpg",
                    image_size_bytes=len(data),
                    image_sha256=sha256(data).hexdigest(),
                )
                for i in range(4)
            ]
            (capture / "events.jsonl").write_text(
                "".join(json.dumps(e) + "\n" for e in events)
            )
            calibration = root / "calibration.json"
            calibration.write_text(
                json.dumps(
                    dict(
                        image_size=[1920, 1080],
                        camera_matrix=[[1200, 0, 960], [0, 1200, 540], [0, 0, 1]],
                        distortion_coefficients=[0, 0, 0, 0, 0],
                        distortion_order=["k1", "k2", "p1", "p2", "k3"],
                    )
                )
            )
            result = walking.prepare(capture, calibration, root / "output", stride=2)
            self.assertEqual(result["frames"], 2)
            self.assertEqual(result["image_size"], [640, 360])
            self.assertIn(
                "Camera1.fx: 400.0", (root / "output/camera.yaml").read_text()
            )
            self.assertAlmostEqual(result["fps"], 5)
            self.assertIn("Camera.fps: 5\n", (root / "output/camera.yaml").read_text())

    def test_replay_paths_split_on_loss_and_map_change(self):
        replay_spec = importlib.util.spec_from_file_location(
            "orb_replay",
            Path(__file__).resolve().parents[1]
            / "benchmarks/orbslam3/export_replay.py",
        )
        replay = importlib.util.module_from_spec(replay_spec)
        replay_spec.loader.exec_module(replay)
        rows = [
            dict(state=state, map_id=map_id, cx="1", cy="2", cz="3")
            for state, map_id in [(2, 0), (2, 0), (3, 0), (2, 0), (2, 1)]
        ]
        updates = list(replay.trajectory_updates(rows))
        self.assertEqual([len(path) for _, _, path in updates], [1, 2, 0, 1, 1])
        self.assertEqual([span for _, span, _ in updates], [0, 0, None, 1, 2])
        rows[0]["cx"] = "nan"
        with self.assertRaises(ValueError):
            list(replay.trajectory_updates(rows))
