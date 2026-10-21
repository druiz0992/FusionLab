import json
import tempfile
import unittest
from pathlib import Path
from hashlib import sha256

from sensor_stream.inspection.camera_replay import write_replay


class CameraReplayTests(unittest.TestCase):
    def test_replay_uses_relative_paths_and_both_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = root / "capture"
            capture.mkdir()
            (capture / "metadata.json").write_text("{}")
            image = capture / "frame one.jpg"
            image.write_bytes(b"frame")
            event = dict(
                type="CameraSample",
                session_id="s",
                stream_id="c",
                width=1,
                height=1,
                encoding="jpeg",
                image_path=image.name,
                image_size_bytes=5,
                image_sha256=sha256(b"frame").hexdigest(),
                source_time_ns=0,
                received_monotonic_ns=100,
            )
            second = {
                **event,
                "source_time_ns": 1_000_000_000,
                "received_monotonic_ns": 3_000_000_100,
            }
            (capture / "events.jsonl").write_text(
                json.dumps(event) + "\n" + json.dumps(second) + "\n"
            )
            output = root / "replay.html"
            self.assertEqual(write_replay(capture, output), 2)
            html = output.read_text()
            self.assertIn("capture/frame%20one.jpg", html)
            self.assertIn('"source": 1.0, "receive": 3.0', html)
            self.assertNotIn("const frames=FRAMES", html)

    def test_empty_capture_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metadata.json").write_text("{}")
            (root / "events.jsonl").write_text("")
            with self.assertRaises(ValueError):
                write_replay(root, root / "replay.html")
