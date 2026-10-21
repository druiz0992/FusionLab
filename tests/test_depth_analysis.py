import tempfile
import struct
import unittest
from pathlib import Path

from sensor_stream.inspection.depth_analysis import pair_frames, analyze
from sensor_stream import DepthSample, ReceiverConfig
from sensor_stream.recording import SessionRecorder


def event(time, session="s", stream="c"):
    return {
        "source_time_ns": time,
        "session_id": session,
        "stream_id": stream,
        "image_path": str(time),
    }


class PairingTests(unittest.TestCase):
    def test_tolerance_reuse_and_session_boundaries(self):
        depth = [event(0), event(5_000_000), event(100_000_000), event(0, "other")]
        cameras = [event(3_000_000)]
        pairs, unmatched = pair_frames(depth, cameras, 5)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(unmatched, 2)
        self.assertEqual(pairs[0]["camera_minus_depth_ms"], 3)
        self.assertEqual(pairs[1]["camera_minus_depth_ms"], -2)

    def test_ambiguous_stream_and_bad_tolerance(self):
        with self.assertRaises(ValueError):
            pair_frames([], [], float("nan"))
        with self.assertRaises(ValueError):
            pair_frames([], [event(0, stream="a"), event(1, stream="b")], 20)

    def test_counts_zero_and_uint16_max_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            with SessionRecorder(Path(directory), ReceiverConfig()) as recorder:
                recorder.write(
                    DepthSample(
                        "d",
                        "s",
                        0,
                        0,
                        0,
                        2,
                        2,
                        "16UC1",
                        struct.pack("<4H", 0, 100, 65535, 200),
                    )
                )
            report = analyze(recorder.path, recorder.path / "report")
            self.assertEqual(report["analyzed_counts"]["zero"], 1)
            self.assertEqual(report["analyzed_counts"]["uint16_max"], 1)
            self.assertEqual(report["zero_fraction"], 0.25)
            self.assertEqual(report["unmatched_depth_frames"], 1)
            self.assertTrue((recorder.path / "report" / "summary.json").is_file())
