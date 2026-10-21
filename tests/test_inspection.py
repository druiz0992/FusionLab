import json
import tempfile
import unittest
from pathlib import Path

from sensor_stream.inspection import load_capture, summarize, plot_capture


def sample(timestamp, session="a"):
    return dict(
        type="ImuSample",
        session_id=session,
        stream_id=session + ":imu",
        source_time_ns=timestamp,
        received_monotonic_ns=timestamp,
        acceleration=[1.0, 0.0, 0.0],
        angular_velocity=[0.0, 0.0, 0.0],
        orientation=[0.0, 0.0, 0.0, 1.0],
    )


class InspectionTests(unittest.TestCase):
    def test_summary_separates_sessions_and_checks_counts(self):
        events = [
            sample(0),
            sample(20_000_000),
            sample(100_000_000),
            sample(0, "b"),
            sample(20_000_000, "b"),
        ]
        result = summarize(
            {"samples_written": 5, "final_stats": {"decoded_samples": 6}}, events
        )
        self.assertEqual(result["imu_samples"], 5)
        self.assertEqual(len(result["warnings"]), 1)
        self.assertEqual(len(result["streams"]), 2)
        self.assertEqual(result["streams"][0]["gaps_over_30ms"], 1)
        self.assertEqual(result["streams"][1]["average_rate_hz"], 50)
        self.assertEqual(result["streams"][0]["acceleration"]["rms_magnitude"], 1)

    def test_empty_and_single_sample(self):
        self.assertEqual(summarize({}, [])["imu_samples"], 0)
        self.assertIsNone(summarize({}, [sample(0)])["streams"][0]["average_rate_hz"])
        with self.assertRaises(ValueError):
            plot_capture([], Path("unused.png"))

    def test_invalid_line_is_identified(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "metadata.json").write_text("{}")
            (path / "events.jsonl").write_text(json.dumps(sample(0)) + "\n{broken\n")
            with self.assertRaisesRegex(ValueError, "line 2"):
                load_capture(path)

    def test_plot_png(self):
        try:
            import matplotlib
        except ImportError:
            self.skipTest("Optional matplotlib not installed")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plot.png"
            plot_capture([sample(0), sample(20_000_000), sample(40_000_000, "b")], path)
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
