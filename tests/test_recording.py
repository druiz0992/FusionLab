import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from sensor_stream import (
    ImuSample,
    ReceiverConfig,
    SourceStats,
    StreamDescriptor,
    StreamStarted,
)
from sensor_stream.recording import SessionRecorder


class RecordingTests(unittest.TestCase):
    def test_records_events_and_final_metadata(self):
        with tempfile.TemporaryDirectory() as root:
            with SessionRecorder(Path(root), ReceiverConfig()) as recorder:
                recorder.write(
                    StreamStarted(StreamDescriptor("imu", "session", "phone"))
                )
                recorder.write(
                    ImuSample(
                        "imu",
                        "session",
                        123,
                        456,
                        789,
                        (1.0, 2.0, 3.0),
                        (0.0, 0.0, 0.0),
                        None,
                    )
                )
                # Lines must be readable before closing the recording.
                lines = (recorder.path / "events.jsonl").read_text().splitlines()
                self.assertEqual(len(lines), 2)
                self.assertEqual(json.loads(lines[1])["source_time_ns"], 123)
                recorder.set_stats(SourceStats(decoded_samples=1))
            metadata = json.loads((recorder.path / "metadata.json").read_text())
            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(metadata["samples_written"], 1)
            self.assertEqual(metadata["events_written"], 2)
            self.assertEqual(metadata["final_stats"]["decoded_samples"], 1)
            with SessionRecorder(Path(root), ReceiverConfig()) as second:
                self.assertNotEqual(second.path, recorder.path)

    def test_interruption_and_failure_preserve_metadata(self):
        with tempfile.TemporaryDirectory() as root:
            for error, status in (
                (asyncio.CancelledError(), "interrupted"),
                (OSError("disk error"), "failed"),
            ):
                recorder = SessionRecorder(Path(root), ReceiverConfig())
                with self.assertRaises(type(error)):
                    with recorder:
                        raise error
                metadata = json.loads((recorder.path / "metadata.json").read_text())
                self.assertEqual(metadata["status"], status)
                self.assertIn("ended_at", metadata)
