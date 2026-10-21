import argparse
import asyncio
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sensor_stream import CameraSample, ReceiverConfig, SourceStats
from sensor_stream.live import LatestFrame
from sensor_stream.cli import _consume
from sensor_stream.recording import SessionRecorder


def frame(timestamp):
    return CameraSample(
        "camera", "session", timestamp, timestamp, timestamp, 16, 12, "jpeg", b"bytes"
    )


class MailboxTests(unittest.TestCase):
    def test_only_latest_display_frame_retained(self):
        mailbox = LatestFrame()
        mailbox.submit(frame(1))
        mailbox.submit(frame(2))
        mailbox.submit(frame(3))
        self.assertEqual(mailbox.skipped, 2)
        self.assertEqual(mailbox.take(), frame(3))
        self.assertIsNone(mailbox.take())


class LiveConsumerTests(unittest.IsolatedAsyncioTestCase):
    async def test_preview_does_not_remove_frames_from_recording_and_closes(self):
        submitted = []
        closed = []

        class Preview:
            def __init__(self, *args):
                pass

            def submit(self, event):
                submitted.append(event)

            async def run(self):
                await asyncio.Future()

            def close(self):
                closed.append(True)

        class Source:
            port = 5678

            def stats(self):
                return SourceStats()

            async def events(self):
                for i in range(3):
                    yield frame(i)

        args = argparse.Namespace(
            live=True, display_rotation=90, host="127.0.0.1", print_samples=False
        )
        with tempfile.TemporaryDirectory() as root:
            with SessionRecorder(Path(root), ReceiverConfig()) as recorder:
                with (
                    patch("sensor_stream.live.LiveCameraPreview", Preview),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    await _consume(args, Source(), recorder)
            events = [
                json.loads(line)
                for line in (recorder.path / "events.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(events), 3)
            self.assertEqual(submitted, [frame(i) for i in range(3)])
            self.assertEqual(closed, [True])
