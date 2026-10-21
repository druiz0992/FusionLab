import argparse
import asyncio
import contextlib
import io
import json
import tempfile
import unittest
import threading
from pathlib import Path
from unittest.mock import patch

from sensor_stream.cli import _listen, _positive_seconds, _write_recording
from sensor_stream import SourceDiscontinuity
from test_sensor_stream import frame, imu_payload


class DurationTests(unittest.TestCase):
    def test_duration_validation(self):
        self.assertEqual(_positive_seconds("0.25"), 0.25)
        for value in ("0", "-1", "nan", "inf", "-inf", "hello"):
            with (
                self.subTest(value=value),
                self.assertRaises(argparse.ArgumentTypeError),
            ):
                _positive_seconds(value)


class TimedCaptureTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_waits_for_inflight_recording_write(self):
        started = threading.Event()
        release = threading.Event()
        written = []

        class SlowRecorder:
            def write(self, event):
                started.set()
                if not release.wait(2):
                    raise RuntimeError("Test writer timed out")
                written.append(event)

        event = SourceDiscontinuity("session", "test")
        task = asyncio.create_task(_write_recording(SlowRecorder(), event))
        try:
            async with asyncio.timeout(2):
                while not started.is_set():
                    await asyncio.sleep(0.001)
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(written, [event])
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)

    async def check_pending_capture(self, timed: bool) -> None:
        ready = asyncio.Event()

        async def stalled_consumer(args, source, recorder):
            _, writer = await asyncio.open_connection("127.0.0.1", source.port)
            try:
                writer.write(b"".join(frame(imu_payload(i), i) for i in range(5)))
                await writer.drain()
                async with asyncio.timeout(2):
                    while source.stats().decoded_samples != 5:
                        await asyncio.sleep(0.001)
                ready.set()
                await asyncio.Future()  # Deliberately leave all events queued.
            finally:
                writer.close()
                await writer.wait_closed()

        with tempfile.TemporaryDirectory() as root:
            args = argparse.Namespace(
                host="127.0.0.1",
                port=0,
                queue_capacity=4096,
                overflow_policy="raise",
                record=True,
                output_dir=Path(root),
                duration=0.15 if timed else None,
                print_samples=False,
            )
            with (
                patch("sensor_stream.cli._consume", stalled_consumer),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                task = asyncio.create_task(_listen(args))
                try:
                    await asyncio.wait_for(ready.wait(), 2)
                    if not timed:
                        task.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await task
                    else:
                        await asyncio.wait_for(task, 2)
                finally:
                    if not task.done():
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
            folder = next(Path(root).iterdir())
            metadata = json.loads((folder / "metadata.json").read_text())
            events = [
                json.loads(line)
                for line in (folder / "events.jsonl").read_text().splitlines()
            ]
            self.assertEqual(events[0]["type"], "StreamStarted")
            self.assertEqual(
                [e["source_time_ns"] for e in events if e["type"] == "ImuSample"],
                list(range(5)),
            )
            self.assertEqual(metadata["samples_written"], 5)
            self.assertEqual(metadata["final_stats"]["decoded_samples"], 5)
            self.assertEqual(metadata["final_stats"]["queue_depth"], 0)
            self.assertEqual(
                metadata["status"], "completed" if timed else "interrupted"
            )

    async def test_timed_stop_drains_pending_samples(self):
        await self.check_pending_capture(timed=True)

    async def test_cancel_drains_pending_samples(self):
        await self.check_pending_capture(timed=False)

    async def test_duration_stops_idle_receiver_and_finalizes_capture(self):
        with tempfile.TemporaryDirectory() as root:
            args = argparse.Namespace(
                host="127.0.0.1",
                port=0,
                queue_capacity=4096,
                overflow_policy="raise",
                record=True,
                output_dir=Path(root),
                duration=0.05,
                print_samples=False,
            )
            with contextlib.redirect_stdout(io.StringIO()) as output:
                await _listen(args)
            folders = list(Path(root).iterdir())
            self.assertEqual(len(folders), 1)
            metadata = json.loads((folders[0] / "metadata.json").read_text())
            self.assertEqual(metadata["status"], "completed")
            self.assertIn("Duration reached", output.getvalue())
            self.assertFalse(metadata["final_stats"]["connected"])
