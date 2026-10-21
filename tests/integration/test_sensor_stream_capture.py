"""End-to-end acquisition: TCP/USB sources, recording, inspection and CLI capture."""

import argparse
import asyncio
import contextlib
import io
import json
import plistlib
import socket
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from sensor_stream import (
    BufferOverflowError,
    ImuSample,
    ReceiverConfig,
    SensorStreamSource,
    SourceDiscontinuity,
    StreamStarted,
    CameraSample,
    GpsSample,
    DepthSample,
    SourceStats,
)
from sensor_stream.protocol.decoder import decode_message
from sensor_stream.protocol.framing import HEADER
from sensor_stream.protocol.usb import MUX_HEADER, connect_phone
from sensor_stream.recording import SessionRecorder
from sensor_stream.inspection import load_capture, summarize, plot_capture
from sensor_stream.inspection.gps import covariance_sigma
from sensor_stream.cli import _listen, _write_recording, _consume

from support.wire import (
    imu_payload,
    frame,
    usb_frame,
    camera_payload,
    gps_payload,
    depth_payload,
)


def camera_frame(timestamp):
    return CameraSample(
        "camera", "session", timestamp, timestamp, timestamp, 16, 12, "jpeg", b"bytes"
    )


class SourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_read_completing_during_close_does_not_count_unsaved_sample(self):
        source = SensorStreamSource()
        reader = Mock()
        reader.read.return_value = object()
        writer = Mock()
        writer.wait_closed = AsyncMock()

        async def closing_read(awaitable, timeout):
            source._closed = True
            return frame(imu_payload())

        with patch("sensor_stream.source.asyncio.wait_for", closing_read):
            await source._receive(reader, writer, False)
        self.assertEqual(source.stats().decoded_samples, 0)
        self.assertEqual(source.stats().queue_depth, 0)
        writer.close.assert_called_once()

    async def wait_until(self, predicate):
        async with asyncio.timeout(2):
            while not predicate():
                await asyncio.sleep(0.005)

    async def connect(self, source):
        _, writer = await asyncio.open_connection("127.0.0.1", source.port)
        self.addAsyncCleanup(self.close_writer, writer)
        return writer

    async def close_writer(self, writer):
        writer.close()
        await writer.wait_closed()

    async def test_receive_reconnect_and_stats(self):
        async with SensorStreamSource(
            ReceiverConfig(bind_host="127.0.0.1", bind_port=0)
        ) as source:
            events = source.events()
            writer = await self.connect(source)
            writer.write(frame(imu_payload()))
            await writer.drain()
            descriptor = await asyncio.wait_for(anext(events), 2)
            sample = await asyncio.wait_for(anext(events), 2)
            self.assertIsInstance(descriptor, StreamStarted)
            self.assertIsInstance(sample, ImuSample)
            self.assertEqual(sample.acceleration, (1.5, 0.0, 0.0))
            self.assertEqual(sample.stream_id, descriptor.descriptor.stream_id)
            self.assertEqual(descriptor.descriptor.acceleration_units, "unverified")
            await self.close_writer(writer)
            self.assertIsInstance(
                await asyncio.wait_for(anext(events), 2), SourceDiscontinuity
            )
            await self.wait_until(lambda: not source._clients)
            second = await self.connect(source)
            second.write(frame(imu_payload(124)))
            await second.drain()
            started = await asyncio.wait_for(anext(events), 2)
            self.assertNotEqual(started.descriptor.session_id, sample.session_id)
            await asyncio.wait_for(anext(events), 2)
            self.assertEqual(source.stats().connections, 2)
            self.assertEqual(source.stats().decoded_samples, 2)
            await events.aclose()

    async def test_close_unblocks_waiting_consumer(self):
        source = SensorStreamSource(ReceiverConfig(bind_host="127.0.0.1", bind_port=0))
        async with source:
            events = source.events()
            waiting = asyncio.create_task(anext(events))
            await asyncio.sleep(0)
            with self.assertRaises(RuntimeError):
                await anext(source.events())
            await source.aclose()
            with self.assertRaises(StopAsyncIteration):
                await asyncio.wait_for(waiting, 2)
        await source.aclose()

    async def test_cancel_consumer_cleans_up(self):
        source = SensorStreamSource(ReceiverConfig(bind_host="127.0.0.1", bind_port=0))

        async def consume():
            async with source:
                async for _ in source.events():
                    pass

        task = asyncio.create_task(consume())
        await self.wait_until(lambda: source._server is not None)
        await self.connect(source)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(source._closed)
        self.assertFalse(source._clients)

    async def test_overflow_is_fatal(self):
        config = ReceiverConfig(bind_host="127.0.0.1", bind_port=0, queue_capacity=1)
        async with SensorStreamSource(config) as source:
            writer = await self.connect(source)
            writer.write(frame(imu_payload()))
            await writer.drain()
            await self.wait_until(lambda: source._failure is not None)
            with self.assertRaises(BufferOverflowError):
                await anext(source.events())

    async def test_drop_oldest_keeps_descriptor(self):
        config = ReceiverConfig(
            bind_host="127.0.0.1",
            bind_port=0,
            queue_capacity=2,
            overflow_policy="drop_oldest",
        )
        async with SensorStreamSource(config) as source:
            writer = await self.connect(source)
            writer.write(frame(imu_payload(123), 1) + frame(imu_payload(124), 2))
            await writer.drain()
            await self.wait_until(lambda: source.stats().decoded_samples == 2)
            events = source.events()
            self.assertIsInstance(await anext(events), StreamStarted)
            self.assertEqual((await anext(events)).source_time_ns, 124)
            self.assertEqual(source.stats().dropped_samples, 1)
            await events.aclose()

    async def test_malformed_unsupported_and_regression(self):
        async with SensorStreamSource(
            ReceiverConfig(bind_host="127.0.0.1", bind_port=0)
        ) as source:
            writer = await self.connect(source)
            writer.write(
                frame(b"\xff", 1)
                + frame(b"\x2a\x00", 2)
                + frame(imu_payload(10), 3)
                + frame(imu_payload(9), 4)
            )
            await writer.drain()
            await self.wait_until(lambda: source.stats().decoded_samples == 2)
            stats = source.stats()
            self.assertEqual(stats.malformed_messages, 1)
            self.assertEqual(stats.unsupported_messages, 1)
            self.assertEqual(stats.timestamp_regressions, 1)

    async def test_protocol_error_does_not_stop_listener(self):
        async with SensorStreamSource(
            ReceiverConfig(bind_host="127.0.0.1", bind_port=0)
        ) as source:
            writer = await self.connect(source)
            writer.write(HEADER.pack(1, 0, 1, 0xFFFFFFFF))
            await writer.drain()
            await self.wait_until(lambda: source.stats().protocol_errors == 1)
            await self.wait_until(lambda: not source._clients)
            self.assertTrue(source._server.is_serving())

    async def test_second_device_rejected(self):
        async with SensorStreamSource(
            ReceiverConfig(bind_host="127.0.0.1", bind_port=0)
        ) as source:
            await self.connect(source)
            await self.wait_until(lambda: source.stats().connected)
            await self.connect(source)
            await self.wait_until(lambda: source.stats().rejected_connections == 1)
            self.assertTrue(source.stats().connected)


class UsbSourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_discovery_connect_and_delivery(self):
        tasks = set()

        async def daemon(reader, writer):
            task = asyncio.current_task()
            tasks.add(task)
            try:
                length, version, kind, tag = MUX_HEADER.unpack(
                    await reader.readexactly(16)
                )
                request = plistlib.loads(await reader.readexactly(length - 16))
                if request["MessageType"] == "ListDevices":
                    reply = {
                        "DeviceList": [
                            {
                                "DeviceID": 7,
                                "Properties": {
                                    "ConnectionType": "USB",
                                    "SerialNumber": "test",
                                },
                            }
                        ]
                    }
                else:
                    self.assertEqual(request["PortNumber"], socket.htons(2345))
                    reply = {"MessageType": "Result", "Number": 0}
                payload = plistlib.dumps(reply)
                writer.write(MUX_HEADER.pack(16 + len(payload), 1, 8, tag) + payload)
                if request["MessageType"] == "Connect":
                    writer.write(usb_frame(imu_payload()))
                await writer.drain()
                await reader.read()
            finally:
                writer.close()
                await writer.wait_closed()
                tasks.discard(task)

        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "mux.sock")
            server = await asyncio.start_unix_server(daemon, path)

            async def connect(port, udid):
                return await connect_phone(port, udid, path)

            try:
                with patch("sensor_stream.source.connect_phone", connect):
                    async with SensorStreamSource(
                        ReceiverConfig(transport="usb")
                    ) as source:
                        events = source.events()
                        await asyncio.wait_for(anext(events), 2)
                        self.assertIsInstance(
                            await asyncio.wait_for(anext(events), 2), ImuSample
                        )
                        await events.aclose()
            finally:
                server.close()
                await server.wait_closed()
                if tasks:
                    await asyncio.gather(*tuple(tasks))


class CameraRecordingTests(unittest.TestCase):
    def test_record_integrity_and_missing_file(self):
        sample = CameraSample(
            "camera", "session", 123, 456, 789, 16, 12, "jpeg", b"image bytes"
        )
        with tempfile.TemporaryDirectory() as root:
            with SessionRecorder(Path(root), ReceiverConfig()) as recorder:
                recorder.write(sample)
            metadata, events = load_capture(recorder.path)
            row = events[0]
            target = recorder.path / row["image_path"]
            self.assertEqual(target.read_bytes(), sample.image_data)
            self.assertNotIn("image_data", row)
            result = summarize(metadata, events)
            self.assertEqual(result["camera_samples"], 1)
            self.assertEqual(result["camera_streams"][0]["file_checks"], {"ok": 1})
            self.assertFalse(result["warnings"])
            target.write_bytes(b"corrupt")
            _, events = load_capture(recorder.path)
            self.assertTrue(summarize(metadata, events)["warnings"])
            row["image_path"] = "../../outside.jpg"
            (recorder.path / "events.jsonl").write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "escapes"):
                load_capture(recorder.path)

    def test_thumbnail_plot(self):
        try:
            from PIL import Image
            import matplotlib
        except ImportError:
            self.skipTest("Optional plot dependencies unavailable")
        buffer = io.BytesIO()
        Image.new("RGB", (16, 12), "red").save(buffer, format="JPEG")
        with tempfile.TemporaryDirectory() as root:
            with SessionRecorder(Path(root), ReceiverConfig()) as recorder:
                recorder.write(
                    CameraSample(
                        "camera",
                        "session",
                        123,
                        456,
                        789,
                        16,
                        12,
                        "jpeg",
                        buffer.getvalue(),
                    )
                )
            _, events = load_capture(recorder.path)
            output = recorder.path / "analysis.png"
            plot_capture(events, output, capture=recorder.path)
            self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_camera_byte_bound(self):
        source = SensorStreamSource(ReceiverConfig(max_buffer_bytes=5))
        source._emit(
            CameraSample("camera", "session", 123, 456, 789, 16, 12, "jpeg", b"123456")
        )
        self.assertIsInstance(source._failure, BufferOverflowError)


class CameraSourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_receive_and_drain_camera(self):
        async with SensorStreamSource(
            ReceiverConfig(bind_host="127.0.0.1", bind_port=0)
        ) as source:
            _, writer = await asyncio.open_connection("127.0.0.1", source.port)
            try:
                wire = frame(camera_payload())
                writer.write(wire[:7])
                await writer.drain()
                writer.write(wire[7:])
                await writer.drain()
                async with asyncio.timeout(2):
                    while source.stats().decoded_camera_samples != 1:
                        await asyncio.sleep(0.005)
                rows = await source.aclose(drain=True)
                self.assertEqual(rows[0].descriptor.kind, "camera")
                self.assertIsInstance(rows[1], CameraSample)
                self.assertEqual(rows[1].width, 16)
                self.assertEqual(source.stats().decoded_samples, 1)
            finally:
                writer.close()
                await writer.wait_closed()


class GpsRecordingTests(unittest.TestCase):
    def test_record_inspect_and_plot(self):
        sample = GpsSample(
            "gps",
            "session",
            123,
            456,
            789,
            40.0,
            -3.0,
            650.0,
            0,
            1,
            (4.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 16.0),
            2,
        )
        with tempfile.TemporaryDirectory() as root:
            with SessionRecorder(Path(root), ReceiverConfig()) as recorder:
                recorder.write(sample)
            metadata, events = load_capture(recorder.path)
            result = summarize(metadata, events)
            self.assertEqual(result["gps_samples"], 1)
            self.assertFalse(result["warnings"])
            self.assertEqual(metadata["gps_samples_written"], 1)
            self.assertEqual(result["gps_streams"][0]["valid_fix_samples"], 1)
            self.assertEqual(covariance_sigma(events[0]), (2.0, 3.0, 4.0))
            events[0]["position_covariance_type"] = 0
            self.assertIsNone(covariance_sigma(events[0]))
            try:
                import matplotlib
            except ImportError:
                return
            plot_capture(events, recorder.path / "gps.png")
            self.assertEqual(
                (recorder.path / "gps.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n"
            )


class GpsSourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_mixed_streams_have_separate_clocks_and_descriptors(self):
        async with SensorStreamSource(
            ReceiverConfig(bind_host="127.0.0.1", bind_port=0)
        ) as source:
            _, writer = await asyncio.open_connection("127.0.0.1", source.port)
            try:
                writer.write(
                    frame(imu_payload(1000), 1)
                    + frame(gps_payload(500), 2)
                    + frame(imu_payload(1100), 3)
                )
                await writer.drain()
                async with asyncio.timeout(2):
                    while source.stats().decoded_samples != 3:
                        await asyncio.sleep(0.005)
                events = source.events()
                rows = [await anext(events) for _ in range(5)]
                await events.aclose()
                self.assertEqual(rows[0].descriptor.kind, "imu")
                self.assertEqual(rows[2].descriptor.kind, "gps")
                self.assertIsInstance(rows[1], ImuSample)
                self.assertIsInstance(rows[3], GpsSample)
                self.assertEqual(source.stats().timestamp_regressions, 0)
                self.assertEqual(source.stats().decoded_imu_samples, 2)
                self.assertEqual(source.stats().decoded_gps_samples, 1)
            finally:
                writer.close()
                await writer.wait_closed()


class DepthRecordingTests(unittest.TestCase):
    def test_record_inspect_and_plot(self):
        record = decode_message(depth_payload())
        sample = DepthSample(
            "depth", "session", 123, 456, 789, 2, 2, "16UC1", record.depth_data
        )
        with tempfile.TemporaryDirectory() as directory:
            with SessionRecorder(Path(directory), ReceiverConfig()) as recorder:
                recorder.write(sample)
            metadata, events = load_capture(recorder.path)
            result = summarize(metadata, events)
            self.assertEqual(result["depth_samples"], 1)
            self.assertEqual(metadata["depth_samples_written"], 1)
            self.assertEqual(result["depth_streams"][0]["zero_pixels"], 1)
            self.assertEqual(
                result["depth_streams"][0]["center_positive_median_range_raw"],
                [2000, 2000],
            )
            self.assertFalse(result["warnings"])
            try:
                import matplotlib
            except ImportError:
                return
            plot_capture(events, recorder.path / "analysis.png", capture=recorder.path)
            self.assertEqual(
                (recorder.path / "analysis.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n"
            )


class DepthSourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_receive_and_drain(self):
        async with SensorStreamSource(
            ReceiverConfig(bind_host="127.0.0.1", bind_port=0)
        ) as source:
            _, writer = await asyncio.open_connection("127.0.0.1", source.port)
            try:
                writer.write(frame(depth_payload()))
                await writer.drain()
                async with asyncio.timeout(2):
                    while source.stats().decoded_depth_samples != 1:
                        await asyncio.sleep(0.005)
                events = await source.aclose(drain=True)
                self.assertEqual(events[0].descriptor.kind, "depth")
                self.assertIsInstance(events[1], DepthSample)
            finally:
                writer.close()
                await writer.wait_closed()


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
                    yield camera_frame(i)

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
            self.assertEqual(submitted, [camera_frame(i) for i in range(3)])
            self.assertEqual(closed, [True])
