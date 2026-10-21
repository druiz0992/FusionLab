import asyncio
import io
import json
import tempfile
import unittest
from pathlib import Path

from sensor_stream import (
    CameraSample,
    ReceiverConfig,
    SensorStreamSource,
    BufferOverflowError,
)
from sensor_stream.protocol.decoder import decode_message, CameraRecord
from sensor_stream.protocol.proto.imu_pb2 import Envelope
from sensor_stream.recording import SessionRecorder
from sensor_stream.inspection import load_capture, summarize, plot_capture
from test_sensor_stream import frame


def camera_payload(data=b"\xff\xd8test\xff\xd9", timestamp=123):
    message = Envelope()
    camera = message.camera
    camera.timestamp = timestamp
    camera.frame_id = "camera"
    camera.width = 16
    camera.height = 12
    camera.encoding = "jpeg"
    camera.image_data = data
    return message.SerializeToString()


class CameraTests(unittest.TestCase):
    def test_independent_wire_fixture(self):
        # Envelope tag 2, timestamp=1,width=16,height=12, encoding=jpeg,data=abc.
        payload = b"\x08\x01\x10\x10\x18\x0c\x22\x04jpeg\x2a\x03abc"
        record = decode_message(b"\x12" + bytes([len(payload)]) + payload)
        self.assertIsInstance(record, CameraRecord)
        self.assertEqual(
            (record.width, record.height, record.image_data), (16, 12, b"abc")
        )
        with self.assertRaises(ValueError):
            decode_message(b"\x12\x00")

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
