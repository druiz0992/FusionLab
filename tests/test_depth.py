import asyncio
import struct
import tempfile
import unittest
from pathlib import Path
from sensor_stream import DepthSample, ReceiverConfig, SensorStreamSource
from sensor_stream.protocol.proto.imu_pb2 import Envelope
from sensor_stream.protocol.decoder import decode_message, DepthRecord
from sensor_stream.recording import SessionRecorder
from sensor_stream.inspection import load_capture, summarize, plot_capture
from test_sensor_stream import frame


def payload():
    message = Envelope()
    depth = message.depth
    depth.timestamp = 123
    depth.width = 2
    depth.height = 2
    depth.encoding = "16UC1"
    depth.frame_id = "depth"
    depth.depth_data = struct.pack("<4H", 0, 1000, 2000, 3000)
    return message.SerializeToString()


class DepthTests(unittest.TestCase):
    def test_decode_and_length_validation(self):
        record = decode_message(payload())
        self.assertIsInstance(record, DepthRecord)
        message = Envelope()
        message.ParseFromString(payload())
        message.depth.width = 3
        with self.assertRaises(ValueError):
            decode_message(message.SerializeToString())

    def test_record_inspect_and_plot(self):
        record = decode_message(payload())
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
                writer.write(frame(payload()))
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
