import asyncio
import json
import struct
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from sensor_stream import GpsSample, ImuSample, ReceiverConfig, SensorStreamSource
from sensor_stream.protocol.decoder import decode_message, GpsRecord
from sensor_stream.protocol.proto.imu_pb2 import Envelope
from sensor_stream.recording import SessionRecorder
from sensor_stream.inspection import load_capture, summarize, plot_capture
from sensor_stream.gps_inspection import covariance_sigma
from test_sensor_stream import frame, imu_payload


def gps_payload(timestamp=123, no_fix=False):
    message = Envelope()
    gps = message.gps
    gps.timestamp = timestamp
    gps.frame_id = "gps_link"
    gps.status.status = -1 if no_fix else 0
    gps.status.service = 1
    gps.latitude = 40.0
    gps.longitude = -3.0
    gps.altitude = 650.0
    gps.position_covariance.extend([4.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 16.0])
    gps.position_covariance_type = 2
    return message.SerializeToString()


class GpsTests(unittest.TestCase):
    def test_independent_wire_fixture(self):
        # Envelope tag 4; GPS status submessage, double coordinates and altitude.
        payload = (
            b"\x08\x7b\x1a\x00"
            + b"\x21"
            + struct.pack("<d", 40.0)
            + b"\x29"
            + struct.pack("<d", -3.0)
            + b"\x31"
            + struct.pack("<d", 650.0)
        )
        record = decode_message(b"\x22" + bytes([len(payload)]) + payload)
        self.assertIsInstance(record, GpsRecord)
        self.assertEqual(
            (record.latitude, record.longitude, record.altitude), (40.0, -3.0, 650.0)
        )
        self.assertEqual(record.status, 0)

    def test_missing_status_nofix_and_invalid_fix(self):
        record = decode_message(b"\x22\x00")
        self.assertIsNone(record.status)
        message = Envelope()
        message.ParseFromString(gps_payload(no_fix=True))
        message.gps.latitude = float("nan")
        message.gps.altitude = float("nan")
        record = decode_message(message.SerializeToString())
        self.assertIsNone(record.latitude)
        self.assertIsNone(record.altitude)
        message.gps.status.status = 0
        with self.assertRaises(ValueError):
            decode_message(message.SerializeToString())

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
