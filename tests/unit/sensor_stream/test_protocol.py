"""Wire framing, Protobuf decoding and USB PeerTalk parsing."""

import math
import struct
import unittest

from sensor_stream import ProtocolError, ReceiverConfig
from sensor_stream.protocol.decoder import (
    decode_message,
    CameraRecord,
    GpsRecord,
    DepthRecord,
)
from sensor_stream.protocol.framing import ChunkParser
from sensor_stream.protocol.proto.imu_pb2 import Envelope
from sensor_stream.protocol.usb import UsbParser, PEER_HEADER

from support.wire import imu_payload, frame, gps_payload, depth_payload, usb_frame


class FramingTests(unittest.TestCase):
    def test_continuous_partial_chunks_get_individual_deadlines(self):
        parser = ChunkParser(ReceiverConfig(assembly_timeout_s=1))
        first = frame(b"abc", 1)
        second = frame(b"def", 2)
        third = frame(b"ghi", 3)
        self.assertEqual(parser.feed(first[:5], 0), [])
        self.assertEqual(parser.feed(first[5:] + second[:5], 0.6), [b"abc"])
        self.assertEqual(parser.feed(second[5:] + third[:5], 1.2), [b"def"])
        self.assertEqual(parser.feed(third[5:], 1.8), [b"ghi"])

    def test_fragmentation_and_coalescing(self):
        wire = frame(b"abc", 1, 0, 0) + frame(b"def", 1, 1) + frame(b"ghi", 2)
        parser = ChunkParser(ReceiverConfig())
        result = []
        for byte in wire:
            result.extend(parser.feed(bytes([byte]), 0))
        self.assertEqual(result, [b"abcdef", b"ghi"])
        parser.finish()
        self.assertEqual(ChunkParser(ReceiverConfig()).feed(wire, 0), result)

    def test_interleaved_messages(self):
        parser = ChunkParser(ReceiverConfig())
        wire = frame(b"a", 1, 0, 0) + frame(b"b", 2) + frame(b"c", 1, 1)
        self.assertEqual(parser.feed(wire, 0), [b"b", b"ac"])

    def test_bounds(self):
        cases = [
            (ReceiverConfig(max_chunk_bytes=1), frame(b"xx")),
            (
                ReceiverConfig(max_message_bytes=1),
                frame(b"a", 1, 0, 0) + frame(b"b", 1, 1),
            ),
            (
                ReceiverConfig(max_inflight_messages=1),
                frame(b"a", 1, 0, 0) + frame(b"b", 2, 0, 0),
            ),
            (ReceiverConfig(max_chunks_per_message=1), frame(b"x", 1, 1)),
            (ReceiverConfig(), frame(b"x", last=2)),
        ]
        for config, wire in cases:
            with self.subTest(config=config), self.assertRaises(ProtocolError):
                ChunkParser(config).feed(wire, 0)

    def test_timeout_and_truncation(self):
        for wire in (b"x", frame(b"x", 1, 0, 0)):
            parser = ChunkParser(ReceiverConfig(assembly_timeout_s=1))
            parser.feed(wire, 0)
            with self.assertRaises(ProtocolError):
                parser.check_timeout(1)
            with self.assertRaises(ProtocolError):
                parser.finish()

    def test_conflicting_duplicate(self):
        parser = ChunkParser(ReceiverConfig())
        parser.feed(frame(b"x", 1, 0, 0), 0)
        with self.assertRaises(ProtocolError):
            parser.feed(frame(b"y", 1, 0, 0), 0)


class DecoderTests(unittest.TestCase):
    def test_known_wire_fixture(self):
        # Independently encoded: envelope field 1; IMU timestamp, accel and gyro.
        vec = b"\x0d" + struct.pack("<f", 2.5)
        imu = b"\x08\x7b\x12\x05" + vec + b"\x1a\x00"
        record = decode_message(b"\x0a" + bytes([len(imu)]) + imu)
        self.assertEqual(record.timestamp, 123)
        self.assertEqual(record.acceleration, (2.5, 0.0, 0.0))
        self.assertIsNone(record.orientation)

    def test_imu_and_unsupported(self):
        record = decode_message(imu_payload())
        self.assertEqual(record.angular_velocity, (0.0, 0.0, -2.0))
        self.assertEqual(record.orientation, (0.0, 0.0, 0.0, 1.0))
        self.assertIsNone(decode_message(b"\x2a\x00"))

    def test_invalid(self):
        for payload in (b"\xff", b"\x0a\x00"):
            with self.assertRaises(ValueError):
                decode_message(payload)
        message = Envelope()
        message.ParseFromString(imu_payload())
        message.imu.accel.x = math.nan
        with self.assertRaises(ValueError):
            decode_message(message.SerializeToString())


class CameraWireTests(unittest.TestCase):
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


class GpsWireTests(unittest.TestCase):
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


class DepthWireTests(unittest.TestCase):
    def test_decode_and_length_validation(self):
        record = decode_message(depth_payload())
        self.assertIsInstance(record, DepthRecord)
        message = Envelope()
        message.ParseFromString(depth_payload())
        message.depth.width = 3
        with self.assertRaises(ValueError):
            decode_message(message.SerializeToString())


class UsbParserTests(unittest.TestCase):
    def test_fragmented_frames_and_heartbeat(self):
        parser = UsbParser(ReceiverConfig())
        wire = PEER_HEADER.pack(1, 201, 0, 0) + usb_frame(b"abc") + usb_frame(b"def")
        results = []
        for byte in wire:
            results.extend(parser.feed(bytes([byte]), 0))
        self.assertEqual(results, [b"abc", b"def"])
        parser.finish()

    def test_invalid_lengths(self):
        with self.assertRaises(ProtocolError):
            UsbParser(ReceiverConfig(max_message_bytes=4)).feed(usb_frame(b"12345"), 0)
        with self.assertRaises(ProtocolError):
            UsbParser(ReceiverConfig()).feed(
                PEER_HEADER.pack(1, 200, 0, 5) + b"\x00\x00\x00\x09x", 0
            )
