import asyncio
import math
import struct
import unittest
from unittest.mock import AsyncMock, Mock, patch

from sensor_stream import (
    BufferOverflowError,
    ImuSample,
    ProtocolError,
    ReceiverConfig,
    SensorStreamSource,
    SourceDiscontinuity,
    StreamStarted,
)
from sensor_stream.protocol.decoder import decode_message
from sensor_stream.protocol.framing import ChunkParser, HEADER
from sensor_stream.protocol.proto.imu_pb2 import Envelope


def imu_payload(timestamp: int = 123) -> bytes:
    message = Envelope()
    message.imu.timestamp = timestamp
    message.imu.accel.x = 1.5
    message.imu.gyro.z = -2.0
    message.imu.orientation.w = 1.0
    message.imu.frame_id = "phone"
    return message.SerializeToString()


def frame(payload: bytes, message_id: int = 1, index: int = 0, last: int = 1) -> bytes:
    return HEADER.pack(message_id, index, last, len(payload)) + payload


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


if __name__ == "__main__":
    unittest.main()
