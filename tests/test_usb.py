import asyncio
import plistlib
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from sensor_stream import ReceiverConfig, SensorStreamSource, ImuSample, ProtocolError
from sensor_stream.protocol.usb import (
    UsbParser,
    PEER_HEADER,
    LENGTH,
    MUX_HEADER,
    connect_phone,
)
from test_sensor_stream import imu_payload


def usb_frame(payload):
    body = LENGTH.pack(len(payload)) + payload
    return PEER_HEADER.pack(1, 200, 0, len(body)) + body


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
