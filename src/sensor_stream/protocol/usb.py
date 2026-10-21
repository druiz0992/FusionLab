"""usbmuxd plist connection and bounded SensorStream PeerTalk framing."""

import asyncio
import plistlib
import socket
from struct import Struct
from .framing import ChunkParser
from ..errors import ProtocolError, TransportError

# usbmuxd uses little-endian headers; PeerTalk and its length use network order.
MUX_HEADER = Struct("<IIII")
PEER_HEADER = Struct("!IIII")
LENGTH = Struct("!I")


async def exchange(reader, writer, request: dict) -> dict:
    payload = plistlib.dumps(
        {
            "ClientVersionString": "FusionLab",
            "ProgName": "sensor-stream",
            "kLibUSBMuxVersion": 3,
            **request,
        }
    )
    writer.write(MUX_HEADER.pack(16 + len(payload), 1, 8, 1) + payload)
    await writer.drain()
    header = await reader.readexactly(16)
    size, version, kind, tag = MUX_HEADER.unpack(header)
    if not 16 <= size <= 1024 * 1024 or version != 1 or kind != 8 or tag != 1:
        raise TransportError("Invalid usbmuxd reply header")
    response = plistlib.loads(await reader.readexactly(size - 16))
    if not isinstance(response, dict):
        raise TransportError("Invalid usbmuxd plist reply")
    return response


async def list_devices(path="/var/run/usbmuxd") -> list[dict]:
    reader, writer = await asyncio.open_unix_connection(path)
    try:
        async with asyncio.timeout(5):
            result = await exchange(reader, writer, {"MessageType": "ListDevices"})
        return [
            d
            for d in result.get("DeviceList", [])
            if d.get("Properties", {}).get("ConnectionType") == "USB"
        ]
    finally:
        writer.close()
        await writer.wait_closed()


async def connect_phone(port: int, udid: str | None = None, path="/var/run/usbmuxd"):
    devices = await list_devices(path)
    if udid:
        devices = [
            d for d in devices if d.get("Properties", {}).get("SerialNumber") == udid
        ]
    if not devices:
        raise TransportError(
            "No matching USB iPhone detected; connect, unlock and trust this computer"
        )
    if len(devices) > 1:
        raise TransportError("Multiple USB devices; select one with --usb-udid")
    reader, writer = await asyncio.open_unix_connection(path)
    try:
        async with asyncio.timeout(5):
            result = await exchange(
                reader,
                writer,
                {
                    "MessageType": "Connect",
                    "DeviceID": devices[0]["DeviceID"],
                    "PortNumber": socket.htons(port),
                },
            )
        if result.get("Number") != 0:
            raise TransportError(
                f"USB connection refused (usbmuxd result {result.get('Number')}); enable SensorStream USB on port {port}"
            )
        # After a successful Connect reply, this socket carries the phone stream.
        return reader, writer
    except BaseException:
        writer.close()
        await writer.wait_closed()
        raise


class UsbParser(ChunkParser):
    def feed(self, data: bytes, now: float) -> list[bytes]:
        self.check_timeout(now)
        self.buffer.extend(data)
        messages = []
        consumed = False
        while len(self.buffer) >= 16:
            version, kind, tag, size = PEER_HEADER.unpack_from(self.buffer)
            if size > self.config.max_message_bytes + 4:
                raise ProtocolError(f"USB frame size {size} exceeds limit")
            if len(self.buffer) < 16 + size:
                break
            payload = bytes(self.buffer[16 : 16 + size])
            del self.buffer[: 16 + size]
            consumed = True
            if kind == 201:
                continue
            if kind != 200:
                raise ProtocolError(f"Unsupported USB PeerTalk frame type {kind}")
            if len(payload) < 4:
                raise ProtocolError("USB data frame missing Protobuf length")
            # Strip the inner Protobuf length only after checking it matches the frame.
            length = LENGTH.unpack_from(payload)[0]
            if length != len(payload) - 4:
                raise ProtocolError("USB Protobuf length mismatch")
            messages.append(payload[4:])
        if self.buffer:
            if consumed or self.partial_started is None:
                self.partial_started = now
        else:
            self.partial_started = None
        return messages
