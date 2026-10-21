"""SensorStream wire payloads and transport framing built from the local schema.

Payload builders return serialized Protobuf `Envelope` messages; `frame` wraps a
payload in a Wi-Fi TCP chunk header and `usb_frame` in a PeerTalk data frame.
"""

import struct

from sensor_stream.protocol.framing import HEADER
from sensor_stream.protocol.proto.imu_pb2 import Envelope
from sensor_stream.protocol.usb import LENGTH, PEER_HEADER


def imu_payload(timestamp: int = 123) -> bytes:
    message = Envelope()
    message.imu.timestamp = timestamp
    message.imu.accel.x = 1.5
    message.imu.gyro.z = -2.0
    message.imu.orientation.w = 1.0
    message.imu.frame_id = "phone"
    return message.SerializeToString()


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


def depth_payload():
    """A 2×2 16UC1 little-endian depth map: 0, 1000, 2000, 3000 raw units."""
    message = Envelope()
    depth = message.depth
    depth.timestamp = 123
    depth.width = 2
    depth.height = 2
    depth.encoding = "16UC1"
    depth.frame_id = "depth"
    depth.depth_data = struct.pack("<4H", 0, 1000, 2000, 3000)
    return message.SerializeToString()


def frame(payload: bytes, message_id: int = 1, index: int = 0, last: int = 1) -> bytes:
    """Wi-Fi TCP chunk: (message id, chunk index, last index, size) + payload."""
    return HEADER.pack(message_id, index, last, len(payload)) + payload


def usb_frame(payload):
    """PeerTalk type-200 data frame carrying a length-prefixed payload."""
    body = LENGTH.pack(len(payload)) + payload
    return PEER_HEADER.pack(1, 200, 0, len(body)) + body
