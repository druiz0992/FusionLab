from dataclasses import dataclass
from math import isfinite

from google.protobuf.message import DecodeError

from ..models import Quaternion, Vector3
from .proto.imu_pb2 import Envelope


@dataclass(frozen=True, slots=True)
class ImuRecord:
    timestamp: int
    frame_id: str
    acceleration: Vector3
    angular_velocity: Vector3
    orientation: Quaternion | None


@dataclass(frozen=True, slots=True)
class GpsRecord:
    timestamp: int
    frame_id: str
    latitude: float | None
    longitude: float | None
    altitude: float | None
    status: int | None
    service: int | None
    covariance: tuple[float | None, ...]
    covariance_type: int


@dataclass(frozen=True, slots=True)
class CameraRecord:
    timestamp: int
    frame_id: str
    width: int
    height: int
    encoding: str
    image_data: bytes


@dataclass(frozen=True, slots=True)
class DepthRecord:
    timestamp: int
    frame_id: str
    width: int
    height: int
    encoding: str
    depth_data: bytes


def decode_message(
    payload: bytes,
) -> ImuRecord | GpsRecord | CameraRecord | DepthRecord | None:
    """Decode IMU/GPS; return None for unsupported types, ValueError if invalid."""
    message = Envelope()
    try:
        message.ParseFromString(payload)
    except DecodeError as exc:
        raise ValueError("Malformed Protobuf") from exc
    # The envelope selects one sensor payload. These records contain wire data;
    # session identity and laptop receive timestamps are added by the source.
    if message.WhichOneof("payload") == "depth":
        depth = message.depth
        if not depth.width or not depth.height or not depth.depth_data:
            raise ValueError("Depth requires positive dimensions and payload")
        bytes_per_pixel = {"16uc1": 2, "32fc1": 4}.get(depth.encoding.lower())
        if (
            bytes_per_pixel
            and len(depth.depth_data) != depth.width * depth.height * bytes_per_pixel
        ):
            raise ValueError("Depth payload length does not match dimensions/encoding")
        return DepthRecord(
            depth.timestamp,
            depth.frame_id,
            depth.width,
            depth.height,
            depth.encoding,
            bytes(depth.depth_data),
        )
    if message.WhichOneof("payload") == "camera":
        camera = message.camera
        if not camera.width or not camera.height or not camera.image_data:
            raise ValueError("Camera requires positive dimensions and image payload")
        return CameraRecord(
            camera.timestamp,
            camera.frame_id,
            camera.width,
            camera.height,
            camera.encoding,
            bytes(camera.image_data),
        )
    if message.WhichOneof("payload") == "gps":
        gps = message.gps
        status = gps.status.status if gps.HasField("status") else None

        # Retain unavailable GPS values as None so recordings can use strict JSON
        # without inventing a numeric fix for NaN or infinity.
        def finite(value: float) -> float | None:
            return value if isfinite(value) else None

        lat, lon, alt = (finite(v) for v in (gps.latitude, gps.longitude, gps.altitude))
        if status is not None and status >= 0:
            if (
                lat is None
                or lon is None
                or not -90 <= lat <= 90
                or not -180 <= lon <= 180
            ):
                raise ValueError("Invalid coordinates in GPS fix")
        return GpsRecord(
            gps.timestamp,
            gps.frame_id,
            lat,
            lon,
            alt,
            status,
            gps.status.service if gps.HasField("status") else None,
            tuple(finite(v) for v in gps.position_covariance),
            gps.position_covariance_type,
        )
    if message.WhichOneof("payload") != "imu":
        return None
    imu = message.imu
    if not imu.HasField("accel") or not imu.HasField("gyro"):
        raise ValueError("IMU requires acceleration and angular velocity")
    acceleration = (imu.accel.x, imu.accel.y, imu.accel.z)
    velocity = (imu.gyro.x, imu.gyro.y, imu.gyro.z)
    orientation = None
    if imu.HasField("orientation"):
        q = imu.orientation
        orientation = (q.x, q.y, q.z, q.w)
    if not all(isfinite(v) for v in (*acceleration, *velocity, *(orientation or ()))):
        raise ValueError("Nonfinite IMU value")
    return ImuRecord(imu.timestamp, imu.frame_id, acceleration, velocity, orientation)
