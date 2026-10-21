# 4. Sensors and acquisition

Updated 2026-10-21 from the 2026-10-17 phone characterization and the
2026-10-18 acquisition reference. Capture commands and options are in the
[README](../README.md#capture-sensor-stream).

## Sensor conventions and limitations

These findings apply to the tested iPhone 17 Pro and SensorStream setup. Raw
recordings retain wire values; interpretations and unit conversions must be
explicit in processing.

| Input | Supported interpretation | Still unresolved |
| --- | --- | --- |
| IMU | Approximately 50 Hz; gyro rad/s; acceleration in g, experimentally supported as gravity compensated; reported device axes +X right, +Y top, +Z out of screen | Exact acquisition timing, bias/filtering, absolute reference axes and independent accuracy |
| Orientation | Hamilton x,y,z,w, body-to-reference interpretation supported by rotation/slide comparisons | Phone orientation is another estimate, not independent ground truth; app restart resets its reference |
| RGB | Stored 1920×1080 JPEGs; display rotation leaves estimator pixels unchanged; USB sustained about 30 FPS in tested captures | Exposure timing, rolling shutter, camera identity/processing stability and independent calibration validation |
| Rear depth | Tested 16UC1 little-endian values support millimetres: central medians 496 and 1020 at measured 0.50 m and 1.00 m | Axial versus radial depth, invalid/saturated values, RGB registration, confidence and other modes |
| GPS | Indoor reception and recording verified; fixes can be sparse and irregular | Outdoor accuracy, covariance-to-preview mapping, altitude datum and producer location sources |

Use 9.80665 m/s² per g only as an explicit acceleration conversion. It does not
calibrate bias or supply the specific force assumed by standard inertial
preintegration. Treat reconnects/gaps as estimator discontinuities unless their
reference continuity has been established.

Combined RGB/depth/IMU recording worked over USB. Tested Wi-Fi RGB accumulated
backlog. Nearest source timestamps are useful for inspection but do not establish
exposure synchronization; RGB and depth must not be paired by frame index.
Approximate edge agreement after resizing is not calibrated registration.

The selected RGB fit remains provisional and `needs_review` (see
[camera calibration](5_camera_and_visual_odometry.md#camera-calibration)).
Camera–IMU extrinsics/timing and RGB–depth geometry are not established.
IMU, depth and GPS currently supply no constraints to the camera-only VO estimator.

The original characterization captures are under `data/archive/sensor_validation/`;
use [data/catalog.json](../data/catalog.json) to map historical capture paths to
their archive locations. Those recordings preserve observations, not a task list.

## Acquisition scope

`sensor_stream` receives and records iPhone SensorStream IMU, GPS, encoded RGB
images and rear depth maps without ROS2. Wi-Fi uses TCP chunk assembly and a
Protobuf envelope; USB uses usbmuxd and PeerTalk framing. Unsupported variants
are counted rather than interpreted as valid samples.

The receiver preserves wire measurements. Units, gravity interpretation,
calibration, interpolation and localization belong in processing modules.

## Python API

```python
import asyncio
from sensor_stream import ImuSample, SensorStreamSource

async def main():
    async with SensorStreamSource() as source:
        async for event in source.events():
            if isinstance(event, ImuSample):
                print(event.acceleration)

asyncio.run(main())
```

`ReceiverConfig` selects transport, connection, memory and overflow behavior.
The async iterator carries stream declarations, typed samples and discontinuities.
There is one consumer, bounded queue/buffer handling and source statistics.
Reconnects carry distinct session identity; they do not imply a continuous
estimator coordinate frame. See the source package for exact dataclass fields.

Sample families include `ImuSample`, `GpsSample`, `CameraSample` and `DepthSample`.
Source timestamps are preserved alongside local monotonic/Unix receive times.
Source-clock meaning and exposure timing are not independently calibrated.

## Recording and shutdown

`listen --record` creates a new timestamped capture with metadata, JSONL events
and binary camera/depth assets. Hashes and byte counts support integrity checks.
Ctrl+C and timed stops drain decoded queued samples before finalizing metadata;
undecoded network bytes and abrupt process termination are outside that guarantee.
A normal Ctrl+C recording can have status `interrupted`.

## What is excluded

Automatic sensor synchronization, camera/depth/IMU calibration, point-cloud
support, localization, and general multisubscriber/network APIs are not supplied
by this acquisition layer. Live RGB preview can drop display frames while the
recorder retains delivered samples. Local skipping does not remove phone/network
backlog. USB was the reliable tested choice for combined high-rate recordings.

## Wire protocol

USB uses the local usbmuxd plist protocol for USB-only discovery and Connect,
then switches to SensorStream PeerTalk frames: 16-byte big-endian header
(version/type/tag/payload size), type 200 data carrying a 4-byte big-endian
Protobuf length, and type 201 heartbeat. Payloads share the existing decoder.
The laptop connects to the phone port (default 2345), rather than listening.
Discovery/connect and retries are cancellation-aware; reconnects create sessions.
No pairing records are read or modified by this receiver.
Protocol references: the official SensorStream receiver and
[libusbmuxd client](https://github.com/libimobiledevice/libusbmuxd/blob/master/src/libusbmuxd.c).

Wire decoding and explicit processing interpretations are kept separate so
inferred conventions do not silently alter recorded data.

The framing implementation and minimal IMU schema were independently written
for interoperability using the public driver as protocol reference:

- https://github.com/martijnhabers/sensorstream_driver/blob/main/src/protocol_handler.cpp
- https://github.com/martijnhabers/sensorstream_driver/blob/main/proto/sensor.proto

References inspected on 2026-10-17. An exact upstream commit has not yet been
pinned; the decoder must be validated against the installed phone app.
No upstream implementation or full schema has been vendored.

The local schema preserves IMU, GPS and camera field numbers and wire types. Other
known message variants are opaque bytes so they can be skipped without implementing
their schemas. The schema stays in `imu.proto` for compatibility with the existing
generation path; it also contains the GPS envelope variant. Generated `imu_pb2.py`
is checked in; ordinary users only need the Python Protobuf runtime. Regenerate
after changing the local schema with:

```bash
protoc -I src/sensor_stream/protocol/proto \
  --python_out=src/sensor_stream/protocol/proto \
  src/sensor_stream/protocol/proto/imu.proto
```

The receiver accepts one device, starts a session per connection, and publishes
descriptors per frame identifier. Timestamp regressions increment a counter;
they do not silently adjust clocks or imply a confirmed phone restart.

Malformed Protobuf messages are counted and skipped. Invalid framing or assembly
timeouts close the client connection and increment `protocol_errors`; the
listener remains available for reconnection. Queue overflow is fatal by default.
The optional `drop_oldest` policy discards only samples, preserving control events;
a queue filled entirely with control events still fails explicitly.

Buffer bounds cover chunk bytes, message bytes, message count, and chunk count.
An idle connection is allowed, but partial chunks or messages have a deadline.
Frames are emitted in message-completion order, without timestamp sorting.
`messages_received` counts reconstructed messages, not TCP packets.
