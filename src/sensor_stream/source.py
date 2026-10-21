"""Single-device asynchronous TCP source."""

import asyncio
from collections import deque
from dataclasses import replace
from time import monotonic, monotonic_ns, time_ns
from typing import AsyncIterator, Self
from uuid import uuid4

from .config import ReceiverConfig
from .errors import BufferOverflowError, ProtocolError, TransportError
from .models import (
    ImuSample,
    GpsSample,
    GpsStreamDescriptor,
    CameraSample,
    CameraStreamDescriptor,
    DepthSample,
    DepthStreamDescriptor,
    SensorEvent,
    SourceDiscontinuity,
    StreamDescriptor,
    StreamStarted,
)
from .protocol.decoder import decode_message, GpsRecord, CameraRecord, DepthRecord
from .protocol.framing import ChunkParser
from .stats import SourceStats
from .protocol.usb import UsbParser, connect_phone


class SensorStreamSource:
    def __init__(self, config: ReceiverConfig | None = None) -> None:
        self.config = config or ReceiverConfig()
        self._server: asyncio.Server | None = None
        self._opened = False
        self._closed = False
        self._consuming = False
        # Reception produces events; one consumer drains this bounded queue.
        # The wake event notifies that consumer of data, failure or shutdown.
        self._queue: deque[SensorEvent] = deque()
        self._wake = asyncio.Event()
        self._failure: Exception | None = None
        self._stats = SourceStats()
        self._clients: set[asyncio.Task[None]] = set()

    async def __aenter__(self) -> Self:
        if self._opened or self._closed:
            raise RuntimeError("Source instances have one lifecycle")
        self._opened = True
        if self.config.transport == "usb":
            task = asyncio.create_task(self._usb_loop())
            self._clients.add(task)
            task.add_done_callback(self._clients.discard)
            return self
        try:
            self._server = await asyncio.start_server(
                self._accept, self.config.bind_host, self.config.bind_port
            )
        except OSError as exc:
            self._closed = True
            raise TransportError(str(exc)) from exc
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    @property
    def port(self) -> int:
        """Bound port, including when port 0 requested an ephemeral listener."""
        if self.config.transport == "usb":
            return self.config.usb_port
        if self._server is None or not self._server.sockets:
            raise RuntimeError("Source is not listening")
        return int(self._server.sockets[0].getsockname()[1])

    def stats(self) -> SourceStats:
        # Return an immutable snapshot rather than exposing mutable receiver state.
        return replace(self._stats, queue_depth=len(self._queue))

    def _count(self, **increments: int) -> None:
        self._stats = replace(
            self._stats,
            **{
                name: getattr(self._stats, name) + amount
                for name, amount in increments.items()
            },
        )

    def _fail(self, error: Exception) -> None:
        if self._failure is None:
            self._failure = error
        if self._server:
            self._server.close()
        self._wake.set()

    def _emit(self, event: SensorEvent) -> None:
        if self._closed or self._failure:
            return
        payload_size = (
            len(event.image_data)
            if isinstance(event, CameraSample)
            else len(event.depth_data) if isinstance(event, DepthSample) else 0
        )
        if payload_size > self.config.max_buffer_bytes:
            self._fail(BufferOverflowError("Camera frame exceeds byte buffer capacity"))
            return
        # Enforce both event-count and image-byte limits: a few large frames can
        # exhaust memory even when the queue contains relatively few events.
        while (
            len(self._queue) >= self.config.queue_capacity
            or payload_size
            + sum(
                len(e.image_data) if isinstance(e, CameraSample) else len(e.depth_data)
                for e in self._queue
                if isinstance(e, (CameraSample, DepthSample))
            )
            > self.config.max_buffer_bytes
        ):
            if self.config.overflow_policy == "drop_oldest":
                # Drop measurements only. Stream declarations and disconnect markers
                # must remain so the consumer can interpret the remaining samples.
                for old in self._queue:
                    if isinstance(
                        old, (ImuSample, GpsSample, CameraSample, DepthSample)
                    ):
                        self._queue.remove(old)
                        self._count(dropped_samples=1)
                        break
                else:
                    self._fail(BufferOverflowError("Buffer full of control events"))
                    return
            else:
                self._fail(BufferOverflowError("Event buffer capacity exceeded"))
                return
        self._queue.append(event)
        self._stats = replace(
            self._stats,
            queue_high_water=max(self._stats.queue_high_water, len(self._queue)),
        )
        self._wake.set()

    async def events(self) -> AsyncIterator[SensorEvent]:
        if not self._opened:
            raise RuntimeError("Open source using async with first")
        if self._consuming:
            raise RuntimeError("Only one active consumer is supported")
        self._consuming = True
        try:
            while True:
                if self._closed:
                    return
                if self._failure:
                    raise self._failure
                if self._queue:
                    yield self._queue.popleft()
                    continue
                # No await occurs between checking the queue and clearing the flag,
                # so a producer cannot slip a notification between these two operations.
                self._wake.clear()
                await self._wake.wait()
        finally:
            self._consuming = False

    def _accept(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        # Reserve the device slot synchronously before scheduling its task.
        busy = bool(self._clients) or self._closed or self._failure is not None
        task = asyncio.create_task(self._receive(reader, writer, busy))
        # Track rejected connections for cleanup too.
        self._clients.add(task)
        task.add_done_callback(self._clients.discard)

    async def _usb_loop(self) -> None:
        while not self._closed and self._failure is None:
            try:
                reader, writer = await connect_phone(
                    self.config.usb_port, self.config.usb_udid
                )
            except (
                OSError,
                TransportError,
                TimeoutError,
                asyncio.IncompleteReadError,
                ValueError,
            ) as exc:
                self._stats = replace(self._stats, last_error=f"USB: {exc}")
            else:
                await self._receive(reader, writer, False)
            if not self._closed and self._failure is None:
                await asyncio.sleep(1)

    async def _receive(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, rejected: bool
    ) -> None:
        # A new connection gets a new session, even for the same device/frame ID.
        # This makes reconnects explicit to recording and integration consumers.
        session = uuid4().hex
        parser = (
            UsbParser(self.config)
            if self.config.transport == "usb"
            else ChunkParser(self.config)
        )
        descriptors: dict[
            tuple[str, str],
            StreamDescriptor
            | GpsStreamDescriptor
            | CameraStreamDescriptor
            | DepthStreamDescriptor,
        ] = {}
        previous_times: dict[tuple[str, str], int] = {}
        gps_first_receive: int | None = None
        gps_samples = 0
        camera_first_receive: int | None = None
        camera_samples = 0
        camera_first_source: int | None = None
        depth_first_receive: int | None = None
        depth_samples = 0
        disconnect_reason = "peer closed connection"
        first_receive: int | None = None
        samples = 0
        if not rejected:
            self._count(connections=1)
            self._stats = replace(
                self._stats,
                connected=True,
                observed_rate_hz=0.0,
                observed_gps_rate_hz=0.0,
                observed_camera_rate_hz=0.0,
                observed_depth_rate_hz=0.0,
                last_error=None,
            )
        try:
            if rejected:
                self._count(rejected_connections=1)
                return
            while not self._closed and self._failure is None:
                try:
                    data = await asyncio.wait_for(reader.read(65536), timeout=0.2)
                except TimeoutError:
                    parser.check_timeout(monotonic())
                    continue
                # A ready read can finish while aclose() is stopping producers.
                # Do not decode/count samples that _emit() would refuse after close.
                if self._closed or self._failure is not None:
                    break
                if not data:
                    parser.finish()
                    break
                # Capture laptop timing once per socket read. Monotonic time measures
                # intervals; Unix time provides a wall-clock reference for the recording.
                received, unix = monotonic_ns(), time_ns()
                self._count(bytes_received=len(data))
                self._stats = replace(self._stats, last_receive_monotonic_ns=received)
                for payload in parser.feed(data, monotonic()):
                    self._count(messages_received=1)
                    try:
                        record = decode_message(payload)
                    except ValueError:
                        self._count(malformed_messages=1)
                        continue
                    if record is None:
                        self._count(unsupported_messages=1)
                        continue
                    kind = (
                        "depth"
                        if isinstance(record, DepthRecord)
                        else (
                            "camera"
                            if isinstance(record, CameraRecord)
                            else "gps" if isinstance(record, GpsRecord) else "imu"
                        )
                    )
                    # Sensor kind is part of the key because different sensors may report
                    # the same frame name while still needing distinct stream declarations.
                    stream_key = (kind, record.frame_id)
                    previous_time = previous_times.get(stream_key)
                    if previous_time is not None and record.timestamp < previous_time:
                        self._count(timestamp_regressions=1)
                    previous_times[stream_key] = record.timestamp
                    descriptor = descriptors.get(stream_key)
                    if descriptor is None:
                        descriptor_type = (
                            DepthStreamDescriptor
                            if kind == "depth"
                            else (
                                CameraStreamDescriptor
                                if kind == "camera"
                                else (
                                    GpsStreamDescriptor
                                    if kind == "gps"
                                    else StreamDescriptor
                                )
                            )
                        )
                        descriptor = descriptor_type(
                            f"{session}:{kind}:{len(descriptors)}",
                            session,
                            record.frame_id,
                        )
                        descriptors[stream_key] = descriptor
                        # Emit the declaration before its first sample so consumers know the stream.
                        self._emit(StreamStarted(descriptor))
                    if isinstance(record, DepthRecord):
                        self._emit(
                            DepthSample(
                                descriptor.stream_id,
                                session,
                                record.timestamp,
                                received,
                                unix,
                                record.width,
                                record.height,
                                record.encoding,
                                record.depth_data,
                            )
                        )
                        self._count(decoded_samples=1, decoded_depth_samples=1)
                        depth_samples += 1
                        if depth_first_receive is None:
                            depth_first_receive = received
                        elapsed = (received - depth_first_receive) / 1e9
                        self._stats = replace(
                            self._stats,
                            observed_depth_rate_hz=(
                                (depth_samples - 1) / elapsed if elapsed > 0 else 0.0
                            ),
                        )
                        if self._failure:
                            break
                        continue
                    if isinstance(record, CameraRecord):
                        self._emit(
                            CameraSample(
                                descriptor.stream_id,
                                session,
                                record.timestamp,
                                received,
                                unix,
                                record.width,
                                record.height,
                                record.encoding,
                                record.image_data,
                            )
                        )
                        self._count(decoded_samples=1, decoded_camera_samples=1)
                        camera_samples += 1
                        if camera_first_receive is None:
                            camera_first_receive = received
                        if camera_first_source is None:
                            camera_first_source = record.timestamp
                        elapsed = (received - camera_first_receive) / 1e9
                        self._stats = replace(
                            self._stats,
                            observed_camera_rate_hz=(
                                (camera_samples - 1) / elapsed if elapsed > 0 else 0.0
                            ),
                            camera_relative_delay_s=elapsed
                            - (record.timestamp - camera_first_source) / 1e9,
                        )
                        if self._failure:
                            break
                        continue
                    if isinstance(record, GpsRecord):
                        self._emit(
                            GpsSample(
                                descriptor.stream_id,
                                session,
                                record.timestamp,
                                received,
                                unix,
                                record.latitude,
                                record.longitude,
                                record.altitude,
                                record.status,
                                record.service,
                                record.covariance,
                                record.covariance_type,
                            )
                        )
                        self._count(decoded_samples=1, decoded_gps_samples=1)
                        gps_samples += 1
                        if gps_first_receive is None:
                            gps_first_receive = received
                        elapsed = (received - gps_first_receive) / 1e9
                        self._stats = replace(
                            self._stats,
                            observed_gps_rate_hz=(
                                (gps_samples - 1) / elapsed if elapsed > 0 else 0.0
                            ),
                        )
                        if self._failure:
                            break
                        continue
                    self._emit(
                        ImuSample(
                            descriptor.stream_id,
                            session,
                            record.timestamp,
                            received,
                            unix,
                            record.acceleration,
                            record.angular_velocity,
                            record.orientation,
                        )
                    )
                    self._count(decoded_samples=1, decoded_imu_samples=1)
                    samples += 1
                    if first_receive is None:
                        first_receive = received
                    elapsed = (received - first_receive) / 1e9
                    self._stats = replace(
                        self._stats,
                        observed_rate_hz=(
                            (samples - 1) / elapsed if elapsed > 0 else 0.0
                        ),
                    )
                    if self._failure:
                        break
        except ProtocolError as exc:
            self._count(protocol_errors=1)
            disconnect_reason = f"ProtocolError: {exc}"
            self._stats = replace(self._stats, last_error=disconnect_reason)
        except (OSError, ConnectionError) as exc:
            self._count(transport_errors=1)
            disconnect_reason = f"{type(exc).__name__}: {exc}"
            self._stats = replace(self._stats, last_error=disconnect_reason)
        except asyncio.CancelledError:
            disconnect_reason = "receiver stopped"
            raise
        except Exception as exc:
            self._fail(TransportError(f"Receiver failed: {exc}"))
        finally:
            if not rejected:
                self._stats = replace(self._stats, connected=False)
                self._emit(SourceDiscontinuity(session, disconnect_reason))
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass

    async def aclose(self, *, drain: bool = False) -> tuple[SensorEvent, ...]:
        """Stop reception; optionally return queued events for final persistence.

        Returned events are removed from the source and retain delivery order.
        Bytes not yet decoded when reception stops are outside this guarantee.
        """
        self._closed = True
        self._wake.set()
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        clients = tuple(self._clients)
        for task in clients:
            task.cancel()
        if clients:
            await asyncio.gather(*clients, return_exceptions=True)
        pending = tuple(self._queue) if drain else ()
        self._queue.clear()
        return pending
