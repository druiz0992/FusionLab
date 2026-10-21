"""Local TCP benchmark using a saved JPEG; no phone or external network needed.

PYTHONPATH=src python3 benchmarks/camera_receiver.py CAPTURE --frames 180
"""

import argparse
import asyncio
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from sensor_stream import CameraSample, ReceiverConfig, SensorStreamSource
from sensor_stream.cli import _write_recording
from sensor_stream.protocol.framing import HEADER
from sensor_stream.protocol.proto.imu_pb2 import Envelope
from sensor_stream.recording import SessionRecorder


async def run(
    image: bytes, width: int, height: int, frames: int, chunk_bytes: int, record: bool
) -> dict:
    config = ReceiverConfig(bind_host="127.0.0.1", bind_port=0)
    # Bound producer lead, so this measures sustained delivery rather than an
    # arbitrarily large burst overflowing the receiver's finite queue.
    slots = asyncio.Semaphore(16)
    with tempfile.TemporaryDirectory(prefix="fusionlab-camera-benchmark-") as directory:
        recorder = SessionRecorder(Path(directory), config) if record else None
        if recorder:
            recorder.__enter__()
        try:
            async with SensorStreamSource(config) as source:

                async def consume():
                    count = 0
                    async for event in source.events():
                        if recorder:
                            await _write_recording(recorder, event)
                        if isinstance(event, CameraSample):
                            count += 1
                            slots.release()
                            if count == frames:
                                return

                consumer = asyncio.create_task(consume())
                _, writer = await asyncio.open_connection("127.0.0.1", source.port)
                started = perf_counter()
                try:
                    async with asyncio.timeout(60):
                        for i in range(frames):
                            await slots.acquire()
                            envelope = Envelope()
                            camera = envelope.camera
                            camera.timestamp = 1_000_000_000 + i * 33_333_333
                            camera.width = width
                            camera.height = height
                            camera.encoding = "jpeg"
                            camera.frame_id = "benchmark_camera"
                            camera.image_data = image
                            payload = envelope.SerializeToString()
                            for index, offset in enumerate(
                                range(0, len(payload), chunk_bytes)
                            ):
                                chunk = payload[offset : offset + chunk_bytes]
                                writer.write(
                                    HEADER.pack(
                                        i,
                                        index,
                                        int(offset + chunk_bytes >= len(payload)),
                                        len(chunk),
                                    )
                                    + chunk
                                )
                            await writer.drain()
                        await consumer
                    elapsed = perf_counter() - started
                    stats = source.stats()
                finally:
                    writer.close()
                    await writer.wait_closed()
                    consumer.cancel()
                    await asyncio.gather(consumer, return_exceptions=True)
                if recorder:
                    recorder.set_stats(source.stats())
            return {
                "recording": record,
                "frames": frames,
                "frame_bytes": len(image),
                "chunk_bytes": chunk_bytes,
                "elapsed_s": elapsed,
                "frames_per_s": frames / elapsed,
                "payload_MB_per_s": len(image) * frames / elapsed / 1e6,
                "required_MB_per_s_at_30fps": len(image) * 30 / 1e6,
                "stats": asdict(stats),
            }
        finally:
            if recorder:
                recorder.__exit__(None, None, None)


async def main(args):
    with (args.capture / "events.jsonl").open() as file:
        frames = [
            event
            for line in file
            if (event := json.loads(line)).get("type") == "CameraSample"
        ]
    if not frames:
        raise ValueError("Capture has no camera frames")
    event = sorted(frames, key=lambda event: event["image_size_bytes"])[
        len(frames) // 2
    ]
    image = (args.capture / event["image_path"]).read_bytes()
    for chunk in (16 * 1024, 64 * 1024):
        for record in (False, True):
            result = await run(
                image, event["width"], event["height"], args.frames, chunk, record
            )
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--frames", type=int, default=180)
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error("--frames must be positive")
    asyncio.run(main(args))
