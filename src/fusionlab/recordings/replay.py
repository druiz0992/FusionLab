"""Replay saved measurements without rewriting their recorded timestamps."""

import asyncio
from pathlib import Path
from typing import AsyncIterator, Iterator
from math import isfinite

from sensor_stream import ImuSample
from sensor_stream.inspection import load_capture


def read_samples(capture: Path) -> Iterator[ImuSample]:
    _, events = load_capture(capture)
    for event in events:
        if event["type"] == "ImuSample":
            data = {k: v for k, v in event.items() if k != "type"}
            # JSON arrays become lists; restore the tuple-valued measurement fields.
            for key in ("acceleration", "angular_velocity", "orientation"):
                if data.get(key) is not None:
                    data[key] = tuple(data[key])
            yield ImuSample(**data)


async def replay(capture: Path, *, speed: float = 0) -> AsyncIterator[ImuSample]:
    """Speed 0 runs immediately; 1 follows recorded laptop receive timing.

    Receive timing preserves pauses and reconnect delays even if the phone clock
    resets. Input is validated/loaded once; this initial version holds it in memory.
    """
    if not isfinite(speed) or speed < 0:
        raise ValueError("Replay speed must be finite and nonnegative")
    loop = asyncio.get_running_loop()
    start = loop.time()
    origin: int | None = None
    for sample in read_samples(capture):
        if origin is None:
            origin = sample.received_monotonic_ns
        if speed:
            # Schedule against the replay start, rather than sleeping each interval,
            # so time spent processing earlier samples does not accumulate as drift.
            target = start + (sample.received_monotonic_ns - origin) / 1e9 / speed
            await asyncio.sleep(max(0, target - loop.time()))
        yield sample
