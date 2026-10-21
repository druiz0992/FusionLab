"""Decoded event recording, kept outside the device receiver."""

import asyncio
import json
from hashlib import sha256
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Self, TextIO
from uuid import uuid4

from .config import ReceiverConfig
from .models import ImuSample, GpsSample, CameraSample, DepthSample, SensorEvent
from .stats import SourceStats


class SessionRecorder:
    def __init__(self, root: Path, config: ReceiverConfig) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.path = root / f"{stamp}-{uuid4().hex[:8]}"
        self._metadata: dict[str, object] = {
            "format_version": 1,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "recording",
            "receiver_config": asdict(config),
            "measurement_semantics": "wire values; units, axes and gravity unverified",
        }
        self._events: TextIO | None = None
        self._stats: SourceStats | None = None
        self._count = 0
        self._samples = 0
        self._imu_samples = 0
        self._gps_samples = 0
        self._camera_samples = 0
        self._depth_samples = 0

    def _save_metadata(self) -> None:
        # Write then replace so readers see a complete metadata document.
        temporary = self.path / "metadata.json.tmp"
        temporary.write_text(
            json.dumps(self._metadata, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(self.path / "metadata.json")

    def __enter__(self) -> Self:
        self.path.mkdir(parents=True, exist_ok=False)
        self._save_metadata()
        self._events = (self.path / "events.jsonl").open(
            "x", encoding="utf-8", buffering=1
        )
        return self

    def write(self, event: SensorEvent) -> None:
        if self._events is None:
            raise RuntimeError("Recorder is not open")
        # Store event metadata in JSONL, with camera/depth bytes in separate files.
        # The saved size and hash allow offline readers to detect missing/corrupt assets.
        data = asdict(event)
        if isinstance(event, CameraSample):
            image = data.pop("image_data")
            extension = (
                "jpg"
                if event.encoding.lower() in ("jpeg", "jpg", "image/jpeg")
                else "bin"
            )
            relative = Path("camera") / f"{self._camera_samples:08d}.{extension}"
            target = self.path / relative
            target.parent.mkdir(exist_ok=True)
            with target.open("xb") as file:
                file.write(image)
            data.update(
                image_path=relative.as_posix(),
                image_size_bytes=len(image),
                image_sha256=sha256(image).hexdigest(),
            )
        if isinstance(event, DepthSample):
            payload = data.pop("depth_data")
            relative = Path("depth") / f"{self._depth_samples:08d}.bin"
            target = self.path / relative
            target.parent.mkdir(exist_ok=True)
            with target.open("xb") as file:
                file.write(payload)
            data.update(
                image_path=relative.as_posix(),
                image_size_bytes=len(payload),
                image_sha256=sha256(payload).hexdigest(),
            )
        # Write the metadata row after its asset exists so it references saved bytes.
        self._events.write(
            json.dumps({"type": type(event).__name__, **data}, allow_nan=False) + "\n"
        )
        self._count += 1
        self._samples += isinstance(
            event, (ImuSample, GpsSample, CameraSample, DepthSample)
        )
        self._imu_samples += isinstance(event, ImuSample)
        self._gps_samples += isinstance(event, GpsSample)
        self._camera_samples += isinstance(event, CameraSample)
        self._depth_samples += isinstance(event, DepthSample)

    def set_stats(self, stats: SourceStats) -> None:
        self._stats = stats

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._events is not None:
            self._events.close()
            self._events = None
        # Context exit records whether capture ended normally, by interruption,
        # or through an error, together with final counts and receiver statistics.
        status = "completed"
        if exc_type is not None:
            status = (
                "interrupted"
                if issubclass(exc_type, (KeyboardInterrupt, asyncio.CancelledError))
                else "failed"
            )
            self._metadata["error"] = f"{exc_type.__name__}: {exc}"
        self._metadata.update(
            ended_at=datetime.now(timezone.utc).isoformat(),
            status=status,
            events_written=self._count,
            samples_written=self._samples,
            imu_samples_written=self._imu_samples,
            gps_samples_written=self._gps_samples,
            camera_samples_written=self._camera_samples,
            depth_samples_written=self._depth_samples,
        )
        if self._stats is not None:
            self._metadata["final_stats"] = asdict(self._stats)
        self._save_metadata()
