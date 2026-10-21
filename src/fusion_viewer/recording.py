"""Portable viewer log v1: JSONL declarations and content-addressed assets."""

import json
from hashlib import sha256
from dataclasses import asdict
from pathlib import Path
from .models import (
    RunSpec,
    FrameSpec,
    SampleTime,
    ScalarSample,
    OrientationSample,
    ImageSample,
    DepthSample,
    SegmentBoundary,
    Map2D,
    Pose2D,
    Trajectory2D,
    Particles2D,
)


class LogSink:
    """Write typed events to JSONL and deduplicate binary assets by SHA-256."""

    def __init__(self, path: Path):
        path.mkdir(parents=True, exist_ok=False)
        self.path = path
        # Sequence numbers preserve declaration/message order and detect missing rows.
        self.sequence = 0
        self.file = (path / "events.jsonl").open("x", encoding="utf-8")
        (path / "manifest.json").write_text(
            json.dumps({"schema_version": 1, "status": "writing"})
        )

    def write(self, type_name, run, payload):
        self.file.write(
            json.dumps(
                {
                    "schema_version": 1,
                    "sequence": self.sequence,
                    "type": type_name,
                    "run_id": run,
                    "payload": payload,
                },
                allow_nan=False,
            )
            + "\n"
        )
        self.sequence += 1

    def register_run(self, run):
        self.write("RunSpec", run.id, asdict(run))

    def register_frame(self, run, frame):
        self.write("FrameSpec", run, asdict(frame))

    def publish(self, run, message):
        data = asdict(message)
        if isinstance(message, (ImageSample, DepthSample)):
            raw = data.pop("data")
            # Keep large payloads out of JSON and reuse identical asset bytes.
            digest = sha256(raw).hexdigest()
            relative = f"assets/{digest}.bin"
            target = self.path / relative
            target.parent.mkdir(exist_ok=True)
            if not target.exists():
                target.write_bytes(raw)
            data["asset"] = {"path": relative, "sha256": digest, "size": len(raw)}
        self.write(type(message).__name__, run, data)

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()
        (self.path / "manifest.json").write_text(
            json.dumps(
                {"schema_version": 1, "status": "complete", "records": self.sequence}
            )
        )


def read_log(path: Path):
    """Yield typed records in sequence, verifying asset containment and hashes."""
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported viewer log schema")
    classes = {
        c.__name__: c
        for c in (
            ScalarSample,
            OrientationSample,
            ImageSample,
            DepthSample,
            SegmentBoundary,
            Map2D,
            Pose2D,
            Trajectory2D,
            Particles2D,
        )
    }
    with (path / "events.jsonl").open() as file:
        for index, line in enumerate(file):
            record = json.loads(line)
            if record.get("schema_version") != 1 or record.get("sequence") != index:
                raise ValueError("Invalid log version/sequence")
            name, run, data = record["type"], record["run_id"], record["payload"]
            if name == "RunSpec":
                yield run, RunSpec(**data)
                continue
            if name == "FrameSpec":
                yield run, FrameSpec(**data)
                continue
            if name not in classes:
                raise ValueError("Unsupported viewer log message")
            if "asset" in data:
                asset = data.pop("asset")
                target = (path / asset["path"]).resolve()
                if not target.is_relative_to(path.resolve()):
                    raise ValueError("Asset escapes recording")
                raw = target.read_bytes()
                if (
                    len(raw) != asset["size"]
                    or sha256(raw).hexdigest() != asset["sha256"]
                ):
                    raise ValueError("Asset integrity failure")
                data["data"] = raw
            # JSON loses dataclass and tuple types. Rebuild the public message shapes
            # before yielding records to the client for validation and playback.
            data["time"] = SampleTime(**data["time"])
            if name == "OrientationSample":
                data["quaternion_xyzw"] = tuple(data["quaternion_xyzw"])
            if name == "Map2D":
                data["walls"] = tuple(
                    tuple(tuple(p) for p in wall) for wall in data["walls"]
                )
            if name == "Pose2D":
                data["xy"] = tuple(data["xy"])
            if name == "Trajectory2D":
                data["points"] = tuple(tuple(p) for p in data["points"])
            if name == "Particles2D":
                data["poses"] = tuple(tuple(p) for p in data["poses"])
                data["weights"] = tuple(data["weights"])
            yield run, classes[name](**data)
