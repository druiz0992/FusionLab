"""Import EuRoC cam0 and state ground truth without changing source assets.

Reads an extracted ASL-format sequence (`mav0/` with `cam0/` and
`state_groundtruth_estimate0/`) and writes a FusionLab capture directory:
`events.jsonl` camera events, byte-identical `camera/` images, a
`calibration.json` in FusionLab's distortion order, `groundtruth.json` and
`metadata.json`. Every input is validated before the output is created, so a
bad sequence never leaves a partial import.

Frames: EuRoC `T_BS` is the camera (sensor) pose in the body/IMU frame,
T_body_camera; ground-truth states are T_world_body. Quaternions are stored
w, x, y, z in the CSV. Timestamps are integer nanoseconds.
"""

from __future__ import annotations

import csv
import json
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from fusionlab._typing import Array

CAMERA_STREAM_ID = "euroc/cam0"
SOURCE_URL = "https://projects.asl.ethz.ch/datasets/euroc-mav/"
RECEIVE_TIMING = "Synthetic relative source times; not laptop reception"
# FusionLab's OpenCV coefficient order; EuRoC radial-tangential lacks k3.
DISTORTION_ORDER = ["k1", "k2", "p1", "p2", "k3"]
# Ground-truth quaternions must be unit length to this tolerance.
QUATERNION_NORM_TOLERANCE = 1e-3
# Rotation blocks must be orthonormal with det +1 to this tolerance.
RIGID_TOLERANCE = 1e-6


class CameraFrame(NamedTuple):
    """One cam0 image and its source timestamp."""

    stamp_ns: int
    image: Path  # resolved path inside `cam0/data`


@dataclass(frozen=True)
class CameraModel:
    """cam0 pinhole intrinsics, radial-tangential distortion and mounting."""

    width: int  # image size, pixels
    height: int
    fx: float  # focal lengths and principal point, pixels
    fy: float
    cx: float
    cy: float
    distortion: list[float]  # k1, k2, p1, p2
    body_from_camera: Array  # 4×4 T_body_camera

    def calibration_json(self, provenance: Path) -> dict[str, Any]:
        """FusionLab `calibration.json`, appending a zero k3 coefficient."""
        return {
            "image_size": [self.width, self.height],
            "camera_matrix": [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]],
            "distortion_coefficients": self.distortion + [0.0],
            "distortion_order": DISTORTION_ORDER,
            "provenance": str(provenance),
        }


def csv_rows(path: Path | str) -> Iterator[list[str]]:
    """CSV rows of an EuRoC file, skipping blank lines and `#` comments."""
    with Path(path).open() as file:
        yield from csv.reader(
            line for line in file if line.strip() and not line.lstrip().startswith("#")
        )


def rigid_transform(values: Any) -> Array:
    """A validated 4×4 rigid transform from 16 values (row-major) or a matrix."""
    import numpy as np

    matrix = np.asarray(values, dtype=float).reshape(4, 4)
    rotation = matrix[:3, :3]
    if (
        not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1])
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=RIGID_TOLERANCE)
        or not np.isclose(np.linalg.det(rotation), 1, atol=RIGID_TOLERANCE)
    ):
        raise ValueError("Invalid rigid transform")
    return matrix


def import_sequence(source: Path | str, output: Path | str) -> dict[str, Any]:
    """Import one EuRoC sequence (or its `mav0/`) into the new directory `output`.

    Only cam0 and ground truth are read; IMU and cam1 are ignored. Returns the
    written metadata.
    """
    yaml = _load_yaml()
    source, output = Path(source).resolve(), Path(output).resolve()
    mav = source / "mav0" if (source / "mav0").is_dir() else source
    camera_dir = mav / "cam0"
    sensor_path = camera_dir / "sensor.yaml"
    camera = read_camera_model(yaml.safe_load(sensor_path.read_text()))
    frames = read_camera_frames(camera_dir)
    truth = read_ground_truth(mav / "state_groundtruth_estimate0" / "data.csv")
    output.mkdir(parents=True, exist_ok=False)
    _write_camera_events(output, frames)
    _write_json(output / "calibration.json", camera.calibration_json(sensor_path))
    groundtruth = {
        "states": truth,
        "body_from_camera": camera.body_from_camera.tolist(),
    }
    (output / "groundtruth.json").write_text(json.dumps(groundtruth) + "\n")
    metadata = {
        "dataset": "EuRoC",
        "source": str(source),
        "camera": "cam0",
        "frames": len(frames),
        "groundtruth_states": len(truth),
        "receive_timing": RECEIVE_TIMING,
        "source_url": SOURCE_URL,
    }
    _write_json(output / "metadata.json", metadata)
    return metadata


def _load_yaml() -> Any:
    """The PyYAML module, with an install hint if the benchmark extra is missing."""
    try:
        import yaml
    except ImportError as exc:
        raise ValueError("Install the benchmark extra to read EuRoC YAML") from exc
    return yaml


def read_camera_model(sensor: dict[str, Any]) -> CameraModel:
    """Validate a parsed cam0 `sensor.yaml`; only pinhole radial-tangential."""
    import numpy as np

    if (
        sensor["camera_model"] != "pinhole"
        or sensor["distortion_model"] != "radial-tangential"
    ):
        raise ValueError("Only pinhole radial-tangential EuRoC cameras are supported")
    width, height = map(int, sensor["resolution"])
    fx, fy, cx, cy = map(float, sensor["intrinsics"])
    distortion = list(map(float, sensor["distortion_coefficients"]))
    if (
        len(distortion) != 4
        or min(width, height, fx, fy) <= 0
        or not np.isfinite([fx, fy, cx, cy, *distortion]).all()
    ):
        raise ValueError("Invalid camera calibration")
    body_from_camera = rigid_transform(sensor["T_BS"]["data"])
    return CameraModel(width, height, fx, fy, cx, cy, distortion, body_from_camera)


def read_camera_frames(camera_dir: Path) -> list[CameraFrame]:
    """cam0 frames from `data.csv`, strictly increasing, images inside `data/`.

    Paths are resolved and must stay inside `data/`, so a crafted CSV cannot
    make the importer copy files from elsewhere.
    """
    data_dir = (camera_dir / "data").resolve()
    frames: list[CameraFrame] = []
    previous = -1
    for timestamp, filename, *_ in csv_rows(camera_dir / "data.csv"):
        stamp = int(timestamp)
        image = (camera_dir / "data" / filename).resolve()
        if (
            stamp <= previous
            or not image.is_relative_to(data_dir)
            or not image.is_file()
        ):
            raise ValueError("Invalid image path or non-increasing camera timestamp")
        frames.append(CameraFrame(stamp, image))
        previous = stamp
    if not frames:
        raise ValueError("No camera frames")
    return frames


def read_ground_truth(path: Path) -> list[dict[str, Any]]:
    """T_world_body states from `state_groundtruth_estimate0/data.csv`.

    Each row starts with timestamp, position (m) and quaternion w, x, y, z;
    velocities and biases that follow are not used. Returns JSON-ready
    {"source_time_ns", "world_from_body"} records.
    """
    import numpy as np

    truth = []
    previous = -1
    for row in csv_rows(path):
        stamp = int(row[0])
        values = np.asarray(row[1:8], dtype=float)
        if stamp <= previous or not np.isfinite(values).all():
            raise ValueError("Invalid ground-truth state or timestamp")
        truth.append(
            {
                "source_time_ns": stamp,
                "world_from_body": _pose_from_state(values).tolist(),
            }
        )
        previous = stamp
    if len(truth) < 2:
        raise ValueError("At least two ground-truth states required")
    return truth


def _pose_from_state(values: Array) -> Array:
    """4×4 T_world_body from [p_x, p_y, p_z, q_w, q_x, q_y, q_z]."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    quaternion_wxyz = values[3:7]
    if not np.isclose(
        np.linalg.norm(quaternion_wxyz), 1, atol=QUATERNION_NORM_TOLERANCE
    ):
        raise ValueError("Invalid ground-truth quaternion")
    pose = np.eye(4)
    # SciPy expects x, y, z, w.
    pose[:3, :3] = Rotation.from_quat(quaternion_wxyz[[1, 2, 3, 0]]).as_matrix()
    pose[:3, 3] = values[:3]
    return pose


def _write_camera_events(output: Path, frames: list[CameraFrame]) -> None:
    """Copy each image unchanged and write its `CameraSample` event.

    Receive times are synthetic (source time since the first frame); the hash
    and size are of the copied bytes, which tracking verifies before decoding.
    """
    (output / "camera").mkdir()
    origin = frames[0].stamp_ns
    with (output / "events.jsonl").open("x") as file:
        for frame in frames:
            relative = f"camera/{frame.stamp_ns}{frame.image.suffix}"
            shutil.copyfile(frame.image, output / relative)
            data = (output / relative).read_bytes()
            event = {
                "type": "CameraSample",
                "stream_id": CAMERA_STREAM_ID,
                "source_time_ns": frame.stamp_ns,
                "received_monotonic_ns": frame.stamp_ns - origin,
                "image_path": relative,
                "image_size_bytes": len(data),
                "image_sha256": sha256(data).hexdigest(),
            }
            file.write(json.dumps(event) + "\n")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Indented JSON with a trailing newline."""
    path.write_text(json.dumps(data, indent=2) + "\n")
