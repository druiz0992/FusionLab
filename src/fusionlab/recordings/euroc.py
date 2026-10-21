"""Import EuRoC cam0 and state ground truth without changing source assets."""

import csv
from hashlib import sha256
import json
from pathlib import Path
import shutil


def rows(path):
    with Path(path).open() as file:
        yield from csv.reader(
            line for line in file if line.strip() and not line.lstrip().startswith("#")
        )


def rigid_transform(values):
    import numpy as np

    matrix = np.asarray(values, dtype=float).reshape(4, 4)
    if (
        not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1])
        or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-6)
    ):
        raise ValueError("Invalid rigid transform")
    return matrix


def import_sequence(source, output):
    import numpy as np
    from scipy.spatial.transform import Rotation

    try:
        import yaml
    except ImportError as exc:
        raise ValueError("Install the benchmark extra to read EuRoC YAML") from exc
    source, output = Path(source).resolve(), Path(output).resolve()
    mav = source / "mav0" if (source / "mav0").is_dir() else source
    camera = mav / "cam0"
    sensor = yaml.safe_load((camera / "sensor.yaml").read_text())
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
    frames = []
    previous = -1
    for timestamp, filename, *_ in rows(camera / "data.csv"):
        stamp = int(timestamp)
        image = (camera / "data" / filename).resolve()
        if (
            stamp <= previous
            or not image.is_relative_to((camera / "data").resolve())
            or not image.is_file()
        ):
            raise ValueError("Invalid image path or non-increasing camera timestamp")
        frames.append((stamp, image))
        previous = stamp
    if not frames:
        raise ValueError("No camera frames")
    # EuRoC state_groundtruth_estimate0 states describe the IMU/body frame.
    truth = []
    previous = -1
    for row in rows(mav / "state_groundtruth_estimate0/data.csv"):
        stamp = int(row[0])
        values = np.asarray(row[1:8], dtype=float)
        if stamp <= previous or not np.isfinite(values).all():
            raise ValueError("Invalid ground-truth state or timestamp")
        q = values[3:7]  # EuRoC stores w,x,y,z.
        if not np.isclose(np.linalg.norm(q), 1, atol=1e-3):
            raise ValueError("Invalid ground-truth quaternion")
        world_from_body = np.eye(4)
        world_from_body[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        world_from_body[:3, 3] = values[:3]
        truth.append(
            dict(source_time_ns=stamp, world_from_body=world_from_body.tolist())
        )
        previous = stamp
    if len(truth) < 2:
        raise ValueError("At least two ground-truth states required")
    output.mkdir(parents=True, exist_ok=False)
    (output / "camera").mkdir()
    with (output / "events.jsonl").open("x") as file:
        for index, (stamp, image) in enumerate(frames):
            relative = f"camera/{stamp}{image.suffix}"
            shutil.copyfile(image, output / relative)
            data = (output / relative).read_bytes()
            event = dict(
                type="CameraSample",
                stream_id="euroc/cam0",
                source_time_ns=stamp,
                received_monotonic_ns=stamp - frames[0][0],
                image_path=relative,
                image_size_bytes=len(data),
                image_sha256=sha256(data).hexdigest(),
            )
            file.write(json.dumps(event) + "\n")
    calibration = dict(
        image_size=[width, height],
        camera_matrix=[[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
        distortion_coefficients=distortion + [0.0],
        distortion_order=["k1", "k2", "p1", "p2", "k3"],
        provenance=str(camera / "sensor.yaml"),
    )
    (output / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
    (output / "groundtruth.json").write_text(
        json.dumps(dict(states=truth, body_from_camera=body_from_camera.tolist()))
        + "\n"
    )
    metadata = dict(
        dataset="EuRoC",
        source=str(source),
        camera="cam0",
        frames=len(frames),
        groundtruth_states=len(truth),
        receive_timing="Synthetic relative source times; not laptop reception",
        source_url="https://projects.asl.ethz.ch/datasets/euroc-mav/",
    )
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
