"""Write a small synthetic EuRoC ASL sequence with exactly known ground truth.

Used by `scripts/output_parity.sh` to exercise import-euroc, tracking, VO and
evaluate-trajectory end to end, because the real EuRoC data are not in the
workspace. It is a refactor-parity fixture, not a benchmark: the scene is a
cloud of bright dots on a gray background, rendered with the same pinhole
radial-tangential model the importer reads.

Layout written under OUTPUT (which must not exist):
  mav0/cam0/sensor.yaml, mav0/cam0/data.csv, mav0/cam0/data/<ns>.png
  mav0/state_groundtruth_estimate0/data.csv   (p, q as w,x,y,z, then zeros)

Frames: the body moves forward along world +X while yawing slowly; the camera
looks along body +X (EuRoC-style rotated mounting with a small lever arm).
Camera at 20 Hz, ground truth at 200 Hz, all timestamps integer nanoseconds.

Usage: python scripts/make_synthetic_euroc.py OUTPUT
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray

EPOCH_NS = 1_400_000_000_000_000_000
CAMERA_PERIOD_NS = 50_000_000  # 20 Hz
STATE_PERIOD_NS = 5_000_000  # 200 Hz
FRAMES = 60
IMAGE_SIZE = (320, 240)  # width, height in pixels
INTRINSICS = (250.0, 250.0, 160.0, 120.0)  # fx, fy, cx, cy
DISTORTION = (-0.05, 0.01, 0.0, 0.0)  # k1, k2, p1, p2
LANDMARKS = 400
SPEED_M_S = 0.5
YAW_RATE_RAD_S = 0.15
# Camera axes in body coordinates: right = -Y, down = -Z, forward = +X.
R_BODY_CAMERA = ((0.0, 0.0, 1.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0))
BODY_CAMERA_LEVER_ARM_M = (0.05, 0.02, -0.01)


def world_from_body(time_s: float) -> NDArray:
    """4×4 body pose: forward motion with a gentle sideways wave and yaw."""
    import numpy as np

    yaw = YAW_RATE_RAD_S * time_s
    pose = np.eye(4)
    pose[:3, :3] = [
        [np.cos(yaw), -np.sin(yaw), 0.0],
        [np.sin(yaw), np.cos(yaw), 0.0],
        [0.0, 0.0, 1.0],
    ]
    pose[:3, 3] = [SPEED_M_S * time_s, 0.1 * np.sin(time_s), 0.02 * time_s]
    return pose


def body_from_camera() -> NDArray:
    """4×4 camera mounting in the body frame (EuRoC's T_BS)."""
    import numpy as np

    transform = np.eye(4)
    transform[:3, :3] = R_BODY_CAMERA
    transform[:3, 3] = BODY_CAMERA_LEVER_ARM_M
    return transform


def render(points: NDArray, world_from_camera: NDArray) -> NDArray:
    """Gray image with one bright Gaussian dot per visible landmark."""
    import cv2
    import numpy as np

    width, height = IMAGE_SIZE
    camera_from_world = np.linalg.inv(world_from_camera)
    in_camera = points @ camera_from_world[:3, :3].T + camera_from_world[:3, 3]
    visible = in_camera[:, 2] > 0.5
    fx, fy, cx, cy = INTRINSICS
    matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    pixels, _ = cv2.projectPoints(
        in_camera[visible], np.zeros(3), np.zeros(3), matrix, np.array(DISTORTION)
    )
    image = np.full((height, width), 60, dtype=np.uint8)
    brightness = np.linspace(150, 255, len(points)).astype(int)[visible]
    for (x, y), level in zip(pixels.reshape(-1, 2), brightness):
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(image, (round(x), round(y)), 2, int(level), -1)
    return cv2.GaussianBlur(image, (5, 5), 1.0)


def quaternion_wxyz(rotation: NDArray) -> list[float]:
    """Unit quaternion in EuRoC's w, x, y, z order."""
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(rotation).as_quat()
    return [w, x, y, z]


def write_sequence(output: Path) -> None:
    """Write the camera, calibration and ground-truth files."""
    import cv2
    import numpy as np
    import yaml

    mav = output / "mav0"
    camera_dir = mav / "cam0"
    (camera_dir / "data").mkdir(parents=True)
    (mav / "state_groundtruth_estimate0").mkdir()
    sensor = {
        "camera_model": "pinhole",
        "distortion_model": "radial-tangential",
        "resolution": list(IMAGE_SIZE),
        "intrinsics": list(INTRINSICS),
        "distortion_coefficients": list(DISTORTION),
        "T_BS": {"data": body_from_camera().ravel().tolist()},
    }
    (camera_dir / "sensor.yaml").write_text(yaml.safe_dump(sensor))
    points = np.random.default_rng(7).uniform([3, -4, -2], [12, 4, 2], (LANDMARKS, 3))
    frames = ["#timestamp [ns],filename\n"]
    for index in range(FRAMES):
        stamp = EPOCH_NS + index * CAMERA_PERIOD_NS
        pose = world_from_body((stamp - EPOCH_NS) / 1e9) @ body_from_camera()
        cv2.imwrite(str(camera_dir / "data" / f"{stamp}.png"), render(points, pose))
        frames.append(f"{stamp},{stamp}.png\n")
    (camera_dir / "data.csv").write_text("".join(frames))
    states = ["#timestamp,p_x,p_y,p_z,q_w,q_x,q_y,q_z,...\n"]
    duration_ns = (FRAMES - 1) * CAMERA_PERIOD_NS
    for offset in range(0, duration_ns + STATE_PERIOD_NS, STATE_PERIOD_NS):
        pose = world_from_body(offset / 1e9)
        values = [*pose[:3, 3], *quaternion_wxyz(pose[:3, :3])] + [0.0] * 9
        states.append(
            f"{EPOCH_NS + offset},"
            + ",".join(repr(float(value)) for value in values)
            + "\n"
        )
    (mav / "state_groundtruth_estimate0" / "data.csv").write_text("".join(states))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path, help="new directory to create")
    output = parser.parse_args().output
    if output.exists():
        parser.error(f"{output} already exists")
    write_sequence(output)


if __name__ == "__main__":
    main()
