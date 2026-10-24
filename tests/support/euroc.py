"""Tiny EuRoC ASL-format sequences for importer and evaluation tests.

The sequence has eight 96×64 noise images at 100 Hz from epoch 1.4e18 ns, a
pinhole camera (f = 80 px, no distortion) mounted 0.2 m along body +X with
identity rotation, and identity-orientation ground truth at the same stamps,
with body position (0.1 i, (0.1 i)², 0) at frame i.
"""

from pathlib import Path

EPOCH_NS = 1_400_000_000_000_000_000
FRAME_PERIOD_NS = 10_000_000
FRAMES = 8
CAMERA_LEVER_ARM_X_M = 0.2


def write_euroc_sequence(root: Path) -> tuple[Path, int]:
    """Write `root/source/mav0`; return (mav0 path, first timestamp in ns).

    Requires OpenCV, NumPy and PyYAML; callers skip via `require(...)` first.
    """
    import cv2
    import numpy as np
    import yaml

    mav = root / "source/mav0"
    camera = mav / "cam0"
    (camera / "data").mkdir(parents=True)
    truth = mav / "state_groundtruth_estimate0"
    truth.mkdir()
    transform = np.eye(4)
    transform[0, 3] = CAMERA_LEVER_ARM_X_M
    sensor = dict(
        camera_model="pinhole",
        distortion_model="radial-tangential",
        resolution=[96, 64],
        intrinsics=[80, 80, 48, 32],
        distortion_coefficients=[0, 0, 0, 0],
        T_BS=dict(data=transform.ravel().tolist()),
    )
    (camera / "sensor.yaml").write_text(yaml.safe_dump(sensor))
    frames = []
    states = []
    for i in range(FRAMES):
        stamp = EPOCH_NS + i * FRAME_PERIOD_NS
        filename = f"{stamp}.png"
        image = np.random.default_rng(4).integers(0, 255, (64, 96), dtype=np.uint8)
        cv2.imwrite(str(camera / "data" / filename), image)
        frames.append(f"{stamp},{filename}\n")
        # p_x, p_y, p_z, q_w, q_x, q_y, q_z, then unused velocity/bias columns.
        states.append(f"{stamp},{i*0.1},{(i*0.1)**2},0,1,0,0,0,0,0,0\n")
    (camera / "data.csv").write_text("#timestamp,filename\n" + "".join(frames))
    (truth / "data.csv").write_text("#timestamp,p,q\n" + "".join(states))
    return mav, EPOCH_NS
