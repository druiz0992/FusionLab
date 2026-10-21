"""Evaluate fixed intrinsics on separate checkerboard images; never refit K/distortion."""

import argparse
from hashlib import sha256
import json
from pathlib import Path


def score_corners(corners, calibration):
    import cv2 as cv
    import numpy as np

    columns, rows = calibration["inner_corners"]
    obj = np.zeros((columns * rows, 3), np.float32)
    obj[:, :2] = (
        np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
        * calibration["measured_square_mm"]
        / 1000
    )
    points = np.asarray(corners, np.float32).reshape(-1, 2)
    if len(points) != len(obj):
        raise ValueError("Unexpected corner count")
    k = np.asarray(calibration["camera_matrix"], float)
    dist = np.asarray(calibration["distortion_coefficients"], float)
    # Fit board pose on four fifths of corners; score the others separately.
    heldout = np.arange(len(obj)) % 5 == 4
    ok, r, t = cv.solvePnP(obj[~heldout], points[~heldout], k, dist)
    if not ok:
        raise ValueError("Board pose estimation failed")
    projected = cv.projectPoints(obj, r, t, k, dist)[0].reshape(-1, 2)
    residual = np.linalg.norm(projected - points, axis=1)
    return dict(
        all_corner_rms_px=float(np.sqrt(np.mean(residual**2))),
        heldout_corner_rms_px=float(np.sqrt(np.mean(residual[heldout] ** 2))),
        max_corner_error_px=float(residual.max()),
    )


def validate(
    capture, calibration_path, output, *, image_list=None, stride=60, max_views=40
):
    import cv2 as cv
    import numpy as np

    capture, calibration_path, output = map(Path, (capture, calibration_path, output))
    if stride < 1 or max_views < 1:
        raise ValueError("Positive stride and max views required")
    calibration = json.loads(calibration_path.read_text())
    if calibration["distortion_order"] != ["k1", "k2", "p1", "p2", "k3"]:
        raise ValueError("Unsupported model")
    events = {
        e["image_path"]: e
        for e in (
            json.loads(l) for l in (capture / "events.jsonl").read_text().splitlines()
        )
        if e["type"] == "CameraSample"
    }
    if len({e["stream_id"] for e in events.values()}) != 1:
        raise ValueError("Expected one camera stream")
    images = (
        json.loads(Path(image_list).read_text())["accepted_images"]
        if image_list
        else list(events)[::stride]
    )
    scored = []
    rejected = []
    for relative in images[:max_views]:
        if relative not in events:
            raise ValueError("Image absent from capture events")
        path = (capture / relative).resolve()
        if not path.is_relative_to(capture.resolve()):
            raise ValueError("Image path escapes capture")
        data = path.read_bytes()
        e = events[relative]
        if (
            len(data) != e["image_size_bytes"]
            or sha256(data).hexdigest() != e["image_sha256"]
        ):
            raise ValueError("Image integrity failure")
        image = cv.imdecode(np.frombuffer(data, np.uint8), cv.IMREAD_GRAYSCALE)
        if image is None or list(image.shape[::-1]) != calibration["image_size"]:
            raise ValueError("Image/calibration size mismatch")
        ok, corners = cv.findChessboardCornersSB(
            image,
            tuple(calibration["inner_corners"]),
            flags=cv.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not ok:
            rejected.append(relative)
            continue
        scored.append(dict(image=relative, **score_corners(corners, calibration)))
    result = dict(
        capture=str(capture.resolve()),
        calibration=str(calibration_path.resolve()),
        calibration_sha256=sha256(calibration_path.read_bytes()).hexdigest(),
        intrinsics_refitted=False,
        images=scored,
        rejected_images=rejected,
        detected_views=len(scored),
        median_heldout_corner_rms_px=(
            float(np.median([r["heldout_corner_rms_px"] for r in scored]))
            if scored
            else None
        ),
        max_view_heldout_corner_rms_px=max(
            (r["heldout_corner_rms_px"] for r in scored), default=None
        ),
        limits="Board poses fitted with fixed K/distortion using 4/5 corners; other corners scored. Correlated observations, same setup may share target defects. Not independent pose ground truth or a verified camera-mode match.",
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--image-list", type=Path)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--max-views", type=int, default=40)
    a = parser.parse_args()
    result = validate(
        a.capture,
        a.calibration,
        a.output_dir,
        image_list=a.image_list,
        stride=a.stride,
        max_views=a.max_views,
    )
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("images", "rejected_images")},
            indent=2,
        )
    )
