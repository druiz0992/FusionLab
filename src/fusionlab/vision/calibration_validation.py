"""Score fixed RGB intrinsics on a separate checkerboard capture without refitting.

Calibration fitting (`calibration.py`) chooses K and distortion to explain its own
views, so its residuals are optimistic. This module asks a different question:
do the *existing* intrinsics still predict checkerboard corners in new images?

For each verification image the board pose is solved with K/distortion held fixed,
using four fifths of the detected corners. The remaining fifth was not used for
that pose, so their reprojection error measures how well the fixed model predicts
unseen image points. This is a consistency check: it does not provide independent
board-pose ground truth, and a target defect shared with the fitting capture can
hide a systematic error.

Units: pixels refer to the original stored image; board geometry is in metres.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

# Only the five-coefficient OpenCV radial-tangential model is supported.
SUPPORTED_DISTORTION_ORDER = ["k1", "k2", "p1", "p2", "k3"]

# Every fifth corner (indices 4, 9, ...) is held out of the board pose fit.
HELDOUT_EVERY_NTH_CORNER = 5

LIMITS = (
    "Board poses fitted with fixed K/distortion using 4/5 corners; other corners "
    "scored. Correlated observations, same setup may share target defects. Not "
    "independent pose ground truth or a verified camera-mode match."
)


@dataclass(frozen=True)
class FixedCalibration:
    """The parts of a `calibration.json` needed to score it; never modified."""

    camera_matrix: list[list[float]]  # K in original image pixels
    distortion_coefficients: list[float]  # k1, k2, p1, p2, k3
    inner_corners: tuple[int, int]  # (columns, rows) of inner corners
    measured_square_mm: float  # user-measured printed square size
    image_size: tuple[int, int]  # (width, height) in pixels

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> FixedCalibration:
        """Build from a parsed calibration file, rejecting unsupported models."""
        if data["distortion_order"] != SUPPORTED_DISTORTION_ORDER:
            raise ValueError("Unsupported model")
        columns, rows = data["inner_corners"]
        width, height = data["image_size"]
        return cls(
            camera_matrix=data["camera_matrix"],
            distortion_coefficients=data["distortion_coefficients"],
            inner_corners=(int(columns), int(rows)),
            measured_square_mm=float(data["measured_square_mm"]),
            image_size=(int(width), int(height)),
        )


def board_points_m(calibration: FixedCalibration) -> np.ndarray:
    """Return board corner coordinates in metres on the board's z=0 plane.

    The order matches OpenCV's detector: row by row, columns varying fastest.
    """
    import numpy as np

    columns, rows = calibration.inner_corners
    square_m = calibration.measured_square_mm / 1000
    points = np.zeros((columns * rows, 3), np.float32)
    points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * square_m
    return points


def score_corners(corners: Any, calibration: FixedCalibration) -> dict[str, float]:
    """Fit board pose on 4/5 of the corners with fixed intrinsics; score the rest.

    `corners` are detected pixel positions, shaped (N, 2) or (N, 1, 2).
    Returns RMS over all corners, RMS over held-out corners and the worst corner.
    """
    import cv2 as cv
    import numpy as np

    board = board_points_m(calibration)
    pixels = np.asarray(corners, np.float32).reshape(-1, 2)
    if len(pixels) != len(board):
        raise ValueError("Unexpected corner count")
    camera_matrix = np.asarray(calibration.camera_matrix, float)
    distortion = np.asarray(calibration.distortion_coefficients, float)
    heldout = np.arange(len(board)) % HELDOUT_EVERY_NTH_CORNER == (
        HELDOUT_EVERY_NTH_CORNER - 1
    )
    solved, rotation, translation = cv.solvePnP(
        board[~heldout], pixels[~heldout], camera_matrix, distortion
    )
    if not solved:
        raise ValueError("Board pose estimation failed")
    projected = cv.projectPoints(
        board, rotation, translation, camera_matrix, distortion
    )[0].reshape(-1, 2)
    error_px = np.linalg.norm(projected - pixels, axis=1)
    return {
        "all_corner_rms_px": float(np.sqrt(np.mean(error_px**2))),
        "heldout_corner_rms_px": float(np.sqrt(np.mean(error_px[heldout] ** 2))),
        "max_corner_error_px": float(error_px.max()),
    }


def read_camera_events(capture: Path) -> dict[str, dict[str, Any]]:
    """Map each recorded image path to its CameraSample event.

    Mixing camera streams could mix camera modes, so exactly one is required.
    """
    events: dict[str, dict[str, Any]] = {}
    for line in (capture / "events.jsonl").read_text().splitlines():
        event = json.loads(line)
        if event["type"] == "CameraSample":
            events[event["image_path"]] = event
    if len({event["stream_id"] for event in events.values()}) != 1:
        raise ValueError("Expected one camera stream")
    return events


def select_images(
    events: dict[str, dict[str, Any]],
    image_list: Path | None,
    stride: int,
    max_views: int,
) -> list[str]:
    """Choose images: an explicit `detections.json` list, else every `stride`-th."""
    if image_list is not None:
        images = json.loads(image_list.read_text())["accepted_images"]
    else:
        images = list(events)[::stride]
    return images[:max_views]


def read_verified_image(
    capture: Path, relative: str, events: dict[str, dict[str, Any]]
) -> bytes:
    """Read an image after checking it belongs to the capture and is unmodified."""
    if relative not in events:
        raise ValueError("Image absent from capture events")
    path = (capture / relative).resolve()
    if not path.is_relative_to(capture.resolve()):
        raise ValueError("Image path escapes capture")
    data = path.read_bytes()
    event = events[relative]
    if (
        len(data) != event["image_size_bytes"]
        or sha256(data).hexdigest() != event["image_sha256"]
    ):
        raise ValueError("Image integrity failure")
    return data


def detect_board(data: bytes, calibration: FixedCalibration) -> Any | None:
    """Decode an image and detect the calibration's board; None if not found.

    The image must keep its original stored dimensions: K is only valid there.
    """
    import cv2 as cv
    import numpy as np

    image = cv.imdecode(np.frombuffer(data, np.uint8), cv.IMREAD_GRAYSCALE)
    if image is None or tuple(image.shape[::-1]) != calibration.image_size:
        raise ValueError("Image/calibration size mismatch")
    found, corners = cv.findChessboardCornersSB(
        image, calibration.inner_corners, flags=cv.CALIB_CB_NORMALIZE_IMAGE
    )
    return corners if found else None


def validate_calibration(
    capture: Path,
    calibration_path: Path,
    output: Path,
    *,
    image_list: Path | None = None,
    stride: int = 60,
    max_views: int = 40,
) -> dict[str, Any]:
    """Score a fixed calibration on a verification capture; write `summary.json`.

    `output` must be a new directory. Neither the images nor the calibration
    file are modified.
    """
    import numpy as np

    capture = Path(capture)
    calibration_path = Path(calibration_path)
    output = Path(output)
    if stride < 1 or max_views < 1:
        raise ValueError("Positive stride and max views required")
    calibration = FixedCalibration.from_json(json.loads(calibration_path.read_text()))
    events = read_camera_events(capture)
    scored: list[dict[str, Any]] = []
    rejected: list[str] = []
    for relative in select_images(events, image_list, stride, max_views):
        corners = detect_board(
            read_verified_image(capture, relative, events), calibration
        )
        if corners is None:
            rejected.append(relative)
            continue
        scored.append({"image": relative, **score_corners(corners, calibration)})
    heldout_rms = [view["heldout_corner_rms_px"] for view in scored]
    result = {
        "capture": str(capture.resolve()),
        "calibration": str(calibration_path.resolve()),
        "calibration_sha256": sha256(calibration_path.read_bytes()).hexdigest(),
        "intrinsics_refitted": False,
        "images": scored,
        "rejected_images": rejected,
        "detected_views": len(scored),
        "median_heldout_corner_rms_px": (
            float(np.median(heldout_rms)) if heldout_rms else None
        ),
        "max_view_heldout_corner_rms_px": max(heldout_rms, default=None),
        "limits": LIMITS,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
