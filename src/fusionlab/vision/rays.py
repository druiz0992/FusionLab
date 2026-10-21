"""Undistorted camera rays for feature tracks.

A ray is the normalized image coordinate (x, y) of a track, i.e. the direction
(x, y, 1) in the camera frame (+X right, +Y down, +Z forward) with lens
distortion removed. The calibration is expressed in original stored pixels while
tracks are in resized tracking pixels, so K is scaled per axis first.
"""

from typing import Any

from fusionlab.vision.tracks import TrackFrame

SUPPORTED_DISTORTION_ORDER = ["k1", "k2", "p1", "p2", "k3"]


def tracking_camera_matrix(row: TrackFrame, calibration: dict[str, Any]):
    """Calibration K scaled from original pixels to this row's tracking pixels."""
    import numpy as np

    if row.original_size != tuple(calibration["image_size"]):
        raise ValueError("Calibration and original image dimensions differ")
    if calibration.get("distortion_order") != SUPPORTED_DISTORTION_ORDER:
        raise ValueError("Unsupported distortion model")
    scale = np.asarray(row.tracking_size) / np.asarray(row.original_size)
    if not np.allclose(scale, row.pixel_scale_xy):
        raise ValueError("Inconsistent tracking pixel scale")
    return np.diag([*scale, 1.0]) @ np.asarray(calibration["camera_matrix"])


def track_rays(row: TrackFrame, calibration: dict[str, Any]):
    """Return ({track ID: normalized ray (x, y)}, tracking-pixel K)."""
    import cv2 as cv
    import numpy as np

    camera = tracking_camera_matrix(row, calibration)
    identifiers = [track.id for track in row.tracks]
    pixels = np.asarray([track.xy for track in row.tracks], dtype=float).reshape(-1, 2)
    if not np.isfinite(pixels).all():
        raise ValueError("Nonfinite track coordinates")
    if len(pixels):
        normalized = cv.undistortPoints(
            pixels.reshape(-1, 1, 2),
            camera,
            np.asarray(calibration["distortion_coefficients"]),
        ).reshape(-1, 2)
    else:
        normalized = np.empty((0, 2))
    return dict(zip(identifiers, normalized)), camera
