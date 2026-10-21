"""Synthetic pinhole camera, calibration and feature-track rows for vision tests.

The camera is an undistorted 640×480 pinhole with a 500-pixel focal length and
the principal point at the image center. Rows use the `tracks.jsonl` layout with
tracking pixels equal to original pixels (scale 1).
"""

CAMERA_MATRIX = [[500.0, 0, 320], [0, 500, 240], [0, 0, 1]]
IMAGE_SIZE = [640, 480]


def calibration_json() -> dict:
    """A `calibration.json` dict for the synthetic camera (no distortion)."""
    return {
        "image_size": list(IMAGE_SIZE),
        "camera_matrix": [list(row) for row in CAMERA_MATRIX],
        "distortion_coefficients": [0.0] * 5,
        "distortion_order": ["k1", "k2", "p1", "p2", "k3"],
    }


def track_row(time_ns: int, tracks=(), *, segment: int = 1) -> dict:
    """One `tracks.jsonl` row on stream "s"; source and receive times are equal."""
    return {
        "original_size": list(IMAGE_SIZE),
        "tracking_size": list(IMAGE_SIZE),
        "pixel_scale_xy": [1.0, 1.0],
        "stream_id": "s",
        "segment": segment,
        "source_time_ns": time_ns,
        "received_monotonic_ns": time_ns,
        "tracks": list(tracks),
    }


def translating_camera_rows(frames: int) -> list[dict]:
    """Rows for 200 static points seen by a camera moving 0.25 units/frame along +X.

    Points are uniform in x∈[-2, 2], y∈[-1, 1], z∈[4, 8] (seed 30); frames are
    100 ms apart. The camera center after frame i is (0.25 i, 0, 0).
    """
    import cv2 as cv
    import numpy as np

    points = np.random.default_rng(30).uniform([-2, -1, 4], [2, 1, 8], (200, 3))
    camera = np.array(CAMERA_MATRIX)
    rows = []
    for i in range(frames):
        pixels, _ = cv.projectPoints(
            points, np.zeros(3), np.array([-0.25 * i, 0.0, 0.0]), camera, np.zeros(5)
        )
        tracks = [
            {"id": j, "xy": p.tolist()} for j, p in enumerate(pixels.reshape(-1, 2))
        ]
        rows.append(track_row(i * 100_000_000, tracks))
    return rows
