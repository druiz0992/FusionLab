"""Calibrated two-view pose diagnostics; translation has unknown magnitude."""

import json
from pathlib import Path
from typing import Any

from fusionlab.vision.tracks import read_track_frame_pair

# Method caveats saved with every report (what the numbers do not show).
MOTION_SCOPE = (
    "Pairwise candidate only. Translation magnitude is unknown; no trajectory or "
    "metric distance. Rejected poses must not be accumulated."
)

MOTION_WARNING = (
    "Planar scenes, moving objects, rolling shutter and calibration mismatch can "
    "still produce misleading poses."
)

SEPARATE_CALIBRATION_WARNING = (
    "Calibration belongs to a separate recording; matching camera, zoom and "
    "processing settings must be verified by the user."
)


# A pair is accepted only if at least this share of matches has positive depth.
MIN_POSITIVE_DEPTH_RATIO = 0.5
# RANSAC confidence for the essential matrix.
ESSENTIAL_RANSAC_CONFIDENCE = 0.999


def estimate_motion(
    points_a,
    points_b,
    camera_matrix,
    distortion,
    *,
    threshold_px: float = 1.5,
    min_inliers: int = 20,
    min_parallax_deg: float = 1.0,
) -> dict[str, Any]:
    """Estimate X_b = R_ba X_a + t_ba, with camera axes X right, Y down, Z forward.

    `points_a`/`points_b` are matching pixel coordinates (N by 2) in the image
    described by `camera_matrix` and the five OpenCV `distortion` coefficients.
    Positive-depth checks choose an essential-matrix decomposition. Rotation-
    compensated ray angles check whether translation is sufficiently observable.
    The translation has unit length: its magnitude is unobservable from two views.
    These heuristic checks do not replace independent trajectory validation.
    """
    import numpy as np

    a = np.asarray(points_a, dtype=float)
    b = np.asarray(points_b, dtype=float)
    camera = np.asarray(camera_matrix, dtype=float)
    distortion = np.asarray(distortion, dtype=float)
    _validate_motion_inputs(
        a, b, camera, distortion, threshold_px, min_inliers, min_parallax_deg
    )
    rays_a = _undistort(a, camera, distortion)
    rays_b = _undistort(b, camera, distortion)
    # The RANSAC threshold is given in pixels; rays are in normalized units.
    normalized_threshold = threshold_px / ((camera[0, 0] + camera[1, 1]) / 2)
    essential, mask = _essential_matrix(rays_a, rays_b, normalized_threshold)
    count, essential, rotation, translation, pose_mask = _best_decomposition(
        essential, rays_a, rays_b, mask
    )
    accepted = pose_mask.ravel() != 0
    angles = _rotation_compensated_parallax_deg(rays_a, rays_b, rotation)
    median_parallax = float(np.median(angles[accepted])) if count else None
    reasons = _rejection_reasons(
        count, len(a), median_parallax, min_inliers, min_parallax_deg
    )
    direction = translation.ravel()
    center = -rotation.T @ direction
    return {
        "status": "accepted_pair" if not reasons else "rejected_pair",
        "rejection_reasons": reasons,
        "matches": len(a),
        "essential_inliers": int((mask != 0).sum()),
        "positive_depth_inliers": int(count),
        "inlier_mask": accepted.tolist(),
        "median_parallax_deg": median_parallax,
        "rotation_angle_deg": _rotation_angle_deg(rotation),
        "rotation_b_from_a": rotation.tolist(),
        "translation_b_from_a_unit": direction.tolist(),
        "camera_b_center_in_a_unit": center.tolist(),
        "essential_matrix": essential.tolist(),
        "thresholds": {
            "ransac_tracking_px": threshold_px,
            "min_inliers": min_inliers,
            "min_parallax_deg": min_parallax_deg,
            "min_inlier_ratio": MIN_POSITIVE_DEPTH_RATIO,
        },
        "scope": MOTION_SCOPE,
        "warnings": [MOTION_WARNING],
    }


def _validate_motion_inputs(
    a, b, camera, distortion, threshold_px, min_inliers, min_parallax_deg
) -> None:
    """Reject malformed correspondences, cameras and thresholds."""
    import numpy as np

    if a.ndim != 2 or a.shape[1:] != (2,) or b.shape != a.shape or len(a) < 8:
        raise ValueError("Need at least eight matching N by 2 correspondences")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Correspondences must be finite")
    if (
        camera.shape != (3, 3)
        or not np.isfinite(camera).all()
        or camera[0, 0] <= 0
        or camera[1, 1] <= 0
        or not np.allclose(camera[2], [0, 0, 1])
    ):
        raise ValueError("Invalid pinhole camera matrix")
    if distortion.size != 5 or not np.isfinite(distortion).all():
        raise ValueError("Expected five finite OpenCV distortion coefficients")
    if (
        not np.isfinite(threshold_px)
        or threshold_px <= 0
        or min_inliers < 8
        or not np.isfinite(min_parallax_deg)
        or min_parallax_deg < 0
    ):
        raise ValueError("Invalid motion quality thresholds")


def _undistort(pixels, camera, distortion):
    """Pixels to normalized, distortion-corrected camera rays (N by 2)."""
    import cv2

    return cv2.undistortPoints(pixels.reshape(-1, 1, 2), camera, distortion).reshape(
        -1, 2
    )


def _essential_matrix(rays_a, rays_b, normalized_threshold: float):
    """RANSAC essential matrix on normalized rays (identity camera)."""
    import cv2
    import numpy as np

    essential, mask = cv2.findEssentialMat(
        rays_a,
        rays_b,
        np.eye(3),
        method=cv2.RANSAC,
        prob=ESSENTIAL_RANSAC_CONFIDENCE,
        threshold=normalized_threshold,
    )
    if essential is None or mask is None or not np.isfinite(essential).all():
        raise ValueError("Essential matrix estimation failed")
    return essential, mask


def _best_decomposition(essential, rays_a, rays_b, mask):
    """Pick the (R, t) with most positive-depth points over all E candidates.

    Some solvers return several stacked 3x3 essential matrices; each has four
    (R, t) decompositions, and recoverPose keeps the one placing most points in
    front of both cameras. Returns (count, E, R, t, positive-depth mask).
    """
    import cv2
    import numpy as np

    best = None
    for candidate in essential.reshape(-1, 3, 3):
        count, rotation, translation, pose_mask = cv2.recoverPose(
            candidate, rays_a, rays_b, np.eye(3), mask=mask.copy()
        )
        if best is None or count > best[0]:
            best = (count, candidate, rotation, translation, pose_mask)
    return best


def _rotation_compensated_parallax_deg(rays_a, rays_b, rotation):
    """Angle between each ray in b and its ray in a rotated into b's frame.

    Pure rotation gives zero angles, so these measure how much translation the
    pair can reveal, independent of camera turning.
    """
    import numpy as np

    unit_a = np.column_stack((rays_a, np.ones(len(rays_a))))
    unit_b = np.column_stack((rays_b, np.ones(len(rays_a))))
    unit_a /= np.linalg.norm(unit_a, axis=1)[:, None]
    unit_b /= np.linalg.norm(unit_b, axis=1)[:, None]
    rotated_a = unit_a @ rotation.T
    return np.degrees(np.arccos(np.clip(np.sum(rotated_a * unit_b, axis=1), -1, 1)))


def _rejection_reasons(
    count: int,
    matches: int,
    median_parallax: float | None,
    min_inliers: int,
    min_parallax_deg: float,
) -> list[str]:
    """Names of every failed acceptance check; empty when the pair is accepted."""
    reasons = []
    if count < min_inliers:
        reasons.append("insufficient_positive_depth_inliers")
    if count / matches < MIN_POSITIVE_DEPTH_RATIO:
        reasons.append("low_positive_depth_inlier_ratio")
    if median_parallax is None or median_parallax < min_parallax_deg:
        reasons.append("low_translation_parallax")
    return reasons


def _rotation_angle_deg(rotation) -> float:
    """Rotation magnitude from trace(R) = 1 + 2 cos(theta)."""
    import numpy as np

    return float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))))


def analyze_motion(
    tracks: Path,
    calibration: Path,
    output: Path,
    *,
    frame_a: int = 1000,
    frame_b: int = 1010,
) -> dict[str, Any]:
    """Load one tracking pair, scale calibration to tracking pixels, save diagnostics."""
    import numpy as np

    a, b = read_track_frame_pair(tracks, frame_a, frame_b)
    pair_geometry = (
        "stream_id",
        "segment",
        "tracking_size",
        "original_size",
        "pixel_scale_xy",
    )
    if any(getattr(a, key) != getattr(b, key) for key in pair_geometry):
        raise ValueError("Pair must share stream, segment and image geometry")
    calibration_data = json.loads(calibration.read_text())
    if tuple(calibration_data["image_size"]) != a.original_size:
        raise ValueError("Calibration image dimensions do not match original capture")
    if calibration_data.get("distortion_order") != ["k1", "k2", "p1", "p2", "k3"]:
        raise ValueError("Unsupported calibration distortion model")
    camera = np.asarray(calibration_data["camera_matrix"], dtype=float)
    # Track coordinates are resized, not display-rotated. Scale each image axis.
    sx, sy = a.pixel_scale_xy
    expected = np.asarray(a.tracking_size) / np.asarray(a.original_size)
    if not np.allclose([sx, sy], expected):
        raise ValueError("Inconsistent tracking coordinate scale")
    camera = np.diag([sx, sy, 1.0]) @ camera
    first = a.positions()
    second = b.positions()
    shared = sorted(first.keys() & second.keys())
    report = estimate_motion(
        [first[key] for key in shared],
        [second[key] for key in shared],
        camera,
        calibration_data["distortion_coefficients"],
    )
    report.update(
        frame_a=frame_a,
        frame_b=frame_b,
        track_ids=shared,
        source_interval_s=(b.source_time_ns - a.source_time_ns) / 1e9,
        tracks=str(tracks.resolve()),
        calibration=str(calibration.resolve()),
        calibration_status=calibration_data.get("status", "unknown"),
        tracking_camera_matrix=camera.tolist(),
    )
    report["warnings"].append(SEPARATE_CALIBRATION_WARNING)
    output.mkdir(parents=True, exist_ok=False)
    (output / "motion.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report
