"""Calibrated two-view pose diagnostics; translation has unknown magnitude."""

import json
from pathlib import Path

from fusionlab.vision.tracks import read_track_frame_pair


def estimate_motion(
    points_a,
    points_b,
    camera_matrix,
    distortion,
    *,
    threshold_px=1.5,
    min_inliers=20,
    min_parallax_deg=1.0,
):
    """Estimate X_b = R_ba X_a + t_ba, with camera axes X right, Y down, Z forward.

    Positive-depth checks choose an essential-matrix decomposition. Rotation-
    compensated ray angles check whether translation is sufficiently observable.
    These heuristic checks do not replace independent trajectory validation.
    """
    import cv2
    import numpy as np

    a = np.asarray(points_a, dtype=float)
    b = np.asarray(points_b, dtype=float)
    camera = np.asarray(camera_matrix, dtype=float)
    distortion = np.asarray(distortion, dtype=float)
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

    # Correct lens distortion before estimating geometry in normalized camera rays.
    rays_a = cv2.undistortPoints(a.reshape(-1, 1, 2), camera, distortion).reshape(-1, 2)
    rays_b = cv2.undistortPoints(b.reshape(-1, 1, 2), camera, distortion).reshape(-1, 2)
    normalized_threshold = threshold_px / ((camera[0, 0] + camera[1, 1]) / 2)
    essential, mask = cv2.findEssentialMat(
        rays_a,
        rays_b,
        np.eye(3),
        method=cv2.RANSAC,
        prob=0.999,
        threshold=normalized_threshold,
    )
    if essential is None or mask is None or not np.isfinite(essential).all():
        raise ValueError("Essential matrix estimation failed")
    best = None
    # Some solvers return stacked 3x3 candidates; compare positive-depth support.
    for candidate in essential.reshape(-1, 3, 3):
        count, rotation, translation, pose_mask = cv2.recoverPose(
            candidate, rays_a, rays_b, np.eye(3), mask=mask.copy()
        )
        if best is None or count > best[0]:
            best = (count, candidate, rotation, translation, pose_mask)
    count, essential, rotation, translation, pose_mask = best
    accepted = pose_mask.ravel() != 0
    unit_a = np.column_stack((rays_a, np.ones(len(a))))
    unit_b = np.column_stack((rays_b, np.ones(len(a))))
    unit_a /= np.linalg.norm(unit_a, axis=1)[:, None]
    unit_b /= np.linalg.norm(unit_b, axis=1)[:, None]
    rotated_a = unit_a @ rotation.T
    angles = np.degrees(np.arccos(np.clip(np.sum(rotated_a * unit_b, axis=1), -1, 1)))
    median_parallax = float(np.median(angles[accepted])) if count else None
    reasons = []
    if count < min_inliers:
        reasons.append("insufficient_positive_depth_inliers")
    if count / len(a) < 0.5:
        reasons.append("low_positive_depth_inlier_ratio")
    if median_parallax is None or median_parallax < min_parallax_deg:
        reasons.append("low_translation_parallax")
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
        "rotation_angle_deg": float(
            np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
        ),
        "rotation_b_from_a": rotation.tolist(),
        "translation_b_from_a_unit": direction.tolist(),
        "camera_b_center_in_a_unit": center.tolist(),
        "essential_matrix": essential.tolist(),
        "thresholds": {
            "ransac_tracking_px": threshold_px,
            "min_inliers": min_inliers,
            "min_parallax_deg": min_parallax_deg,
            "min_inlier_ratio": 0.5,
        },
        "scope": "Pairwise candidate only. Translation magnitude is unknown; no trajectory or metric distance. Rejected poses must not be accumulated.",
        "warnings": [
            "Planar scenes, moving objects, rolling shutter and calibration mismatch can still produce misleading poses."
        ],
    }


def analyze_motion(
    tracks: Path, calibration: Path, output: Path, *, frame_a=1000, frame_b=1010
):
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
    report["warnings"].append(
        "Calibration belongs to a separate recording; matching camera, zoom and processing settings must be verified by the user."
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "motion.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report
