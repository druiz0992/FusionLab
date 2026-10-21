"""Uncalibrated image-pair geometry. No metric motion or camera pose is inferred."""

import json
from hashlib import sha256
from pathlib import Path

from fusionlab.vision.tracks import read_track_frame_pair


def estimate_geometry(points_a, points_b, *, threshold_px=1.5):
    """Fit x_b.T F x_a = 0 and report Sampson residuals in tracking pixels.

    RANSAC inliers are a consistency test, not independent ground truth. Planar
    scenes and pure rotation can yield ambiguous fundamental matrices.
    """
    import cv2
    import numpy as np

    a = np.asarray(points_a, dtype=float)
    b = np.asarray(points_b, dtype=float)
    if a.ndim != 2 or a.shape[1:] != (2,) or b.shape != a.shape:
        raise ValueError("Correspondences must be matching N by 2 arrays")
    if len(a) < 8 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Need at least eight finite correspondences")
    if not np.isfinite(threshold_px) or threshold_px <= 0:
        raise ValueError("RANSAC threshold must be finite and positive")
    fundamental, mask = cv2.findFundamentalMat(a, b, cv2.FM_RANSAC, threshold_px, 0.99)
    if (
        fundamental is None
        or fundamental.shape != (3, 3)
        or not np.isfinite(fundamental).all()
        or mask is None
    ):
        raise ValueError(
            "Fundamental matrix estimation failed; choose a different frame pair"
        )
    inliers = mask.ravel().astype(bool)
    if len(inliers) != len(a) or inliers.sum() < 8:
        raise ValueError(
            "Fewer than eight geometric inliers; choose a different frame pair"
        )
    # Homogeneous pixel coordinates and first-order distance to the epipolar constraint.
    homogeneous_a = np.column_stack((a, np.ones(len(a))))
    homogeneous_b = np.column_stack((b, np.ones(len(b))))
    lines_b = homogeneous_a @ fundamental.T
    lines_a = homogeneous_b @ fundamental
    numerator = np.sum(homogeneous_b * lines_b, axis=1)
    denominator = np.sum(lines_a[:, :2] ** 2 + lines_b[:, :2] ** 2, axis=1)
    if np.any(denominator <= 1e-20):
        raise ValueError("Degenerate epipolar lines; choose a different frame pair")
    residual = np.abs(numerator) / np.sqrt(denominator)
    # Homography support is only a warning indicator, not a competing pose estimate.
    _, homography_mask = cv2.findHomography(a, b, cv2.RANSAC, threshold_px)
    homography_inliers = (
        int(homography_mask.sum()) if homography_mask is not None else 0
    )
    return {
        "fundamental_matrix": fundamental.tolist(),
        "matches": len(a),
        "inliers": int(inliers.sum()),
        "inlier_mask": inliers.tolist(),
        "sampson_residual_px": residual.tolist(),
        "median_inlier_sampson_px": float(np.median(residual[inliers])),
        "median_image_displacement_px": float(np.median(np.linalg.norm(b - a, axis=1))),
        "homography_inliers": homography_inliers,
        "warnings": (
            [
                "Strong homography support: planar scene or rotation may make geometry ambiguous."
            ]
            if homography_inliers >= 0.9 * inliers.sum()
            else []
        ),
        "scope": "Uncalibrated epipolar geometry only; no pose, scale or verified motion. Distortion is not corrected.",
    }


def _draw_lines(image, lines, points):
    """Clip homogeneous lines against image edges, including near-vertical lines."""
    import cv2

    height, width = image.shape[:2]
    for index, ((a, b, c), point) in enumerate(zip(lines, points)):
        color = (
            (index * 71 + 80) % 256,
            (index * 139 + 100) % 256,
            (index * 47 + 150) % 256,
        )
        intersections = []
        if abs(b) > 1e-12:
            intersections.extend((x, -(a * x + c) / b) for x in (0, width - 1))
        if abs(a) > 1e-12:
            intersections.extend((-(b * y + c) / a, y) for y in (0, height - 1))
        visible = [
            (round(x), round(y))
            for x, y in intersections
            if 0 <= x < width and 0 <= y < height
        ]
        if len(visible) >= 2:
            cv2.line(image, visible[0], visible[-1], color, 1)
        cv2.circle(image, tuple(round(float(v)) for v in point), 4, color, -1)


def analyze_pair(
    capture: Path,
    tracks: Path,
    output: Path,
    *,
    frame_a=0,
    frame_b=10,
    display_rotation=90,
):
    """Read shared track IDs and verify original image assets before rendering."""
    import cv2
    import numpy as np

    if display_rotation not in (0, 90, 180, 270):
        raise ValueError("Invalid display rotation")
    first, second = read_track_frame_pair(tracks, frame_a, frame_b)
    selected = {frame_a: first, frame_b: second}
    if first.geometry_key != second.geometry_key:
        raise ValueError("Frames must share a stream, segment and tracking dimensions")
    lookup_a = first.positions()
    lookup_b = second.positions()
    identifiers = sorted(lookup_a.keys() & lookup_b.keys())
    points_a = np.array([lookup_a[key] for key in identifiers]).reshape(-1, 2)
    points_b = np.array([lookup_b[key] for key in identifiers]).reshape(-1, 2)
    report = estimate_geometry(points_a, points_b)
    images = {}
    with (capture / "events.jsonl").open() as file:
        for line in file:
            event = json.loads(line)
            for index, row in selected.items():
                if (
                    event["type"] != "CameraSample"
                    or event["stream_id"] != row.stream_id
                    or event["source_time_ns"] != row.source_time_ns
                    or event["received_monotonic_ns"] != row.received_monotonic_ns
                ):
                    continue
                path = (capture / event["image_path"]).resolve()
                if not path.is_relative_to(capture.resolve()):
                    raise ValueError("Image path escapes capture")
                raw = path.read_bytes()
                if (
                    len(raw) != event["image_size_bytes"]
                    or sha256(raw).hexdigest() != event["image_sha256"]
                ):
                    raise ValueError("Camera asset integrity failure")
                image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("Image decode failed")
                images[index] = cv2.resize(
                    image, row.tracking_size, interpolation=cv2.INTER_AREA
                )
            if len(images) == 2:
                break
    if len(images) != 2:
        raise ValueError("Tracking frames were not found in this capture")
    inliers = np.flatnonzero(report["inlier_mask"])[:30]
    fundamental = np.array(report["fundamental_matrix"])
    lines_a = np.column_stack((points_b[inliers], np.ones(len(inliers)))) @ fundamental
    lines_b = (
        np.column_stack((points_a[inliers], np.ones(len(inliers)))) @ fundamental.T
    )
    _draw_lines(images[frame_a], lines_a, points_a[inliers])
    _draw_lines(images[frame_b], lines_b, points_b[inliers])
    output.mkdir(parents=True, exist_ok=False)
    for index, image in images.items():
        if display_rotation:
            image = cv2.rotate(
                image,
                {
                    90: cv2.ROTATE_90_CLOCKWISE,
                    180: cv2.ROTATE_180,
                    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
                }[display_rotation],
            )
        if not cv2.imwrite(str(output / f"frame_{index:06d}.png"), image):
            raise ValueError("Image write failed")
    report.update(
        frame_a=frame_a,
        frame_b=frame_b,
        track_ids=identifiers,
        source_interval_s=(second.source_time_ns - first.source_time_ns) / 1e9,
        tracking_size=list(first.tracking_size),
        display_rotation_clockwise_deg=display_rotation,
        coordinate_convention="Unrotated tracking pixels, +X right and +Y down; x_b.T F x_a = 0",
    )
    (output / "geometry.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report
