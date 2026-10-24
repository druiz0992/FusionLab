"""Two-view landmark triangulation from normalized rays and known poses.

Poses are 3×4 world-to-camera matrices [R | t] (X_camera = R X_world + t) and
rays are normalized, undistorted image coordinates. Points come out in the
poses' world frame and share their arbitrary segment scale. A candidate is
usable only if it is in front of both cameras, reprojects well in both views
and its two viewing rays meet at a large enough angle (parallax); with little
parallax the depth along the ray is poorly constrained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fusionlab._typing import Array

# Homogeneous weights below this are treated as points at infinity.
MIN_HOMOGENEOUS_WEIGHT = 1e-10


@dataclass
class TriangulationCounts:
    """Outcome counts for one triangulation pass, in saved JSON key order."""

    candidates: int = 0
    accepted: int = 0
    invalid_depth_or_homogeneous: int = 0
    high_reprojection_error: int = 0
    low_parallax: int = 0
    accepted_parallax_deg: list[float] = field(default_factory=list)
    accepted_max_reprojection_px: list[float] = field(default_factory=list)


def camera_center(pose: Array) -> Array:
    """World position of a camera: C = -R.T t."""
    return -pose[:, :3].T @ pose[:, 3]


def triangulate_rays(
    rays_a: Array, rays_b: Array, pose_a: Array, pose_b: Array
) -> tuple[Array, Array, Array, Array]:
    """Linear (DLT) triangulation of matching normalized rays.

    Returns world points (N, 3), the same points in camera a and camera b
    coordinates, and a mask that is False for points at infinity, behind
    either camera, or nonfinite. Masked points are left at the origin.
    """
    import cv2

    import numpy as np

    homogeneous = cv2.triangulatePoints(pose_a, pose_b, rays_a.T, rays_b.T)
    valid = np.abs(homogeneous[3]) > MIN_HOMOGENEOUS_WEIGHT
    points = np.zeros((len(rays_a), 3))
    points[valid] = (homogeneous[:3, valid] / homogeneous[3, valid]).T
    camera_a = points @ pose_a[:, :3].T + pose_a[:, 3]
    camera_b = points @ pose_b[:, :3].T + pose_b[:, 3]
    valid &= (
        (camera_a[:, 2] > 0) & (camera_b[:, 2] > 0) & np.isfinite(points).all(axis=1)
    )
    return points, camera_a, camera_b, valid


def candidate_quality(
    point: Array,
    camera_a: Array,
    camera_b: Array,
    ray_a: Array,
    ray_b: Array,
    center_a: Array,
    center_b: Array,
    focal: float,
) -> tuple[float, float]:
    """Worst-view reprojection error (approx. px) and ray parallax (deg).

    `camera_a`/`camera_b` are the point in each camera's frame and
    `center_a`/`center_b` the camera centers in the world frame. Parallax is
    the angle at the point between the two lines of sight.
    """
    import numpy as np

    error_a = np.linalg.norm(camera_a[:2] / camera_a[2] - ray_a)
    error_b = np.linalg.norm(camera_b[:2] / camera_b[2] - ray_b)
    sight_a = point - center_a
    sight_b = point - center_b
    cosine = sight_a @ sight_b / (np.linalg.norm(sight_a) * np.linalg.norm(sight_b))
    parallax = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    return float(max(error_a, error_b) * focal), float(parallax)
