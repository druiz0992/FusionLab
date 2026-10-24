"""Perspective-n-point pose hypotheses against an existing landmark map.

Every image coordinate here is a normalized, undistorted ray (x, y), i.e. an
observation by an identity camera matrix, so pixel thresholds are converted with
the mean tracking focal length. Poses are world-to-camera, X_camera = R X_world
+ t, carried as OpenCV Rodrigues rotation and translation vectors. A landmark
supports a pose when it reprojects within PNP_INLIER_THRESHOLD_PX and lies in
front of the camera.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fusionlab._typing import Array

# Reprojection threshold for PnP support, in approximate tracking pixels.
PNP_INLIER_THRESHOLD_PX = 3.0
# Iterative PnP needs this many correspondences to refine a seed pose.
MIN_REFINEMENT_CORRESPONDENCES = 6
# EPnP RANSAC budget and the probability of drawing one outlier-free sample.
RANSAC_ITERATIONS = 100
RANSAC_CONFIDENCE = 0.999


@dataclass(frozen=True)
class PnpCandidate:
    """A solved pose hypothesis and the correspondences it was solved from.

    `objects` are map landmarks in the segment world frame (arbitrary units)
    and `rays` their normalized observations in this frame, row for row.
    """

    objects: Array  # (N, 3) world landmarks
    rays: Array  # (N, 2) normalized image coordinates
    rotation: Array  # (3, 1) Rodrigues world-to-camera rotation
    translation: Array  # (3, 1) world-to-camera translation
    inliers: Array  # (K, 1) indices into objects/rays; may be empty

    def world_to_camera(self) -> Array:
        """The 3×4 matrix [R | t]."""
        import cv2

        import numpy as np

        return np.column_stack(
            (cv2.Rodrigues(self.rotation)[0], self.translation.ravel())
        )

    def reprojection_errors_px(self, focal: float) -> Array:
        """Per-landmark reprojection error at this pose, in approximate pixels."""
        return reprojection_errors_px(
            self.objects, self.rays, self.rotation, self.translation, focal
        )


def solve_pnp_ransac(objects: Array, rays: Array, focal: float) -> PnpCandidate | None:
    """EPnP RANSAC with a 3-pixel-equivalent threshold; None if it fails.

    OpenCV reports inliers whenever it succeeds; an absent set is treated as
    empty so the support gates, not this function, reject it.
    """
    import cv2

    import numpy as np

    ok, rotation, translation, inliers = cv2.solvePnPRansac(
        objects,
        rays,
        np.eye(3),
        None,
        iterationsCount=RANSAC_ITERATIONS,
        reprojectionError=PNP_INLIER_THRESHOLD_PX / focal,
        confidence=RANSAC_CONFIDENCE,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok:
        return None
    if inliers is None:
        inliers = np.empty((0, 1), dtype=int)
    return PnpCandidate(objects, rays, rotation, translation, inliers)


def refine_on_inliers(
    objects: Array, rays: Array, rotation: Array, translation: Array, inliers: Array
) -> tuple[bool, Array, Array]:
    """Levenberg–Marquardt PnP on the inlier subset, seeded with a pose.

    Returns (ok, rotation, translation). The seed is copied because OpenCV may
    update an extrinsic guess in place.
    """
    import cv2

    import numpy as np

    selected = np.asarray(inliers).ravel()
    ok, refined_rotation, refined_translation = cv2.solvePnP(
        objects[selected],
        rays[selected],
        np.eye(3),
        None,
        rotation.copy(),
        translation.copy(),
        True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    return bool(ok), refined_rotation, refined_translation


def positive_depth_support(
    objects: Array, rays: Array, rotation: Array, translation: Array, focal: float
) -> Array:
    """(K, 1) indices of landmarks within the threshold and in front of the camera."""
    import cv2

    import numpy as np

    residuals = reprojection_errors_px(objects, rays, rotation, translation, focal)
    camera = objects @ cv2.Rodrigues(rotation)[0].T + translation.ravel()
    return np.flatnonzero(
        np.isfinite(residuals)
        & (residuals <= PNP_INLIER_THRESHOLD_PX)
        & (camera[:, 2] > 0)
    ).reshape(-1, 1)


def refine_pose_support(
    objects: Array,
    pixels: Array,
    rotation: Array,
    translation: Array,
    inliers: Array | None,
    focal: float,
) -> tuple[bool, Array, Array, Array]:
    """Refine a candidate once, then recompute positive-depth 3-pixel support.

    `pixels` are normalized rays. The seed needs six correspondences for
    iterative PnP; the final acceptance gates (20 correspondences, 50% support)
    are applied by the caller. Returns (ok, rotation, translation, support);
    on failure the seed pose is returned with empty support.
    """
    import numpy as np

    no_support = np.empty((0, 1), dtype=int)
    if inliers is None or len(inliers) < MIN_REFINEMENT_CORRESPONDENCES:
        return False, rotation, translation, no_support
    ok, refined_rotation, refined_translation = refine_on_inliers(
        objects, pixels, rotation, translation, inliers
    )
    if (
        not ok
        or not np.isfinite(refined_rotation).all()
        or not np.isfinite(refined_translation).all()
    ):
        return False, rotation, translation, no_support
    support = positive_depth_support(
        objects, pixels, refined_rotation, refined_translation, focal
    )
    return True, refined_rotation, refined_translation, support


def reprojection_errors_px(
    objects: Array, rays: Array, rotation: Array, translation: Array, focal: float
) -> Array:
    """Distance between projected landmarks and observed rays, approx. px.

    Points behind the camera still project (with flipped sign); callers that
    need physical support also test depth.
    """
    import cv2

    import numpy as np

    projected, _ = cv2.projectPoints(objects, rotation, translation, np.eye(3), None)
    return np.linalg.norm(projected.reshape(-1, 2) - rays, axis=1) * focal
