"""Experimental sparse monocular VO. Each reset has independent arbitrary scale."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fusionlab.vision.calibrated_motion import estimate_motion
from fusionlab.vision.odometry.frame_result import FrameResult
from fusionlab.vision.tracks import TrackFrame

# Frames further apart than this (source clock) start a new segment.
MAX_FRAME_GAP_NS = 500_000_000
# Minimum bootstrap tracks, triangulated landmarks, PnP correspondences and inliers.
MIN_MAP_CORRESPONDENCES = 20
# PnP inliers must also be at least this share of visible map landmarks.
MIN_PNP_SUPPORT_RATIO = 0.5
# Reprojection threshold for PnP support, in approximate tracking pixels.
PNP_INLIER_THRESHOLD_PX = 3.0


@dataclass
class _PnpCandidate:
    """A PnP pose hypothesis and the correspondences it was solved from.

    `objects` are map landmarks (segment world frame, arbitrary units) and
    `pixels` their normalized image rays in this frame. Rotation/translation
    are OpenCV Rodrigues/translation vectors for world-to-camera.
    """

    ok: bool = False
    rotation: Any = None
    translation: Any = None
    inliers: Any = None  # (N, 1) indices into objects/pixels
    objects: Any = None
    pixels: Any = None


def refine_pose_support(objects, pixels, rotation, translation, inliers, focal):
    """Refine a candidate once, then recompute positive-depth 3-pixel support.

    The seed needs six correspondences for iterative PnP; the final acceptance
    gates still require 20 correspondences and 50% support in the caller.
    Inputs are copied because OpenCV may update its pose guess in place.
    """
    import cv2
    import numpy as np

    if inliers is None or len(inliers) < 6:
        return False, rotation, translation, np.empty((0, 1), dtype=int)
    accepted = np.asarray(inliers).ravel()
    ok, refined_rotation, refined_translation = cv2.solvePnP(
        objects[accepted],
        pixels[accepted],
        np.eye(3),
        None,
        rotation.copy(),
        translation.copy(),
        True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if (
        not ok
        or not np.isfinite(refined_rotation).all()
        or not np.isfinite(refined_translation).all()
    ):
        return False, rotation, translation, np.empty((0, 1), dtype=int)
    projected, _ = cv2.projectPoints(
        objects, refined_rotation, refined_translation, np.eye(3), None
    )
    residuals = np.linalg.norm(projected.reshape(-1, 2) - pixels, axis=1) * focal
    camera = (
        objects @ cv2.Rodrigues(refined_rotation)[0].T + refined_translation.ravel()
    )
    support = np.flatnonzero(
        np.isfinite(residuals)
        & (residuals <= PNP_INLIER_THRESHOLD_PX)
        & (camera[:, 2] > 0)
    ).reshape(-1, 1)
    return True, refined_rotation, refined_translation, support


def choose_keyframe(
    reference_points,
    reference_pose,
    current_points,
    current_pose,
    *,
    parallax_deg=2.0,
    survival_ratio=0.6,
):
    """Balance rotation-compensated ray motion against loss of shared tracks.

    This uses observed rays, not estimated landmark depths. Rotation-only motion
    should not trigger the parallax condition. Low survival refreshes the
    reference before more tracks disappear, even if depth geometry remains weak.
    """
    import numpy as np

    shared = sorted(reference_points.keys() & current_points.keys())
    survival = len(shared) / len(reference_points) if reference_points else 0.0
    median_angle = 0.0
    if shared:
        a = np.column_stack(
            ([reference_points[key] for key in shared], np.ones(len(shared)))
        )
        b = np.column_stack(
            ([current_points[key] for key in shared], np.ones(len(shared)))
        )
        a /= np.linalg.norm(a, axis=1)[:, None]
        b /= np.linalg.norm(b, axis=1)[:, None]
        relative_rotation = current_pose[:, :3] @ reference_pose[:, :3].T
        cosines = np.sum((a @ relative_rotation.T) * b, axis=1)
        median_angle = float(np.median(np.degrees(np.arccos(np.clip(cosines, -1, 1)))))
    if survival < survival_ratio or len(shared) < 20:
        reason = "track_survival"
    elif median_angle >= parallax_deg:
        reason = "parallax"
    else:
        reason = "wait_for_baseline"
    return {
        "selected": reason != "wait_for_baseline",
        "reason": reason,
        "shared_tracks": len(shared),
        "track_survival_ratio": survival,
        "median_rotation_compensated_parallax_deg": median_angle,
        "parallax_threshold_deg": parallax_deg,
        "survival_threshold": survival_ratio,
    }


class MonocularOdometry:
    """Triangulate persistent feature IDs, then recover world-to-camera poses.

    A segment's first camera defines the world axes. Its initialization baseline
    is one arbitrary unit. No unit translations are accumulated after bootstrap:
    subsequent translations come from PnP against the existing 3D landmarks.
    """

    def __init__(
        self,
        calibration,
        *,
        pose_acceptance="baseline",
        keyframe_policy="baseline",
        landmark_policy="baseline",
        bundle_adjustment=False,
    ):
        if pose_acceptance not in ("baseline", "refined"):
            raise ValueError("Unknown pose acceptance method")
        if keyframe_policy not in ("baseline", "parallax"):
            raise ValueError("Unknown keyframe policy")
        if landmark_policy not in ("baseline", "validation", "refinement", "multiview"):
            raise ValueError("Unknown landmark policy")
        self.bundle_adjustment = bundle_adjustment
        self.landmark_policy = landmark_policy
        self.keyframe_policy = keyframe_policy
        self.pose_acceptance = pose_acceptance
        import cv2
        import numpy as np

        self.cv, self.np = cv2, np
        self.calibration = calibration
        self.anchor = None
        self.reference = None
        self.pose = None
        self.landmarks = {}
        self.bundle_window = []
        self.pending_landmarks = {}
        self.landmark_observations = {}
        self.confirmed_landmarks = set()
        self.segment = 0
        self.geometry = None
        self.previous_time = None
        self.landmark_birth = {}
        self.step = 0
        self.triangulation_diagnostics = {}
        self.last_pnp_snapshot = None

    def _normalized(self, row: TrackFrame):
        """Undistort track pixels into normalized camera rays, keyed by track ID.

        Calibration is in original stored pixels; tracks are in resized tracking
        pixels, so K is scaled per axis before undistortion.
        """
        np, cv = self.np, self.cv
        if row.original_size != tuple(self.calibration["image_size"]):
            raise ValueError("Calibration and original image dimensions differ")
        if self.calibration.get("distortion_order") != ["k1", "k2", "p1", "p2", "k3"]:
            raise ValueError("Unsupported distortion model")
        scale = np.asarray(row.tracking_size) / np.asarray(row.original_size)
        if not np.allclose(scale, row.pixel_scale_xy):
            raise ValueError("Inconsistent tracking pixel scale")
        camera = np.diag([*scale, 1.0]) @ np.asarray(self.calibration["camera_matrix"])
        identifiers = [track.id for track in row.tracks]
        pixels = np.asarray([track.xy for track in row.tracks], dtype=float).reshape(
            -1, 2
        )
        if not np.isfinite(pixels).all():
            raise ValueError("Nonfinite track coordinates")
        if len(pixels):
            normalized = cv.undistortPoints(
                pixels.reshape(-1, 1, 2),
                camera,
                np.asarray(self.calibration["distortion_coefficients"]),
            ).reshape(-1, 2)
        else:
            normalized = np.empty((0, 2))
        return dict(zip(identifiers, normalized)), camera

    def _triangulate(
        self, points_a, pose_a, points_b, pose_b, focal, *, bootstrap=False
    ):
        """Accept positive-depth points with parallax and low reprojection error."""
        np, cv = self.np, self.cv
        identifiers = sorted(
            (points_a.keys() & points_b.keys())
            - self.landmarks.keys()
            - self.pending_landmarks.keys()
        )
        self.triangulation_diagnostics = {
            "candidates": len(identifiers),
            "accepted": 0,
            "invalid_depth_or_homogeneous": 0,
            "high_reprojection_error": 0,
            "low_parallax": 0,
            "accepted_parallax_deg": [],
            "accepted_max_reprojection_px": [],
        }
        if not identifiers:
            return 0
        a = np.array([points_a[key] for key in identifiers])
        b = np.array([points_b[key] for key in identifiers])
        homogeneous = cv.triangulatePoints(pose_a, pose_b, a.T, b.T)
        valid = np.abs(homogeneous[3]) > 1e-10
        points = np.zeros((len(identifiers), 3))
        points[valid] = (homogeneous[:3, valid] / homogeneous[3, valid]).T
        camera_a = points @ pose_a[:, :3].T + pose_a[:, 3]
        camera_b = points @ pose_b[:, :3].T + pose_b[:, 3]
        valid &= (
            (camera_a[:, 2] > 0)
            & (camera_b[:, 2] > 0)
            & np.isfinite(points).all(axis=1)
        )
        for index, key in enumerate(identifiers):
            if not valid[index]:
                self.triangulation_diagnostics["invalid_depth_or_homogeneous"] += 1
                continue
            point = points[index]
            error_a = np.linalg.norm(
                camera_a[index, :2] / camera_a[index, 2] - a[index]
            )
            error_b = np.linalg.norm(
                camera_b[index, :2] / camera_b[index, 2] - b[index]
            )
            center_a = -pose_a[:, :3].T @ pose_a[:, 3]
            center_b = -pose_b[:, :3].T @ pose_b[:, 3]
            ray_a = point - center_a
            ray_b = point - center_b
            cosine = ray_a @ ray_b / (np.linalg.norm(ray_a) * np.linalg.norm(ray_b))
            parallax = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
            if max(error_a, error_b) * focal <= 2.0 and parallax >= 1.0:
                if self.landmark_policy != "baseline":
                    self.landmark_observations[key] = [
                        (pose_a.copy(), a[index].copy()),
                        (pose_b.copy(), b[index].copy()),
                    ]
                if (
                    self.landmark_policy in ("validation", "multiview")
                    and not bootstrap
                ):
                    self.pending_landmarks[key] = point
                else:
                    self.landmarks[key] = point
                self.landmark_birth[key] = self.step
                self.triangulation_diagnostics["accepted"] += 1
                self.triangulation_diagnostics["accepted_parallax_deg"].append(
                    float(parallax)
                )
                self.triangulation_diagnostics["accepted_max_reprojection_px"].append(
                    float(max(error_a, error_b) * focal)
                )
            elif max(error_a, error_b) * focal > 2.0:
                self.triangulation_diagnostics["high_reprojection_error"] += 1
            else:
                self.triangulation_diagnostics["low_parallax"] += 1
        return len(identifiers)

    def _update_landmarks(self, points, focal):
        """Validate a new observation after estimating pose from the active map.

        Bootstrap landmarks must provisionally seed the first PnP pose. Later
        candidates remain pending until a third view validates them; they never
        participate in estimating the pose used for their own third-view test.
        """
        from fusionlab.vision.odometry.landmark_refinement import (
            refine_landmark,
            reprojection_errors,
        )

        np = self.np
        diagnostics = {
            "confirmed": 0,
            "rejected_third_view": 0,
            "refined": 0,
            "pending": 0,
        }
        for pending, population in (
            (False, self.landmarks),
            (True, self.pending_landmarks),
        ):
            for key in list(population):
                if key not in points:
                    continue
                point = population[key]
                history = self.landmark_observations.get(key)
                if history is None:
                    continue
                observation = (self.pose.copy(), points[key].copy())
                error = reprojection_errors(point, [observation], focal)[0]
                if not np.isfinite(error) or error > 3.0:
                    if (
                        self.landmark_policy in ("validation", "multiview")
                        and key not in self.confirmed_landmarks
                    ):
                        del population[key]
                        self.landmark_observations.pop(key, None)
                        diagnostics["rejected_third_view"] += 1
                    continue
                # Preserve the initial view and four latest views to bound work.
                history = history + [observation]
                if len(history) > 5:
                    history = [history[0]] + history[-4:]
                candidate = (
                    refine_landmark(point, history)
                    if self.landmark_policy in ("refinement", "multiview")
                    else point.copy()
                )
                errors = reprojection_errors(candidate, history, focal)
                if np.isfinite(errors).all() and errors.max() <= 3.0:
                    self.landmark_observations[key] = history
                    population[key] = candidate
                    if self.landmark_policy in ("refinement", "multiview"):
                        diagnostics["refined"] += 1
                    if key not in self.confirmed_landmarks:
                        self.confirmed_landmarks.add(key)
                        diagnostics["confirmed"] += 1
                    if pending:
                        self.landmarks[key] = candidate
                        del self.pending_landmarks[key]
        live = set(points)
        self.pending_landmarks = {
            key: point for key, point in self.pending_landmarks.items() if key in live
        }
        keep = (self.landmarks.keys() | self.pending_landmarks.keys()) & live
        self.landmark_observations = {
            key: history
            for key, history in self.landmark_observations.items()
            if key in keep
        }
        self.confirmed_landmarks.intersection_update(keep)
        diagnostics["pending"] = len(self.pending_landmarks)
        return diagnostics

    def _adjust_bundle(self, points, focal):
        """Optimize a bounded local window and validate current map support."""
        from fusionlab.vision.odometry.local_bundle_adjustment import optimize_window

        np = self.np
        if not self.bundle_window:
            self.bundle_window.append(
                {
                    "pose": np.column_stack((np.eye(3), np.zeros(3))),
                    "points": self.anchor[1],
                    "step": None,
                    "received_monotonic_ns": self.anchor[0].received_monotonic_ns,
                }
            )
        self.bundle_window.append(
            {"pose": self.pose.copy(), "points": points, "step": self.step}
        )
        self.bundle_window = self.bundle_window[-5:]
        if len(self.bundle_window) < 4 or self.step % 3:
            return {"attempted": False, "reason": "waiting_for_window_or_interval"}
        window = self.bundle_window
        counts = {
            key: sum(key in frame["points"] for frame in window)
            for key in self.landmarks
        }
        identifiers = sorted(
            (key for key, count in counts.items() if count >= 3),
            key=lambda key: (-counts[key], key),
        )[:80]
        if len(identifiers) < 20:
            return {"attempted": False, "reason": "insufficient_multiview_landmarks"}
        poses = np.array([frame["pose"] for frame in window])
        objects = np.array([self.landmarks[key] for key in identifiers])
        observations = [
            (camera, index, frame["points"][key])
            for camera, frame in enumerate(window)
            for index, key in enumerate(identifiers)
            if key in frame["points"]
        ]
        if any(
            sum(o[0] == index for o in observations) < 10
            for index in range(len(window))
        ):
            return {"attempted": False, "reason": "insufficient_pose_observations"}
        try:
            new_poses, new_points, diagnostics = optimize_window(
                poses, objects, observations, focal=focal
            )
        except ValueError as error:
            # A near-stationary anchor baseline cannot constrain local scale.
            if "Anchor poses" in str(error):
                return {"attempted": False, "reason": "degenerate_anchor_baseline"}
            raise
        diagnostics.update(
            attempted=True,
            window_frames=len(window),
            optimized_landmarks=len(identifiers),
        )
        if not diagnostics["accepted"]:
            return diagnostics
        proposed = {**self.landmarks, **dict(zip(identifiers, new_points))}
        shared = sorted(self.landmarks.keys() & points.keys())

        def current_support(pose, landmarks):
            objects = np.array([landmarks[key] for key in shared])
            camera = objects @ pose[:, :3].T + pose[:, 3]
            errors = np.full(len(shared), np.inf)
            valid = (camera[:, 2] > 1e-6) & np.isfinite(camera).all(axis=1)
            observed = np.array([points[key] for key in shared])
            errors[valid] = (
                np.linalg.norm(
                    camera[valid, :2] / camera[valid, 2, None] - observed[valid], axis=1
                )
                * focal
            )
            return int((errors <= 3.0).sum()), float(np.median(errors))

        before_count, before_error = current_support(self.pose, self.landmarks)
        after_count, after_error = current_support(new_poses[-1], proposed)
        diagnostics.update(
            current_support_before=before_count,
            current_support_after=after_count,
            current_median_error_before_px=before_error,
            current_median_error_after_px=after_error,
        )
        if (
            after_count < before_count
            or not np.isfinite(after_error)
            or after_error > before_error * 1.05
        ):
            diagnostics.update(accepted=False, reason="current_frame_support_degraded")
            return diagnostics
        # Keep stored observation/reference poses consistent with the corrected
        # window. Other historical poses stay fixed; there is no marginalization.
        for history in self.landmark_observations.values():
            for index, (pose, observed) in enumerate(history):
                for old, new in zip(poses, new_poses):
                    if np.array_equal(pose, old):
                        history[index] = (new.copy(), observed)
                        break
        reference_points, reference_pose = self.reference
        for old, new in zip(poses, new_poses):
            if np.array_equal(reference_pose, old):
                self.reference = (reference_points, new.copy())
                break
        for frame, pose in zip(window, new_poses):
            frame["pose"] = pose.copy()
        self.landmarks = proposed
        self.pose = new_poses[-1].copy()
        diagnostics["window_pose_corrections"] = [
            {"step": frame["step"], "world_to_camera": pose.tolist()}
            for frame, pose in zip(window, new_poses)
        ]
        return diagnostics

    def process(self, row: TrackFrame) -> FrameResult:
        """Advance the estimator by one tracking frame.

        Steps, in order:
        1. Undistort track pixels into normalized rays.
        2. Reset all map state across stream/segment/size changes or time gaps.
        3. Without a pose, try to bootstrap a new segment from an anchor frame.
           With a pose, estimate this frame's pose by PnP against the map and,
           if it passes the support gates, extend the map.
        4. Optionally run bounded local bundle adjustment.

        Each step writes its diagnostics into one flat `record` dict, in the
        order they are produced; that order is the `poses.jsonl` key order.
        """
        self.last_pnp_snapshot = None
        self.step += 1
        points, camera = self._normalized(row)
        if self._is_discontinuous(row):
            self._clear_map()
            self.anchor = None
            self.reference = None
        record = self._start_record(row, points)
        focal = float((camera[0, 0] + camera[1, 1]) / 2)
        if self.pose is None:
            tracking = self._bootstrap(row, points, focal, record)
        else:
            tracking = self._track_against_map(row, points, camera, focal, record)
        if tracking:
            self._finish_tracking_frame(points, focal, record)
        return FrameResult.from_json(record)

    # -- Segment state ----------------------------------------------------------

    def _is_discontinuous(self, row: TrackFrame) -> bool:
        """True when this frame cannot continue the previous frame's geometry.

        A different stream, tracker segment or tracking size, a repeated or
        regressing timestamp, or a gap over MAX_FRAME_GAP_NS all start afresh.
        """
        timestamp = row.source_time_ns
        discontinuous = row.geometry_key != self.geometry or (
            self.previous_time is not None
            and (
                timestamp <= self.previous_time
                or timestamp - self.previous_time > MAX_FRAME_GAP_NS
            )
        )
        self.geometry = row.geometry_key
        self.previous_time = timestamp
        return discontinuous

    def _clear_map(self) -> None:
        """Drop the pose and every landmark structure of the current segment."""
        self.pose = None
        self.landmarks = {}
        self.bundle_window = []
        self.pending_landmarks = {}
        self.landmark_observations = {}
        self.confirmed_landmarks = set()

    def _start_record(self, row: TrackFrame, points) -> dict:
        """Core fields and map counts before this frame changes anything."""
        return {
            "source_time_ns": row.source_time_ns,
            "received_monotonic_ns": row.received_monotonic_ns,
            "segment": self.segment,
            "status": "initializing",
            "landmarks": len(self.landmarks),
            "visible_features": len(points),
            "landmarks_before": len(self.landmarks),
            "visible_landmarks": len(self.landmarks.keys() & points.keys()),
            "landmarks_no_longer_visible": len(self.landmarks.keys() - points.keys()),
            "landmarks_added": 0,
        }

    # -- Bootstrap ----------------------------------------------------------------

    def _bootstrap(self, row: TrackFrame, points, focal, record) -> bool:
        """Start a segment from the anchor frame and this one; True on success.

        The anchor camera defines the segment's world axes and the bootstrap
        translation defines one arbitrary unit of length.
        """
        np = self.np
        if self.anchor is None:
            self.anchor = (row, points)
            return False
        anchor_row, anchor_points = self.anchor
        shared = sorted(anchor_points.keys() & points.keys())
        if len(shared) < MIN_MAP_CORRESPONDENCES:
            self.anchor = (row, points)
            record["reason"] = "insufficient_bootstrap_tracks"
            return False
        motion = self._estimate_bootstrap_motion(
            anchor_points, points, shared, focal, record
        )
        if motion is None:
            return False
        self.pose = np.column_stack(
            (motion["rotation_b_from_a"], motion["translation_b_from_a_unit"])
        )
        initial = np.column_stack((np.eye(3), np.zeros(3)))
        accepted = [key for key, keep in zip(shared, motion["inlier_mask"]) if keep]
        self._triangulate(
            {key: anchor_points[key] for key in accepted},
            initial,
            {key: points[key] for key in accepted},
            self.pose,
            focal,
            bootstrap=True,
        )
        record["triangulation"] = self.triangulation_diagnostics.copy()
        record["landmarks_added"] = len(self.landmarks)
        if len(self.landmarks) < MIN_MAP_CORRESPONDENCES:
            # Keep the anchor: the next frame may have a better baseline.
            self._clear_map()
            record["reason"] = "insufficient_triangulated_landmarks"
            return False
        self.segment += 1
        self.reference = (points, self.pose.copy())
        record["bootstrap_origin_received_monotonic_ns"] = (
            anchor_row.received_monotonic_ns
        )
        return True

    def _estimate_bootstrap_motion(self, anchor_points, points, shared, focal, record):
        """Essential-matrix motion between anchor and current rays, or None."""
        np = self.np
        a = np.array([anchor_points[key] for key in shared])
        b = np.array([points[key] for key in shared])
        # Rays are already undistorted. Express them with a synthetic focal
        # length only to retain pixel-based RANSAC thresholds in the helper.
        pseudo_camera = np.diag([focal, focal, 1.0])
        try:
            motion = estimate_motion(a * focal, b * focal, pseudo_camera, np.zeros(5))
        except ValueError as error:
            record["reason"] = str(error)
            return None
        if motion["status"] != "accepted_pair":
            record["reason"] = ",".join(motion["rejection_reasons"])
            return None
        return motion

    # -- Tracking against the map ----------------------------------------------

    def _track_against_map(self, row: TrackFrame, points, camera, focal, record):
        """Estimate this frame's pose by PnP; extend the map. False on loss."""
        shared = sorted(self.landmarks.keys() & points.keys())
        record["pnp_candidates"] = len(shared)
        self._snapshot_map_for_diagnostics(shared, points, camera)
        pnp = self._solve_pnp_ransac(shared, points, focal)
        self._record_pnp_support(pnp, shared, focal, record)
        if self.pose_acceptance == "refined" and pnp.ok:
            self._refine_before_acceptance(pnp, shared, focal, record)
        if not self._passes_support_gates(pnp, shared):
            self._record_counterfactual_refinement(pnp, shared, focal, record)
            self._lose_tracking(
                row, points, self._failure_category(pnp, shared), record
            )
            return False
        self._accept_pose(pnp, focal, record)
        self._extend_map(points, focal, record)
        return True

    def _snapshot_map_for_diagnostics(self, shared, points, camera) -> None:
        """Diagnostic-only copy of the SAME map before current-frame updates."""
        np = self.np
        self.last_pnp_snapshot = {
            "ids": shared,
            "objects": np.array([self.landmarks[key] for key in shared]).reshape(-1, 3),
            "normalized": np.array([points[key] for key in shared]).reshape(-1, 2),
            "camera": camera.copy(),
            "candidate_pose": None,
        }

    def _solve_pnp_ransac(self, shared, points, focal) -> _PnpCandidate:
        """EPnP RANSAC on normalized rays with a 3-pixel-equivalent threshold."""
        np, cv = self.np, self.cv
        pnp = _PnpCandidate()
        if len(shared) < MIN_MAP_CORRESPONDENCES:
            return pnp
        pnp.objects = np.array([self.landmarks[key] for key in shared])
        pnp.pixels = np.array([points[key] for key in shared])
        pnp.ok, pnp.rotation, pnp.translation, pnp.inliers = cv.solvePnPRansac(
            pnp.objects,
            pnp.pixels,
            np.eye(3),
            None,
            iterationsCount=100,
            reprojectionError=PNP_INLIER_THRESHOLD_PX / focal,
            confidence=0.999,
            flags=cv.SOLVEPNP_EPNP,
        )
        if pnp.ok:
            self.last_pnp_snapshot["candidate_pose"] = np.column_stack(
                (cv.Rodrigues(pnp.rotation)[0], pnp.translation.ravel())
            )
        return pnp

    def _record_pnp_support(self, pnp, shared, focal, record) -> None:
        """Save solver support before a failure clears the map.

        This distinguishes missing 3D correspondences from geometric rejection,
        and splits candidate errors by landmark age (recent: at most 3 steps).
        """
        np = self.np
        record["pnp_attempted"] = len(shared) >= MIN_MAP_CORRESPONDENCES
        record["pnp_solver_success"] = bool(pnp.ok)
        record["pnp_inliers"] = len(pnp.inliers) if pnp.inliers is not None else 0
        record["pnp_inlier_ratio"] = (
            record["pnp_inliers"] / len(shared) if shared else 0.0
        )
        if not pnp.ok:
            return
        residuals = self._reprojection_errors_px(pnp, focal)
        finite = residuals[np.isfinite(residuals)]
        record["pnp_median_candidate_error_px"] = (
            float(np.median(finite)) if len(finite) else None
        )
        ages = np.array(
            [self.step - self.landmark_birth.get(key, self.step) for key in shared]
        )
        for label, selection in (("recent", ages <= 3), ("older", ages > 3)):
            errors = residuals[selection & np.isfinite(residuals)]
            record[f"pnp_{label}_landmarks"] = int(selection.sum())
            record[f"pnp_{label}_median_error_px"] = (
                float(np.median(errors)) if len(errors) else None
            )
        if pnp.inliers is not None:
            record["pnp_reported_inliers_over_threshold"] = int(
                (residuals[pnp.inliers.ravel()] > PNP_INLIER_THRESHOLD_PX).sum()
            )
            selected_errors = residuals[pnp.inliers.ravel()]
            finite_inliers = selected_errors[np.isfinite(selected_errors)]
            record["pnp_median_inlier_error_px"] = (
                float(np.median(finite_inliers)) if len(finite_inliers) else None
            )

    def _reprojection_errors_px(self, pnp, focal, rotation=None, translation=None):
        """Per-landmark reprojection error, in approximate tracking pixels."""
        np, cv = self.np, self.cv
        rotation = pnp.rotation if rotation is None else rotation
        translation = pnp.translation if translation is None else translation
        projected, _ = cv.projectPoints(
            pnp.objects, rotation, translation, np.eye(3), None
        )
        return np.linalg.norm(projected.reshape(-1, 2) - pnp.pixels, axis=1) * focal

    def _refine_before_acceptance(self, pnp, shared, focal, record) -> None:
        """`refined` pose acceptance: refine first, then recount support."""
        record["ransac_inliers"] = record["pnp_inliers"]
        pnp.ok, pnp.rotation, pnp.translation, pnp.inliers = refine_pose_support(
            pnp.objects, pnp.pixels, pnp.rotation, pnp.translation, pnp.inliers, focal
        )
        record["pnp_solver_success"] = bool(pnp.ok)
        record["pnp_inliers"] = len(pnp.inliers)
        record["pnp_inlier_ratio"] = len(pnp.inliers) / len(shared) if shared else 0.0

    def _passes_support_gates(self, pnp, shared) -> bool:
        """Accept only with enough inliers that are also a large enough share."""
        return not (
            not pnp.ok
            or pnp.inliers is None
            or len(pnp.inliers) < MIN_MAP_CORRESPONDENCES
            or len(pnp.inliers) / len(shared) < MIN_PNP_SUPPORT_RATIO
        )

    def _record_counterfactual_refinement(self, pnp, shared, focal, record) -> None:
        """Audit only: would refining the inlier set pass the SAME gates?

        Runs for baseline acceptance on rejected candidates. Copies keep this
        diagnostic from changing the estimator pose.
        """
        np, cv = self.np, self.cv
        if not (
            self.pose_acceptance == "baseline"
            and pnp.ok
            and pnp.inliers is not None
            and len(pnp.inliers) >= 6
        ):
            return
        audit_ok, audit_rotation, audit_translation = cv.solvePnP(
            pnp.objects[pnp.inliers.ravel()],
            pnp.pixels[pnp.inliers.ravel()],
            np.eye(3),
            None,
            pnp.rotation.copy(),
            pnp.translation.copy(),
            True,
            flags=cv.SOLVEPNP_ITERATIVE,
        )
        if not audit_ok:
            return
        errors = self._reprojection_errors_px(
            pnp, focal, audit_rotation, audit_translation
        )
        audit_camera = (
            pnp.objects @ cv.Rodrigues(audit_rotation)[0].T + audit_translation.ravel()
        )
        support = int(
            (
                np.isfinite(errors)
                & (errors <= PNP_INLIER_THRESHOLD_PX)
                & (audit_camera[:, 2] > 0)
            ).sum()
        )
        record["diagnostic_refined_support"] = support
        record["diagnostic_refinement_passes_same_gates"] = bool(
            support >= MIN_MAP_CORRESPONDENCES
            and support / len(shared) >= MIN_PNP_SUPPORT_RATIO
        )

    def _failure_category(self, pnp, shared) -> str:
        """Name the first gate that failed; this is not a root cause."""
        if len(shared) < MIN_MAP_CORRESPONDENCES:
            return "too_few_visible_landmarks"
        if not pnp.ok or pnp.inliers is None:
            return "pnp_solver_failed"
        if len(pnp.inliers) < MIN_MAP_CORRESPONDENCES:
            return "too_few_pnp_inliers"
        return "low_pnp_inlier_ratio"

    def _lose_tracking(self, row: TrackFrame, points, failure, record) -> None:
        """Clear the segment; this frame becomes the next bootstrap anchor."""
        record.update(
            status="tracking_lost",
            reason="insufficient_pnp_support",
            failure_category=failure,
        )
        self._clear_map()
        self.anchor = (row, points)
        self.reference = None

    def _accept_pose(self, pnp, focal, record) -> None:
        """Set the pose (baseline: refine on inliers first) and audit its fit."""
        np, cv = self.np, self.cv
        accepted = pnp.inliers.ravel()
        if self.pose_acceptance == "baseline":
            ok, pnp.rotation, pnp.translation = cv.solvePnP(
                pnp.objects[accepted],
                pnp.pixels[accepted],
                np.eye(3),
                None,
                pnp.rotation,
                pnp.translation,
                True,
                flags=cv.SOLVEPNP_ITERATIVE,
            )
            if not ok:
                raise ValueError("PnP refinement failed")
        self.pose = np.column_stack(
            (cv.Rodrigues(pnp.rotation)[0], pnp.translation.ravel())
        )
        # Audit the final pose independently: RANSAC's membership mask is
        # not a guarantee that its returned/refined pose fits every inlier.
        refined_errors = self._reprojection_errors_px(pnp, focal)
        inlier_errors = refined_errors[accepted]
        finite_errors = inlier_errors[np.isfinite(inlier_errors)]
        record["refined_median_inlier_error_px"] = (
            float(np.median(finite_errors)) if len(finite_errors) else None
        )
        record["refined_inliers_over_threshold"] = int(
            (inlier_errors > PNP_INLIER_THRESHOLD_PX).sum()
        )
        record["refined_geometric_inliers"] = int(
            (
                np.isfinite(refined_errors)
                & (refined_errors <= PNP_INLIER_THRESHOLD_PX)
            ).sum()
        )
        record["pnp_inliers"] = len(accepted)

    def _extend_map(self, points, focal, record) -> None:
        """Validate/refine landmarks, triangulate new ones, maybe move keyframe."""
        if self.landmark_policy != "baseline":
            record["multiview_update"] = self._update_landmarks(points, focal)
        # A keyframe must move enough to triangulate new points. Retain the
        # previous keyframe through small motions; replace after replenishing.
        reference_points, reference_pose = self.reference
        old_count = len(self.landmarks)
        if self.keyframe_policy == "parallax":
            decision = choose_keyframe(
                reference_points, reference_pose, points, self.pose
            )
            record["keyframe_decision"] = decision
        else:
            decision = {"selected": True, "reason": "baseline_try_every_frame"}
        if decision["selected"]:
            self._triangulate(
                reference_points, reference_pose, points, self.pose, focal
            )
            record["triangulation"] = self.triangulation_diagnostics.copy()
        record["landmarks_added"] = len(self.landmarks) - old_count
        if (
            (self.keyframe_policy == "parallax" and decision["selected"])
            or len(self.landmarks) > old_count
            or (
                self.landmark_policy in ("validation", "multiview")
                and decision["selected"]
                and self.triangulation_diagnostics.get("accepted", 0) > 0
            )
        ):
            self.reference = (points, self.pose.copy())
        # IDs never return after loss; bound memory to currently visible IDs.
        self.landmarks = {
            key: value for key, value in self.landmarks.items() if key in points
        }

    # -- Completing a tracked frame -------------------------------------------

    def _finish_tracking_frame(self, points, focal, record) -> None:
        """Prune birth steps, optionally run local BA, then report the pose."""
        self.landmark_birth = {
            key: self.landmark_birth[key]
            for key in self.landmarks.keys() | self.pending_landmarks.keys()
            if key in self.landmark_birth
        }
        if self.bundle_adjustment:
            record["bundle_adjustment"] = self._adjust_bundle(points, focal)
        center = -self.pose[:, :3].T @ self.pose[:, 3]
        record.update(
            status="tracking",
            segment=self.segment,
            landmarks=len(self.landmarks),
            camera_center_arbitrary=center.tolist(),
            world_to_camera=self.pose.tolist(),
        )
