"""Experimental sparse monocular VO. Each reset has independent arbitrary scale."""

import json
from contextlib import ExitStack
from pathlib import Path

from fusionlab.vision.calibrated_motion import estimate_motion
from fusionlab.vision.tracks import TrackFrame, read_track_frames


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
        np.isfinite(residuals) & (residuals <= 3.0) & (camera[:, 2] > 0)
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

    def process(self, row: TrackFrame):
        np, cv = self.np, self.cv
        self.last_pnp_snapshot = None
        self.step += 1
        points, camera = self._normalized(row)
        geometry = row.geometry_key
        timestamp = row.source_time_ns
        reset = geometry != self.geometry or (
            self.previous_time is not None
            and (
                timestamp <= self.previous_time
                or timestamp - self.previous_time > 500_000_000
            )
        )
        self.geometry = geometry
        self.previous_time = timestamp
        if reset:
            self.anchor = None
            self.reference = None
            self.pose = None
            self.landmarks = {}
            self.bundle_window = []
            self.pending_landmarks = {}
            self.landmark_observations = {}
            self.confirmed_landmarks = set()
        result = {
            "source_time_ns": timestamp,
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
        focal = float((camera[0, 0] + camera[1, 1]) / 2)
        if self.pose is None:
            if self.anchor is None:
                self.anchor = (row, points)
                return result
            anchor_row, anchor_points = self.anchor
            shared = sorted(anchor_points.keys() & points.keys())
            if len(shared) < 20:
                self.anchor = (row, points)
                result["reason"] = "insufficient_bootstrap_tracks"
                return result
            a = np.array([anchor_points[key] for key in shared])
            b = np.array([points[key] for key in shared])
            # Rays are already undistorted. Express them with a synthetic focal
            # length only to retain pixel-based RANSAC thresholds in the helper.
            pseudo_camera = np.diag([focal, focal, 1.0])
            try:
                motion = estimate_motion(
                    a * focal, b * focal, pseudo_camera, np.zeros(5)
                )
            except ValueError as error:
                result["reason"] = str(error)
                return result
            if motion["status"] != "accepted_pair":
                result["reason"] = ",".join(motion["rejection_reasons"])
                return result
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
            result["triangulation"] = self.triangulation_diagnostics.copy()
            result["landmarks_added"] = len(self.landmarks)
            if len(self.landmarks) < 20:
                self.pose = None
                self.landmarks = {}
                self.bundle_window = []
                self.pending_landmarks = {}
                self.landmark_observations = {}
                self.confirmed_landmarks = set()
                result["reason"] = "insufficient_triangulated_landmarks"
                return result
            self.segment += 1
            self.reference = (points, self.pose.copy())
            result["bootstrap_origin_received_monotonic_ns"] = (
                anchor_row.received_monotonic_ns
            )
        else:
            shared = sorted(self.landmarks.keys() & points.keys())
            result["pnp_candidates"] = len(shared)
            # Diagnostic-only copy of the SAME map before current-frame updates.
            self.last_pnp_snapshot = {
                "ids": shared,
                "objects": np.array([self.landmarks[key] for key in shared]).reshape(
                    -1, 3
                ),
                "normalized": np.array([points[key] for key in shared]).reshape(-1, 2),
                "camera": camera.copy(),
                "candidate_pose": None,
            }
            inliers = None
            ok = False
            if len(shared) >= 20:
                objects = np.array([self.landmarks[key] for key in shared])
                pixels = np.array([points[key] for key in shared])
                ok, rotation, translation, inliers = cv.solvePnPRansac(
                    objects,
                    pixels,
                    np.eye(3),
                    None,
                    iterationsCount=100,
                    reprojectionError=3.0 / focal,
                    confidence=0.999,
                    flags=cv.SOLVEPNP_EPNP,
                )
            if ok:
                self.last_pnp_snapshot["candidate_pose"] = np.column_stack(
                    (cv.Rodrigues(rotation)[0], translation.ravel())
                )
            # Preserve solver support before a failure clears the map. This
            # distinguishes missing 3D correspondences from geometric rejection.
            result["pnp_attempted"] = len(shared) >= 20
            result["pnp_solver_success"] = bool(ok)
            result["pnp_inliers"] = len(inliers) if inliers is not None else 0
            result["pnp_inlier_ratio"] = (
                result["pnp_inliers"] / len(shared) if shared else 0.0
            )
            if ok:
                projected, _ = cv.projectPoints(
                    objects, rotation, translation, np.eye(3), None
                )
                residuals = (
                    np.linalg.norm(projected.reshape(-1, 2) - pixels, axis=1) * focal
                )
                finite = residuals[np.isfinite(residuals)]
                result["pnp_median_candidate_error_px"] = (
                    float(np.median(finite)) if len(finite) else None
                )
                ages = np.array(
                    [
                        self.step - self.landmark_birth.get(key, self.step)
                        for key in shared
                    ]
                )
                for label, selection in (("recent", ages <= 3), ("older", ages > 3)):
                    errors = residuals[selection & np.isfinite(residuals)]
                    result[f"pnp_{label}_landmarks"] = int(selection.sum())
                    result[f"pnp_{label}_median_error_px"] = (
                        float(np.median(errors)) if len(errors) else None
                    )
                if inliers is not None:
                    result["pnp_reported_inliers_over_threshold"] = int(
                        (residuals[inliers.ravel()] > 3.0).sum()
                    )
                    selected_errors = residuals[inliers.ravel()]
                    finite_inliers = selected_errors[np.isfinite(selected_errors)]
                    result["pnp_median_inlier_error_px"] = (
                        float(np.median(finite_inliers))
                        if len(finite_inliers)
                        else None
                    )
            if self.pose_acceptance == "refined" and ok:
                result["ransac_inliers"] = result["pnp_inliers"]
                ok, rotation, translation, inliers = refine_pose_support(
                    objects, pixels, rotation, translation, inliers, focal
                )
                result["pnp_solver_success"] = bool(ok)
                result["pnp_inliers"] = len(inliers)
                result["pnp_inlier_ratio"] = (
                    len(inliers) / len(shared) if shared else 0.0
                )
            if (
                not ok
                or inliers is None
                or len(inliers) < 20
                or len(inliers) / len(shared) < 0.5
            ):
                # Counterfactual audit only: would refining the returned inlier
                # set recover support under the SAME 3 px / 20 / 50% gates?
                # Copies keep this diagnostic from changing the estimator pose.
                if (
                    self.pose_acceptance == "baseline"
                    and ok
                    and inliers is not None
                    and len(inliers) >= 6
                ):
                    audit_ok, audit_rotation, audit_translation = cv.solvePnP(
                        objects[inliers.ravel()],
                        pixels[inliers.ravel()],
                        np.eye(3),
                        None,
                        rotation.copy(),
                        translation.copy(),
                        True,
                        flags=cv.SOLVEPNP_ITERATIVE,
                    )
                    if audit_ok:
                        projected, _ = cv.projectPoints(
                            objects, audit_rotation, audit_translation, np.eye(3), None
                        )
                        errors = (
                            np.linalg.norm(projected.reshape(-1, 2) - pixels, axis=1)
                            * focal
                        )
                        audit_camera = (
                            objects @ cv.Rodrigues(audit_rotation)[0].T
                            + audit_translation.ravel()
                        )
                        support = int(
                            (
                                np.isfinite(errors)
                                & (errors <= 3.0)
                                & (audit_camera[:, 2] > 0)
                            ).sum()
                        )
                        result["diagnostic_refined_support"] = support
                        result["diagnostic_refinement_passes_same_gates"] = bool(
                            support >= 20 and support / len(shared) >= 0.5
                        )
                if len(shared) < 20:
                    failure = "too_few_visible_landmarks"
                elif not ok or inliers is None:
                    failure = "pnp_solver_failed"
                elif len(inliers) < 20:
                    failure = "too_few_pnp_inliers"
                else:
                    failure = "low_pnp_inlier_ratio"
                result.update(
                    status="tracking_lost",
                    reason="insufficient_pnp_support",
                    failure_category=failure,
                )
                self.pose = None
                self.landmarks = {}
                self.bundle_window = []
                self.pending_landmarks = {}
                self.landmark_observations = {}
                self.confirmed_landmarks = set()
                self.anchor = (row, points)
                self.reference = None
                return result
            accepted = inliers.ravel()
            if self.pose_acceptance == "baseline":
                ok, rotation, translation = cv.solvePnP(
                    objects[accepted],
                    pixels[accepted],
                    np.eye(3),
                    None,
                    rotation,
                    translation,
                    True,
                    flags=cv.SOLVEPNP_ITERATIVE,
                )
                if not ok:
                    raise ValueError("PnP refinement failed")
            self.pose = np.column_stack(
                (cv.Rodrigues(rotation)[0], translation.ravel())
            )
            # Audit the final pose independently: RANSAC's membership mask is
            # not a guarantee that its returned/refined pose fits every inlier.
            projected, _ = cv.projectPoints(
                objects, rotation, translation, np.eye(3), None
            )
            refined_errors = (
                np.linalg.norm(projected.reshape(-1, 2) - pixels, axis=1) * focal
            )
            inlier_errors = refined_errors[accepted]
            finite_errors = inlier_errors[np.isfinite(inlier_errors)]
            result["refined_median_inlier_error_px"] = (
                float(np.median(finite_errors)) if len(finite_errors) else None
            )
            result["refined_inliers_over_threshold"] = int((inlier_errors > 3.0).sum())
            result["refined_geometric_inliers"] = int(
                (np.isfinite(refined_errors) & (refined_errors <= 3.0)).sum()
            )
            result["pnp_inliers"] = len(accepted)
            if self.landmark_policy != "baseline":
                result["multiview_update"] = self._update_landmarks(points, focal)
            # A keyframe must move enough to triangulate new points. Retain the
            # previous keyframe through small motions; replace after replenishing.
            reference_points, reference_pose = self.reference
            old_count = len(self.landmarks)
            if self.keyframe_policy == "parallax":
                decision = choose_keyframe(
                    reference_points, reference_pose, points, self.pose
                )
                result["keyframe_decision"] = decision
            else:
                decision = {"selected": True, "reason": "baseline_try_every_frame"}
            if decision["selected"]:
                self._triangulate(
                    reference_points, reference_pose, points, self.pose, focal
                )
                result["triangulation"] = self.triangulation_diagnostics.copy()
            result["landmarks_added"] = len(self.landmarks) - old_count
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
        self.landmark_birth = {
            key: self.landmark_birth[key]
            for key in self.landmarks.keys() | self.pending_landmarks.keys()
            if key in self.landmark_birth
        }
        if self.bundle_adjustment:
            result["bundle_adjustment"] = self._adjust_bundle(points, focal)
        center = -self.pose[:, :3].T @ self.pose[:, 3]
        result.update(
            status="tracking",
            segment=self.segment,
            landmarks=len(self.landmarks),
            camera_center_arbitrary=center.tolist(),
            world_to_camera=self.pose.tolist(),
        )
        return result


class OdometryReplay:
    """Synchronized video, status, and a resettable segment-local path display.

    The display frame is deliberately a container for the currently active
    segment, not a shared physical frame linking the independent trajectories.
    """

    def __init__(self, viewer, tracks):
        from fusion_viewer import RunSpec, FrameSpec

        self.viewer = viewer
        self.tracks_root = tracks.parent.resolve()
        self.origin = None
        self.active_segment = None
        self.path = []
        self.run = viewer.register_run(
            RunSpec(
                "monocular_vo",
                "Experimental monocular VO; independent arbitrary scale per segment",
            )
        )
        viewer.register_frame(
            self.run,
            FrameSpec(
                "active_segment_xz",
                "Display container: current segment initial-camera XZ; cleared on loss/reset; no alignment between segments",
                units="arbitrary",
            ),
        )

    def publish(self, row: TrackFrame, result, index):
        import cv2
        import numpy as np
        from fusion_viewer import SampleTime, Trajectory2D, ScalarSample, ImageSample

        if self.origin is None:
            self.origin = row.received_monotonic_ns
        stamp = SampleTime(
            "vo_replay", "receive", row.received_monotonic_ns - self.origin
        )
        status = result["status"]
        segment = result["segment"]
        if status == "tracking":
            if segment != self.active_segment:
                self.path = [(0.0, 0.0)]
                self.active_segment = segment
            center = result["camera_center_arbitrary"]
            self.path.append((center[0], center[2]))
        else:
            # Do not leave the last successful pose looking current during loss.
            self.path = []
            self.active_segment = None
        self.viewer.publish(
            self.run,
            Trajectory2D(
                entity_id="active_path",
                time=stamp,
                frame_id="active_segment_xz",
                points=tuple(self.path),
            ),
        )
        for key, value, units in (
            (
                "status",
                {"initializing": 0, "tracking": 1, "tracking_lost": 2}[status],
                "state",
            ),
            ("segment", segment, "segment"),
            ("landmarks", result.get("landmarks", 0), "landmarks"),
            ("pnp_inliers", result.get("pnp_inliers", 0), "inliers"),
            ("visible_landmarks", result.get("visible_landmarks", 0), "landmarks"),
            ("landmarks_added", result.get("landmarks_added", 0), "landmarks"),
            (
                "landmarks_no_longer_visible",
                result.get("landmarks_no_longer_visible", 0),
                "landmarks",
            ),
        ):
            self.viewer.publish(
                self.run,
                ScalarSample(
                    entity_id=f"signals/vo/{key}", time=stamp, value=value, units=units
                ),
            )
        # Missing measurements stay absent; zero would imply a perfect fit.
        for key in (
            "pnp_median_candidate_error_px",
            "pnp_median_inlier_error_px",
            "refined_median_inlier_error_px",
        ):
            value = result.get(key)
            if value is not None:
                self.viewer.publish(
                    self.run,
                    ScalarSample(
                        entity_id=f"signals/vo/{key}",
                        time=stamp,
                        value=value,
                        units="px",
                    ),
                )
        overlay = row.overlay_path
        if not overlay:
            raise ValueError(
                "Tracking row has no overlay_path; regenerate feature tracking overlays"
            )
        path = (self.tracks_root / overlay).resolve()
        if not path.is_relative_to(self.tracks_root):
            raise ValueError("Overlay path escapes tracking directory")
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Missing or unreadable tracking overlay: {path}")
        # The tracking overlay already has the selected display rotation applied.
        # Add a separate banner so state and segment are readable at every frame.
        banner = np.zeros((90, image.shape[1], 3), dtype=np.uint8)
        color = (80, 220, 80) if status == "tracking" else (80, 150, 255)
        cv2.putText(
            banner,
            f"Frame {index} | {status}",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )
        cv2.putText(
            banner,
            f"Segment {segment} | {result.get('landmarks',0)} landmarks",
            (8, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            banner,
            "Path resets on loss; arbitrary scale",
            (8, 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 255),
            1,
        )
        image = np.vstack((banner, image))
        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            raise ValueError("Replay video encoding failed")
        self.viewer.publish(
            self.run,
            ImageSample(
                entity_id="rgb/tracking",
                time=stamp,
                width=image.shape[1],
                height=image.shape[0],
                encoding="image/jpeg",
                data=encoded.tobytes(),
            ),
        )


def run_odometry(
    tracks: Path,
    calibration: Path,
    output: Path,
    *,
    stride=5,
    max_frames=None,
    viewer=None,
    viewer_factory=None,
    pose_acceptance="baseline",
    keyframe_policy="baseline",
    landmark_policy="baseline",
    bundle_adjustment=False,
    seed=0,
):
    """Run sequential pose estimation; save failures as well as successful poses."""
    if stride < 1 or (max_frames is not None and max_frames < 1):
        raise ValueError("Counts must be positive")
    tracker = MonocularOdometry(
        json.loads(calibration.read_text()),
        pose_acceptance=pose_acceptance,
        keyframe_policy=keyframe_policy,
        landmark_policy=landmark_policy,
        bundle_adjustment=bundle_adjustment,
    )
    tracker.cv.setRNGSeed(seed)
    output.mkdir(parents=True, exist_ok=False)
    with ExitStack() as resources:
        if viewer_factory is not None:
            viewer = resources.enter_context(viewer_factory())
        counts = {}
        processed = 0
        replay = OdometryReplay(viewer, tracks) if viewer else None
        with (output / "poses.jsonl").open("x") as out:
            for index, row in read_track_frames(tracks):
                if index % stride:
                    continue
                result = tracker.process(row)
                result["frame_index"] = index
                out.write(json.dumps(result, allow_nan=False) + "\n")
                counts[result["status"]] = counts.get(result["status"], 0) + 1
                processed += 1
                if replay is not None:
                    replay.publish(row, result, index)
                if max_frames is not None and processed >= max_frames:
                    break
        if not processed:
            raise ValueError("No tracking frames")
        diagnosis = write_failure_report(output)
        write_geometry_report(output)
        report = {
            "failure_categories": diagnosis["failure_categories"],
            "pose_acceptance": pose_acceptance,
            "keyframe_policy": keyframe_policy,
            "landmark_policy": landmark_policy,
            "bundle_adjustment": bundle_adjustment,
            "ransac_seed": seed,
            "processed_frames": processed,
            "stride": stride,
            "status_counts": counts,
            "segments": tracker.segment,
            "tracks": str(tracks.resolve()),
            "calibration": str(calibration.resolve()),
            "calibration_status": tracker.calibration.get("status"),
            "scope": (
                "Experimental monocular VO, arbitrary scale per segment; "
                + (
                    "bounded local bundle adjustment enabled; "
                    if bundle_adjustment
                    else "no bundle adjustment; "
                )
                + "no loop closure, gravity alignment or metric scale."
            ),
            "camera_configuration": "Requires matching camera, zoom and image processing; resolution alone cannot verify compatibility.",
        }
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        return report


def write_failure_report(output: Path):
    """Summarize observed failure conditions without asserting visual causes."""
    from collections import Counter

    with (output / "poses.jsonl").open() as file:
        rows = [json.loads(line) for line in file]
    origin = rows[0]["received_monotonic_ns"]
    failures = []
    for index, row in enumerate(rows):
        if row["status"] != "tracking_lost":
            continue
        history = [
            previous
            for previous in rows[max(0, index - 5) : index]
            if previous["segment"] == row["segment"]
            and previous["status"] == "tracking"
        ]
        failures.append(
            {
                "frame_index": row["frame_index"],
                "receive_elapsed_s": (row["received_monotonic_ns"] - origin) / 1e9,
                "segment": row["segment"],
                "category": row["failure_category"],
                "landmarks_before": row["landmarks_before"],
                "visible_landmarks": row["visible_landmarks"],
                "landmarks_no_longer_visible": row["landmarks_no_longer_visible"],
                "pnp_inliers": row.get("pnp_inliers", 0),
                "pnp_inlier_ratio": row.get("pnp_inlier_ratio", 0),
                "pnp_median_candidate_error_px": row.get(
                    "pnp_median_candidate_error_px"
                ),
                "preceding_frames": [
                    {
                        key: previous.get(key)
                        for key in (
                            "frame_index",
                            "landmarks",
                            "visible_landmarks",
                            "landmarks_added",
                            "pnp_inliers",
                            "pnp_median_inlier_error_px",
                        )
                    }
                    for previous in history
                ],
            }
        )
    report = {
        "failure_count": len(failures),
        "failure_categories": dict(Counter(f["category"] for f in failures)),
        "failures": failures,
        "interpretation": "Categories describe failed gates, not root causes. Pixel errors approximate normalized reprojection error times mean tracking focal length. Inspect video before attributing blur, calibration mismatch or bad landmarks.",
    }
    (output / "failures.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    text = [
        "# VO tracking failure report",
        "",
        report["interpretation"],
        "",
        "| Frame | Receive time (s) | Segment | Failed condition | Visible / stored landmarks | PnP inliers |",
        "| --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for failure in failures:
        text.append(
            f"| {failure['frame_index']} | {failure['receive_elapsed_s']:.2f} | {failure['segment']} | {failure['category']} | {failure['visible_landmarks']} / {failure['landmarks_before']} | {failure['pnp_inliers']} |"
        )
    text.extend(
        [
            "",
            "See `failures.json` for up to five preceding tracked frames per failure.",
            "Video uses these receive times; frame indices refer to rows in the tracking file.",
        ]
    )
    (output / "failures.md").write_text("\n".join(text) + "\n")
    return report


def write_geometry_report(output: Path):
    """Summarize triangulation sensitivity and counterfactual pose refinement."""
    from statistics import median

    with (output / "poses.jsonl").open() as file:
        rows = [json.loads(line) for line in file]
    angles = [
        angle
        for row in rows
        for angle in row.get("triangulation", {}).get("accepted_parallax_deg", [])
    ]
    rejected = [row for row in rows if row["status"] == "tracking_lost"]
    salvage = [
        row["frame_index"]
        for row in rejected
        if row.get("diagnostic_refinement_passes_same_gates", False)
    ]
    refined = [
        row for row in rows if row.get("refined_median_inlier_error_px") is not None
    ]
    report = {
        "triangulated_landmark_observations": len(angles),
        "median_admitted_parallax_deg": median(angles) if angles else None,
        "fraction_admitted_below_2_deg": (
            sum(angle < 2 for angle in angles) / len(angles) if angles else None
        ),
        "failure_count": len(rejected),
        "counterfactual_same_gate_recoveries": len(salvage),
        "counterfactual_frame_indices": salvage,
        "median_pre_refinement_inlier_error_px": (
            median(r["pnp_median_inlier_error_px"] for r in refined)
            if refined
            else None
        ),
        "median_post_refinement_inlier_error_px": (
            median(r["refined_median_inlier_error_px"] for r in refined)
            if refined
            else None
        ),
        "interpretation": "Diagnostic only. Recovered failed frames are independent counterfactuals on the original map, not a rerun with fewer resets. Counts do not predict the number of segments after a fix. Narrow-angle triangulation is sensitive to errors but this audit does not measure true landmark depth accuracy.",
    }
    (output / "geometry_audit.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    text = [
        "# Landmark and pose-refinement audit",
        "",
        f"Rejected frames passing the same gates after diagnostic refinement: **{len(salvage)} / {len(rejected)}**.",
        f"Admitted landmark observations: **{len(angles)}**.",
    ]
    if angles:
        text.extend(
            [
                f"Median admitted triangulation angle: **{median(angles):.2f}°**.",
                f"Fraction below 2°: **{100*sum(a<2 for a in angles)/len(angles):.1f}%**.",
            ]
        )
    text.extend(
        [
            "",
            report["interpretation"],
            "",
            "Next controlled experiments: refine and recompute geometric support before acceptance; evaluate longer-baseline keyframes and multiview landmark refinement. Do not merely relax thresholds or claim these changes will recover the whole walk.",
        ]
    )
    (output / "geometry_audit.md").write_text("\n".join(text) + "\n")
    return report
