"""Experimental sparse monocular VO. Each reset has independent arbitrary scale.

Pipeline per frame: undistort track pixels to rays; bootstrap a segment from
two views (essential matrix, then triangulation) or estimate the pose by PnP
against the segment's landmark map; extend the map by triangulating against a
reference keyframe; optionally run a bounded local bundle adjustment. Policy
objects (see policies.py) decide the experimental variants.

Frames and units: poses are 3×4 world-to-camera matrices [R | t] with
X_camera = R X_world + t. The world frame is the segment's first camera and the
unit of length is the bootstrap baseline, so values from different segments
must never be compared or joined. Rays are normalized, undistorted image
coordinates; pixel thresholds use the mean tracking focal length.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any

from fusionlab.vision.calibrated_motion import estimate_motion
from fusionlab.vision.odometry.frame_result import FrameResult
from fusionlab.vision.odometry.pnp import (
    PNP_INLIER_THRESHOLD_PX,
    PnpCandidate,
    positive_depth_support,
    refine_on_inliers,
    refine_pose_support,
    reprojection_errors_px,
    solve_pnp_ransac,
)
from fusionlab.vision.odometry.policies import (
    KEYFRAME_POLICIES,
    LANDMARK_POLICIES,
    POSE_ACCEPTANCE,
    lookup,
)
from fusionlab.vision.odometry.triangulation import (
    TriangulationCounts,
    camera_center,
    candidate_quality,
    triangulate_rays,
)
from fusionlab.vision.rays import track_rays
from fusionlab.vision.tracks import TrackFrame

if TYPE_CHECKING:
    from fusionlab._typing import Array, Rays
    from fusionlab.vision.odometry.local_bundle_adjustment import WindowObservation

# Frames further apart than this (source clock) start a new segment.
MAX_FRAME_GAP_NS = 500_000_000
# Minimum bootstrap tracks, triangulated landmarks, PnP correspondences and inliers.
MIN_MAP_CORRESPONDENCES = 20
# PnP inliers must also be at least this share of visible map landmarks.
MIN_PNP_SUPPORT_RATIO = 0.5
# Landmarks born at most this many steps ago count as "recent" in PnP reports.
RECENT_LANDMARK_STEPS = 3

# Triangulation: worst-view reprojection error and minimum ray angle.
TRIANGULATION_MAX_ERROR_PX = 2.0
MIN_TRIANGULATION_PARALLAX_DEG = 1.0
# Landmark validation/refinement: every kept observation must reproject this well.
LANDMARK_MAX_ERROR_PX = 3.0
# Observation history per landmark: the first view plus the latest views.
MAX_LANDMARK_HISTORY = 5

# Local bundle adjustment window and schedule.
BA_WINDOW_FRAMES = 5
BA_MIN_WINDOW_FRAMES = 4
BA_INTERVAL_STEPS = 3
BA_MIN_LANDMARK_VIEWS = 3
BA_MAX_LANDMARKS = 80
BA_MIN_OBSERVATIONS_PER_POSE = 10
# Reject an update if the current-frame median error grows by more than 5%.
BA_MAX_MEDIAN_ERROR_GROWTH = 1.05
# Points at or nearer than this camera depth count as unsupported.
BA_SUPPORT_MIN_DEPTH = 1e-6

# One landmark observation: the observing world-to-camera pose and its ray.
Observation = tuple["Array", "Array"]


@dataclass(frozen=True)
class _Anchor:
    """The frame a segment bootstraps from; its camera is the world origin."""

    row: TrackFrame
    points: Rays


@dataclass(frozen=True)
class _Keyframe:
    """The reference view that new landmarks are triangulated against."""

    points: Rays
    pose: Array  # 3×4 world-to-camera


@dataclass
class _BundleFrame:
    """One pose of the local bundle-adjustment window and its rays."""

    pose: Array  # 3×4 world-to-camera; replaced when BA is adopted
    points: Rays
    step: int | None  # estimator step; None for the segment's origin frame


@dataclass
class _LandmarkUpdateCounts:
    """Third-view validation and refinement outcomes for one frame."""

    confirmed: int = 0
    rejected_third_view: int = 0
    refined: int = 0
    pending: int = 0


@dataclass(frozen=True)
class PnpSnapshot:
    """Diagnostic-only copy of the map PnP saw, before this frame changed it.

    Diagnostics replay the estimator and score other pose hypotheses on exactly
    these correspondences; nothing here feeds estimator decisions.
    """

    ids: list[int]  # track IDs of the visible mapped landmarks
    objects: Array  # (N, 3) landmarks, segment world frame
    normalized: Array  # (N, 2) their rays in this frame
    camera: Array  # 3×3 tracking-pixel camera matrix
    candidate_pose: Array | None  # RANSAC world-to-camera, before gates


def _bounded_history(
    history: list[Observation], observation: Observation
) -> list[Observation]:
    """Append an observation, keeping the first view and the latest ones."""
    history = history + [observation]
    if len(history) > MAX_LANDMARK_HISTORY:
        history = [history[0]] + history[-(MAX_LANDMARK_HISTORY - 1) :]
    return history


def _window_observations(
    window: list[_BundleFrame], identifiers: list[int]
) -> list[WindowObservation]:
    """(frame index, landmark index, ray) for every landmark seen in each frame."""
    from fusionlab.vision.odometry.local_bundle_adjustment import WindowObservation

    return [
        WindowObservation(camera, index, frame.points[key])
        for camera, frame in enumerate(window)
        for index, key in enumerate(identifiers)
        if key in frame.points
    ]


def _origin_pose() -> Array:
    """Identity world-to-camera pose of a segment's first camera."""
    import numpy as np

    return np.column_stack((np.eye(3), np.zeros(3)))


def _median_or_none(values: Array) -> float | None:
    """Median of the finite values, or None when there are none."""
    import numpy as np

    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if len(finite) else None


class MonocularOdometry:
    """Triangulate persistent feature IDs, then recover world-to-camera poses.

    A segment's first camera defines the world axes. Its initialization baseline
    is one arbitrary unit. No unit translations are accumulated after bootstrap:
    subsequent translations come from PnP against the existing 3D landmarks.

    Map state (all keyed by track ID, cleared with the segment):
    `landmarks` are mapped points used by PnP; `pending_landmarks` await
    third-view validation; `landmark_observations` hold bounded (pose, ray)
    histories; `confirmed_landmarks` passed a later view; `bundle_window` is
    the local BA window.
    """

    def __init__(
        self,
        calibration: dict[str, Any],
        *,
        pose_acceptance: str = "baseline",
        keyframe_policy: str = "baseline",
        landmark_policy: str = "baseline",
        bundle_adjustment: bool = False,
    ) -> None:
        # Policies are chosen by name; see policies.py for what each one does.
        self.pose_acceptance = lookup(
            POSE_ACCEPTANCE, pose_acceptance, "pose acceptance method"
        )
        self.keyframe_policy = lookup(
            KEYFRAME_POLICIES, keyframe_policy, "keyframe policy"
        )
        self.landmark_policy = lookup(
            LANDMARK_POLICIES, landmark_policy, "landmark policy"
        )
        self.bundle_adjustment = bundle_adjustment
        # OpenCV and NumPy are imported where used, keeping them optional at
        # package import; fail here, at construction, if they are missing.
        import cv2  # noqa: F401
        import numpy  # noqa: F401

        self.calibration = calibration
        # Segment state: bootstrap origin, triangulation reference, current pose.
        self.anchor: _Anchor | None = None
        self.reference: _Keyframe | None = None
        self.pose: Array | None = None
        self.segment = 0
        # Map state, cleared by `_clear_map`.
        self.landmarks: dict[int, Array] = {}
        self.pending_landmarks: dict[int, Array] = {}
        self.landmark_observations: dict[int, list[Observation]] = {}
        self.confirmed_landmarks: set[int] = set()
        self.bundle_window: list[_BundleFrame] = []
        # Step at which each landmark was triangulated (for age reports).
        self.landmark_birth: dict[int, int] = {}
        # Continuity checks: last tracking geometry and source timestamp.
        self.geometry: tuple[Any, ...] | None = None
        self.previous_time: int | None = None
        self.step = 0
        # Diagnostics of the latest triangulation pass and PnP attempt.
        self.triangulation_diagnostics = TriangulationCounts()
        self.last_pnp_snapshot: PnpSnapshot | None = None

    @property
    def _tracked_pose(self) -> Array:
        """The current pose; only valid while a segment is tracking."""
        if self.pose is None:
            raise RuntimeError("No current pose: the estimator is not tracking")
        return self.pose

    @property
    def _reference_keyframe(self) -> _Keyframe:
        """The triangulation reference; set whenever a segment is tracking."""
        if self.reference is None:
            raise RuntimeError("No reference keyframe: the estimator is not tracking")
        return self.reference

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
        points, camera = track_rays(row, self.calibration)
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

    def _start_record(self, row: TrackFrame, points: Rays) -> dict[str, Any]:
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

    def _bootstrap(
        self, row: TrackFrame, points: Rays, focal: float, record: dict[str, Any]
    ) -> bool:
        """Start a segment from the anchor frame and this one; True on success.

        The anchor camera defines the segment's world axes and the bootstrap
        translation defines one arbitrary unit of length. Without enough shared
        tracks this frame becomes the new anchor; after a failed motion or
        triangulation the anchor is kept for a wider baseline next frame.
        """
        import numpy as np

        if self.anchor is None:
            self.anchor = _Anchor(row, points)
            return False
        anchor = self.anchor
        shared = sorted(anchor.points.keys() & points.keys())
        if len(shared) < MIN_MAP_CORRESPONDENCES:
            self.anchor = _Anchor(row, points)
            record["reason"] = "insufficient_bootstrap_tracks"
            return False
        motion = self._estimate_bootstrap_motion(
            anchor.points, points, shared, focal, record
        )
        if motion is None:
            return False
        pose = np.column_stack(
            (motion["rotation_b_from_a"], motion["translation_b_from_a_unit"])
        )
        self.pose = pose
        accepted = [key for key, keep in zip(shared, motion["inlier_mask"]) if keep]
        self._triangulate(
            {key: anchor.points[key] for key in accepted},
            _origin_pose(),
            {key: points[key] for key in accepted},
            pose,
            focal,
            bootstrap=True,
        )
        record["triangulation"] = asdict(self.triangulation_diagnostics)
        record["landmarks_added"] = len(self.landmarks)
        if len(self.landmarks) < MIN_MAP_CORRESPONDENCES:
            self._clear_map()
            record["reason"] = "insufficient_triangulated_landmarks"
            return False
        self.segment += 1
        self.reference = _Keyframe(points, pose.copy())
        record["bootstrap_origin_received_monotonic_ns"] = (
            anchor.row.received_monotonic_ns
        )
        return True

    def _estimate_bootstrap_motion(
        self,
        anchor_points: Rays,
        points: Rays,
        shared: list[int],
        focal: float,
        record: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Essential-matrix motion between anchor and current rays, or None.

        On rejection the reason is written to `record`.
        """
        import numpy as np

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

    # -- Triangulation --------------------------------------------------------

    def _triangulate(
        self,
        points_a: Rays,
        pose_a: Array,
        points_b: Rays,
        pose_b: Array,
        focal: float,
        *,
        bootstrap: bool = False,
    ) -> None:
        """Triangulate new track IDs seen in both views; admit the good ones.

        A candidate is admitted when it has positive depth in both cameras,
        reprojects within TRIANGULATION_MAX_ERROR_PX in both views, and its
        rays meet at MIN_TRIANGULATION_PARALLAX_DEG or more. Counts per outcome
        are kept in `triangulation_diagnostics`.
        """
        import numpy as np

        identifiers = self._triangulation_candidates(points_a, points_b)
        counts = TriangulationCounts(candidates=len(identifiers))
        self.triangulation_diagnostics = counts
        if not identifiers:
            return
        a = np.array([points_a[key] for key in identifiers])
        b = np.array([points_b[key] for key in identifiers])
        points, camera_a, camera_b, valid = triangulate_rays(a, b, pose_a, pose_b)
        center_a = camera_center(pose_a)
        center_b = camera_center(pose_b)
        for index, key in enumerate(identifiers):
            if not valid[index]:
                counts.invalid_depth_or_homogeneous += 1
                continue
            error_px, parallax = candidate_quality(
                points[index],
                camera_a[index],
                camera_b[index],
                a[index],
                b[index],
                center_a,
                center_b,
                focal,
            )
            if (
                error_px <= TRIANGULATION_MAX_ERROR_PX
                and parallax >= MIN_TRIANGULATION_PARALLAX_DEG
            ):
                observations = ((pose_a, a[index]), (pose_b, b[index]))
                self._admit_landmark(key, points[index], observations, bootstrap)
                counts.accepted += 1
                counts.accepted_parallax_deg.append(parallax)
                counts.accepted_max_reprojection_px.append(error_px)
            elif error_px > TRIANGULATION_MAX_ERROR_PX:
                counts.high_reprojection_error += 1
            else:
                counts.low_parallax += 1

    def _triangulation_candidates(self, points_a: Rays, points_b: Rays) -> list[int]:
        """Track IDs in both views that are neither mapped nor pending."""
        return sorted(
            (points_a.keys() & points_b.keys())
            - self.landmarks.keys()
            - self.pending_landmarks.keys()
        )

    def _admit_landmark(
        self,
        key: int,
        point: Array,
        observations: tuple[Observation, ...],
        bootstrap: bool,
    ) -> None:
        """Store an accepted point as mapped or, under validation, as pending.

        Bootstrap points always enter the map: they must seed the first PnP.
        """
        if self.landmark_policy.keeps_observations:
            self.landmark_observations[key] = [
                (pose.copy(), ray.copy()) for pose, ray in observations
            ]
        if self.landmark_policy.validate_third_view and not bootstrap:
            self.pending_landmarks[key] = point
        else:
            self.landmarks[key] = point
        self.landmark_birth[key] = self.step

    # -- Tracking against the map ----------------------------------------------

    def _track_against_map(
        self,
        row: TrackFrame,
        points: Rays,
        camera: Array,
        focal: float,
        record: dict[str, Any],
    ) -> bool:
        """Estimate this frame's pose by PnP; extend the map. False on loss."""
        import numpy as np

        shared = sorted(self.landmarks.keys() & points.keys())
        record["pnp_candidates"] = len(shared)
        objects = np.array([self.landmarks[key] for key in shared]).reshape(-1, 3)
        rays = np.array([points[key] for key in shared]).reshape(-1, 2)
        pnp = None
        if len(shared) >= MIN_MAP_CORRESPONDENCES:
            pnp = solve_pnp_ransac(objects, rays, focal)
        self.last_pnp_snapshot = PnpSnapshot(
            shared,
            objects,
            rays,
            camera.copy(),
            pnp.world_to_camera() if pnp is not None else None,
        )
        self._record_pnp_support(pnp, shared, focal, record)
        if self.pose_acceptance.refine_before_gates and pnp is not None:
            pnp = self._refine_before_acceptance(pnp, shared, focal, record)
        if pnp is None or not self._passes_support_gates(pnp, shared):
            if pnp is not None and not self.pose_acceptance.refine_before_gates:
                self._record_counterfactual_refinement(pnp, shared, focal, record)
            self._lose_tracking(
                row, points, self._failure_category(pnp, shared), record
            )
            return False
        self._accept_pose(pnp, focal, record)
        self._extend_map(points, focal, record)
        return True

    def _record_pnp_support(
        self,
        pnp: PnpCandidate | None,
        shared: list[int],
        focal: float,
        record: dict[str, Any],
    ) -> None:
        """Save solver support before a failure clears the map.

        This distinguishes missing 3D correspondences from geometric rejection,
        and splits candidate errors by landmark age.
        """
        import numpy as np

        record["pnp_attempted"] = len(shared) >= MIN_MAP_CORRESPONDENCES
        record["pnp_solver_success"] = pnp is not None
        record["pnp_inliers"] = len(pnp.inliers) if pnp is not None else 0
        record["pnp_inlier_ratio"] = (
            record["pnp_inliers"] / len(shared) if shared else 0.0
        )
        if pnp is None:
            return
        residuals = pnp.reprojection_errors_px(focal)
        record["pnp_median_candidate_error_px"] = _median_or_none(residuals)
        ages = np.array(
            [self.step - self.landmark_birth.get(key, self.step) for key in shared]
        )
        recent = ages <= RECENT_LANDMARK_STEPS
        for label, selection in (("recent", recent), ("older", ~recent)):
            record[f"pnp_{label}_landmarks"] = int(selection.sum())
            record[f"pnp_{label}_median_error_px"] = _median_or_none(
                residuals[selection]
            )
        inlier_errors = residuals[pnp.inliers.ravel()]
        record["pnp_reported_inliers_over_threshold"] = int(
            (inlier_errors > PNP_INLIER_THRESHOLD_PX).sum()
        )
        record["pnp_median_inlier_error_px"] = _median_or_none(inlier_errors)

    def _refine_before_acceptance(
        self,
        pnp: PnpCandidate,
        shared: list[int],
        focal: float,
        record: dict[str, Any],
    ) -> PnpCandidate | None:
        """`refined` pose acceptance: refine first, then recount support.

        Returns the refined candidate, or None if refinement failed.
        """
        record["ransac_inliers"] = record["pnp_inliers"]
        ok, rotation, translation, support = refine_pose_support(
            pnp.objects, pnp.rays, pnp.rotation, pnp.translation, pnp.inliers, focal
        )
        record["pnp_solver_success"] = ok
        record["pnp_inliers"] = len(support)
        record["pnp_inlier_ratio"] = len(support) / len(shared) if shared else 0.0
        if not ok:
            return None
        return replace(pnp, rotation=rotation, translation=translation, inliers=support)

    def _passes_support_gates(self, pnp: PnpCandidate, shared: list[int]) -> bool:
        """Accept only with enough inliers that are also a large enough share."""
        return (
            len(pnp.inliers) >= MIN_MAP_CORRESPONDENCES
            and len(pnp.inliers) / len(shared) >= MIN_PNP_SUPPORT_RATIO
        )

    def _record_counterfactual_refinement(
        self,
        pnp: PnpCandidate,
        shared: list[int],
        focal: float,
        record: dict[str, Any],
    ) -> None:
        """Audit only: would refining the inlier set pass the SAME gates?

        Runs for baseline acceptance on rejected candidates. The refinement
        copies its seed, so this diagnostic cannot change the estimator pose.
        Unlike `refine_pose_support`, a nonfinite refined pose still counts
        (with zero support) rather than being skipped.
        """
        if len(pnp.inliers) < 6:
            return
        ok, rotation, translation = refine_on_inliers(
            pnp.objects, pnp.rays, pnp.rotation, pnp.translation, pnp.inliers
        )
        if not ok:
            return
        support = len(
            positive_depth_support(pnp.objects, pnp.rays, rotation, translation, focal)
        )
        record["diagnostic_refined_support"] = support
        record["diagnostic_refinement_passes_same_gates"] = bool(
            support >= MIN_MAP_CORRESPONDENCES
            and support / len(shared) >= MIN_PNP_SUPPORT_RATIO
        )

    def _failure_category(self, pnp: PnpCandidate | None, shared: list[int]) -> str:
        """Name the first gate that failed; this is not a root cause."""
        if len(shared) < MIN_MAP_CORRESPONDENCES:
            return "too_few_visible_landmarks"
        if pnp is None:
            return "pnp_solver_failed"
        if len(pnp.inliers) < MIN_MAP_CORRESPONDENCES:
            return "too_few_pnp_inliers"
        return "low_pnp_inlier_ratio"

    def _lose_tracking(
        self, row: TrackFrame, points: Rays, failure: str, record: dict[str, Any]
    ) -> None:
        """Clear the segment; this frame becomes the next bootstrap anchor."""
        record.update(
            status="tracking_lost",
            reason="insufficient_pnp_support",
            failure_category=failure,
        )
        self._clear_map()
        self.anchor = _Anchor(row, points)
        self.reference = None

    def _accept_pose(
        self, pnp: PnpCandidate, focal: float, record: dict[str, Any]
    ) -> None:
        """Set the pose (baseline: refine on inliers first) and audit its fit."""
        import numpy as np

        if not self.pose_acceptance.refine_before_gates:
            ok, rotation, translation = refine_on_inliers(
                pnp.objects, pnp.rays, pnp.rotation, pnp.translation, pnp.inliers
            )
            if not ok:
                raise ValueError("PnP refinement failed")
            pnp = replace(pnp, rotation=rotation, translation=translation)
        self.pose = pnp.world_to_camera()
        # Audit the final pose independently: RANSAC's membership mask is
        # not a guarantee that its returned/refined pose fits every inlier.
        accepted = pnp.inliers.ravel()
        refined_errors = pnp.reprojection_errors_px(focal)
        inlier_errors = refined_errors[accepted]
        record["refined_median_inlier_error_px"] = _median_or_none(inlier_errors)
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

    # -- Extending the map -----------------------------------------------------

    def _extend_map(self, points: Rays, focal: float, record: dict[str, Any]) -> None:
        """Validate/refine landmarks, triangulate new ones, maybe move keyframe.

        A keyframe must move enough to triangulate new points, so the previous
        reference is retained through small motions and replaced once it has
        replenished the map (or, for explicit policies, whenever selected).
        """
        if self.landmark_policy.keeps_observations:
            record["multiview_update"] = self._update_landmarks(points, focal)
        pose = self._tracked_pose
        reference = self._reference_keyframe
        old_count = len(self.landmarks)
        decision = self.keyframe_policy.decide(
            reference.points, reference.pose, points, pose
        )
        if self.keyframe_policy.explicit:
            record["keyframe_decision"] = decision
        if decision["selected"]:
            self._triangulate(reference.points, reference.pose, points, pose, focal)
            record["triangulation"] = asdict(self.triangulation_diagnostics)
        record["landmarks_added"] = len(self.landmarks) - old_count
        if (
            (self.keyframe_policy.explicit and decision["selected"])
            or len(self.landmarks) > old_count
            or (
                self.landmark_policy.validate_third_view
                and decision["selected"]
                and self.triangulation_diagnostics.accepted > 0
            )
        ):
            self.reference = _Keyframe(points, pose.copy())
        # IDs never return after loss; bound memory to currently visible IDs.
        self.landmarks = {
            key: value for key, value in self.landmarks.items() if key in points
        }

    # -- Landmark validation and refinement -----------------------------------

    def _update_landmarks(self, points: Rays, focal: float) -> dict[str, int]:
        """Validate a new observation after estimating pose from the active map.

        Bootstrap landmarks must provisionally seed the first PnP pose. Later
        candidates remain pending until a third view validates them; they never
        participate in estimating the pose used for their own third-view test.
        Returns the outcome counts for this frame.
        """
        counts = _LandmarkUpdateCounts()
        for pending, population in (
            (False, self.landmarks),
            (True, self.pending_landmarks),
        ):
            for key in list(population):
                if key in points and key in self.landmark_observations:
                    self._observe_landmark(
                        key, population, pending, points[key], focal, counts
                    )
        self._prune_landmark_records(points)
        counts.pending = len(self.pending_landmarks)
        return asdict(counts)

    def _observe_landmark(
        self,
        key: int,
        population: dict[int, Array],
        pending: bool,
        ray: Array,
        focal: float,
        counts: _LandmarkUpdateCounts,
    ) -> None:
        """Test one landmark against the current view; extend or refine it.

        `population` is the dict the landmark lives in (mapped or pending). A
        landmark that fits this view gains the observation and, under the
        refinement policy, a re-estimated position that must still fit every
        kept view; a pending landmark that fits is promoted to the map.
        """
        from fusionlab.vision.odometry.landmark_refinement import (
            refine_landmark,
            reprojection_errors,
        )

        import numpy as np

        point = population[key]
        observation = (self._tracked_pose.copy(), ray.copy())
        error = reprojection_errors(point, [observation], focal)[0]
        if not np.isfinite(error) or error > LANDMARK_MAX_ERROR_PX:
            # Only unconfirmed landmarks are dropped; confirmed ones may simply
            # be occluded or mismatched in this frame.
            if (
                self.landmark_policy.validate_third_view
                and key not in self.confirmed_landmarks
            ):
                del population[key]
                self.landmark_observations.pop(key, None)
                counts.rejected_third_view += 1
            return
        history = _bounded_history(self.landmark_observations[key], observation)
        candidate = (
            refine_landmark(point, history)
            if self.landmark_policy.refine_positions
            else point.copy()
        )
        errors = reprojection_errors(candidate, history, focal)
        if not (np.isfinite(errors).all() and errors.max() <= LANDMARK_MAX_ERROR_PX):
            return
        self.landmark_observations[key] = history
        population[key] = candidate
        if self.landmark_policy.refine_positions:
            counts.refined += 1
        if key not in self.confirmed_landmarks:
            self.confirmed_landmarks.add(key)
            counts.confirmed += 1
        if pending:
            self.landmarks[key] = candidate
            del self.pending_landmarks[key]

    def _prune_landmark_records(self, points: Rays) -> None:
        """Forget pending points, histories and confirmations of unseen IDs."""
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

    # -- Completing a tracked frame -------------------------------------------

    def _finish_tracking_frame(
        self, points: Rays, focal: float, record: dict[str, Any]
    ) -> None:
        """Prune birth steps, optionally run local BA, then report the pose."""
        self.landmark_birth = {
            key: self.landmark_birth[key]
            for key in self.landmarks.keys() | self.pending_landmarks.keys()
            if key in self.landmark_birth
        }
        if self.bundle_adjustment:
            record["bundle_adjustment"] = self._adjust_bundle(points, focal)
        pose = self._tracked_pose
        record.update(
            status="tracking",
            segment=self.segment,
            landmarks=len(self.landmarks),
            camera_center_arbitrary=camera_center(pose).tolist(),
            world_to_camera=pose.tolist(),
        )

    # -- Local bundle adjustment ----------------------------------------------

    def _adjust_bundle(self, points: Rays, focal: float) -> dict[str, Any]:
        """Optimize a bounded local window and validate current map support.

        Every BA_INTERVAL_STEPS frames, jointly refine up to BA_MAX_LANDMARKS
        well-observed landmarks and the window poses. The update is kept only
        if it does not degrade support in the current frame. Returns the
        `bundle_adjustment` diagnostics; skipped attempts carry a `reason`.
        """
        from fusionlab.vision.odometry.local_bundle_adjustment import (
            DegenerateAnchorBaseline,
            optimize_window,
        )

        import numpy as np

        self._append_bundle_frame(points)
        window = self.bundle_window
        if len(window) < BA_MIN_WINDOW_FRAMES or self.step % BA_INTERVAL_STEPS:
            return _skipped_bundle("waiting_for_window_or_interval")
        identifiers = self._select_bundle_landmarks(window)
        if len(identifiers) < MIN_MAP_CORRESPONDENCES:
            return _skipped_bundle("insufficient_multiview_landmarks")
        poses = np.array([frame.pose for frame in window])
        objects = np.array([self.landmarks[key] for key in identifiers])
        observations = _window_observations(window, identifiers)
        if any(
            sum(o.camera == index for o in observations) < BA_MIN_OBSERVATIONS_PER_POSE
            for index in range(len(window))
        ):
            return _skipped_bundle("insufficient_pose_observations")
        try:
            adjustment = optimize_window(poses, objects, observations, focal=focal)
        except DegenerateAnchorBaseline:
            # A near-stationary anchor baseline cannot constrain local scale.
            return _skipped_bundle("degenerate_anchor_baseline")
        diagnostics = adjustment.summary()
        diagnostics.update(
            attempted=True,
            window_frames=len(window),
            optimized_landmarks=len(identifiers),
        )
        if not adjustment.accepted:
            return diagnostics
        new_poses = adjustment.poses
        proposed = {**self.landmarks, **dict(zip(identifiers, adjustment.points))}
        if not self._bundle_keeps_current_support(
            new_poses[-1], proposed, points, focal, diagnostics
        ):
            diagnostics.update(accepted=False, reason="current_frame_support_degraded")
            return diagnostics
        self._apply_bundle_correction(window, poses, new_poses, proposed)
        diagnostics["window_pose_corrections"] = [
            {"step": frame.step, "world_to_camera": pose.tolist()}
            for frame, pose in zip(window, new_poses)
        ]
        return diagnostics

    def _append_bundle_frame(self, points: Rays) -> None:
        """Add this frame to the sliding window, seeding it with the origin."""
        if not self.bundle_window:
            if self.anchor is None:
                raise RuntimeError("A tracking segment always has an anchor frame")
            self.bundle_window.append(
                _BundleFrame(_origin_pose(), self.anchor.points, None)
            )
        self.bundle_window.append(
            _BundleFrame(self._tracked_pose.copy(), points, self.step)
        )
        self.bundle_window = self.bundle_window[-BA_WINDOW_FRAMES:]

    def _select_bundle_landmarks(self, window: list[_BundleFrame]) -> list[int]:
        """Landmarks seen in enough window frames, most-observed first."""
        counts = {
            key: sum(key in frame.points for frame in window) for key in self.landmarks
        }
        return sorted(
            (key for key, count in counts.items() if count >= BA_MIN_LANDMARK_VIEWS),
            key=lambda key: (-counts[key], key),
        )[:BA_MAX_LANDMARKS]

    def _bundle_keeps_current_support(
        self,
        new_pose: Array,
        proposed: dict[int, Array],
        points: Rays,
        focal: float,
        diagnostics: dict[str, Any],
    ) -> bool:
        """Guard: the update must not reduce support in the current frame.

        The selected-window cost can improve while the full current map fits
        worse; that case is rejected (see docs/3_experiment_results.md).
        """
        import numpy as np

        shared = sorted(self.landmarks.keys() & points.keys())
        before_count, before_error = _current_support(
            self._tracked_pose, self.landmarks, shared, points, focal
        )
        after_count, after_error = _current_support(
            new_pose, proposed, shared, points, focal
        )
        diagnostics.update(
            current_support_before=before_count,
            current_support_after=after_count,
            current_median_error_before_px=before_error,
            current_median_error_after_px=after_error,
        )
        return not (
            after_count < before_count
            or not np.isfinite(after_error)
            or after_error > before_error * BA_MAX_MEDIAN_ERROR_GROWTH
        )

    def _apply_bundle_correction(
        self,
        window: list[_BundleFrame],
        poses: Array,
        new_poses: Array,
        proposed: dict[int, Array],
    ) -> None:
        """Adopt corrected window poses and landmarks everywhere they are stored.

        Observation histories and the reference keyframe refer to window poses by
        value, so they are updated too. Other historical poses stay fixed; there
        is no marginalization, and saved output poses are never rewritten.
        """
        for history in self.landmark_observations.values():
            for index, (pose, observed) in enumerate(history):
                corrected = _corrected_pose(pose, poses, new_poses)
                if corrected is not None:
                    history[index] = (corrected, observed)
        reference = self._reference_keyframe
        corrected = _corrected_pose(reference.pose, poses, new_poses)
        if corrected is not None:
            self.reference = _Keyframe(reference.points, corrected)
        for frame, pose in zip(window, new_poses):
            frame.pose = pose.copy()
        self.landmarks = proposed
        self.pose = new_poses[-1].copy()


def _skipped_bundle(reason: str) -> dict[str, Any]:
    """Diagnostics for a bundle adjustment that was not attempted."""
    return {"attempted": False, "reason": reason}


def _corrected_pose(pose: Array, old_poses: Array, new_poses: Array) -> Array | None:
    """A copy of the BA correction for `pose` if it is a window pose, else None."""
    import numpy as np

    for old, new in zip(old_poses, new_poses):
        if np.array_equal(pose, old):
            return new.copy()
    return None


def _current_support(
    pose: Array,
    landmarks: dict[int, Array],
    shared: list[int],
    points: Rays,
    focal: float,
) -> tuple[int, float]:
    """Count landmarks within 3 px in this frame, and the median error.

    Points at or behind BA_SUPPORT_MIN_DEPTH get infinite error, so they count
    as unsupported and raise the median rather than being dropped.
    """
    import numpy as np

    objects = np.array([landmarks[key] for key in shared])
    camera = objects @ pose[:, :3].T + pose[:, 3]
    errors = np.full(len(shared), np.inf)
    valid = (camera[:, 2] > BA_SUPPORT_MIN_DEPTH) & np.isfinite(camera).all(axis=1)
    observed = np.array([points[key] for key in shared])
    errors[valid] = (
        np.linalg.norm(
            camera[valid, :2] / camera[valid, 2, None] - observed[valid], axis=1
        )
        * focal
    )
    return int((errors <= PNP_INLIER_THRESHOLD_PX).sum()), float(np.median(errors))
