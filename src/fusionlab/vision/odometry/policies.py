"""Experimental VO policy variants, selectable by name from the CLI.

Each registry maps a CLI name to a policy object. `MonocularOdometry` only asks
the policy questions ("refine before the support gates?", "is this a
keyframe?"), so a new variant is a new registry entry, or a new class for
keyframe rules, without editing the estimator loop. "baseline" is always the
default and the reference for comparisons; variants are experiments, not
recommendations (see docs/3_experiment_results.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

# -- Pose acceptance ----------------------------------------------------------


@dataclass(frozen=True)
class PoseAcceptance:
    """When the PnP pose is refined relative to the support gates.

    baseline: gate on RANSAC support, then refine the pose on the inliers; a
              rejected candidate gets a counterfactual "would refinement pass?"
              audit for diagnostics only.
    refined:  refine first, recount positive-depth 3 px support, then gate.
    """

    name: str
    refine_before_gates: bool


POSE_ACCEPTANCE: dict[str, PoseAcceptance] = {
    "baseline": PoseAcceptance("baseline", refine_before_gates=False),
    "refined": PoseAcceptance("refined", refine_before_gates=True),
}


# -- Landmark management ------------------------------------------------------


@dataclass(frozen=True)
class LandmarkPolicy:
    """How triangulated landmarks are confirmed and improved.

    validate_third_view: new (non-bootstrap) landmarks stay pending, excluded
        from PnP, until a third view reprojects within 3 px; unconfirmed
        landmarks that fail a later view are removed.
    refine_positions: re-estimate each landmark from its bounded observation
        history (first view plus four latest) whenever it is observed again.
    """

    name: str
    validate_third_view: bool
    refine_positions: bool

    @property
    def keeps_observations(self) -> bool:
        """Observation histories are only needed to validate or refine."""
        return self.validate_third_view or self.refine_positions


LANDMARK_POLICIES: dict[str, LandmarkPolicy] = {
    "baseline": LandmarkPolicy("baseline", False, False),
    "validation": LandmarkPolicy("validation", True, False),
    "refinement": LandmarkPolicy("refinement", False, True),
    "multiview": LandmarkPolicy("multiview", True, True),
}


# -- Keyframe selection -------------------------------------------------------


class KeyframePolicy(Protocol):
    """Decides when to triangulate against the reference keyframe."""

    name: str
    # True when the decision is saved in diagnostics and a selected keyframe
    # always becomes the new reference, even if it added no landmarks.
    explicit: bool

    def decide(
        self, reference_points: Any, reference_pose: Any, points: Any, pose: Any
    ) -> dict[str, Any]:
        """Return at least {"selected": bool, "reason": str}."""
        ...


@dataclass(frozen=True)
class EveryFrameKeyframes:
    """baseline: try triangulating against the reference on every frame."""

    name: str = "baseline"
    explicit: bool = False

    def decide(self, reference_points, reference_pose, points, pose):
        return {"selected": True, "reason": "baseline_try_every_frame"}


@dataclass(frozen=True)
class ParallaxKeyframes:
    """parallax: wait for rotation-compensated parallax or track loss."""

    name: str = "parallax"
    explicit: bool = True
    parallax_deg: float = 2.0
    survival_ratio: float = 0.6

    def decide(self, reference_points, reference_pose, points, pose):
        return choose_keyframe(
            reference_points,
            reference_pose,
            points,
            pose,
            parallax_deg=self.parallax_deg,
            survival_ratio=self.survival_ratio,
        )


KEYFRAME_POLICIES: dict[str, KeyframePolicy] = {
    "baseline": EveryFrameKeyframes(),
    "parallax": ParallaxKeyframes(),
}


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


def lookup(registry: dict[str, Any], name: str, description: str) -> Any:
    """Return the named policy, or raise listing the known names."""
    if name not in registry:
        known = ", ".join(registry)
        raise ValueError(f"Unknown {description}: {name!r} (choose from {known})")
    return registry[name]
