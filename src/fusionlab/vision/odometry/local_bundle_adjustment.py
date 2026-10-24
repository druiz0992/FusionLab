"""Bounded local bundle adjustment with two fixed camera-pose anchors.

Jointly refines the world-to-camera poses of a short window and the landmarks
they observe by minimizing normalized reprojection error, expressed in
approximate pixels by multiplying by the mean tracking focal length. The first
two poses stay fixed: they keep the window's arbitrary frame and scale (gauge),
so an update cannot drift or rescale the segment.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from fusionlab._typing import Array

# Method caveats saved with every report (what the numbers do not show).
BUNDLE_ADJUSTMENT_LIMITS = (
    "Bounded cost-decreasing step may be accepted at evaluation limit; convergence "
    "is reported separately. Two complete anchor poses remain fixed."
)

# Poses 0 and 1 are held fixed; only poses 2..N-1 are optimized.
FIXED_ANCHOR_POSES = 2
# At least two optimized poses on top of the anchors.
MIN_WINDOW_POSES = 4
# Every landmark must be seen from this many distinct poses.
MIN_LANDMARK_VIEWS = 3
# Anchor camera centers closer than this cannot fix the window scale.
MIN_ANCHOR_BASELINE = 1e-6
# Depths are floored here before division to keep residuals finite.
DEPTH_FLOOR = 1e-6
# Points closer than this depth add a penalty residual pushing them in front.
DEPTH_PENALTY_MARGIN = 1e-4
# Soft-L1 robust loss scale, in approximate pixels.
SOFT_L1_SCALE_PX = 2.0
# Solver stopping tolerances (relative), shared by ftol/xtol/gtol.
SOLVER_TOLERANCE = 1e-4


class WindowObservation(NamedTuple):
    """One landmark seen from one window pose."""

    camera: int  # index into the window poses
    landmark: int  # index into the window landmarks
    ray: Array | Sequence[float]  # normalized, undistorted image (x, y)


class DegenerateAnchorBaseline(ValueError):
    """The two fixed anchor poses share a camera center, so scale is free."""


@dataclass(frozen=True)
class WindowAdjustment:
    """Result of one bounded bundle-adjustment attempt.

    `poses` and `points` are the refined values when `accepted`, otherwise
    unchanged copies of the inputs. Costs are SciPy's soft-L1 objective and RMS
    values are x/y reprojection errors, both in approximate tracking pixels
    over the selected window only, not the whole map.
    """

    poses: Array  # (N, 3, 4) world-to-camera [R | t], anchors unchanged
    points: Array  # (M, 3) landmarks in the segment's arbitrary-scale world
    accepted: bool  # cost decreased, finite, every point in front of its cameras
    initial_robust_cost: float
    final_robust_cost: float
    initial_rms_px: float
    final_rms_px: float
    positive_depth: bool
    solver_converged: bool  # SciPy's own success flag; acceptance ignores it
    evaluations: int  # residual evaluations used, at most the budget

    def summary(self) -> dict[str, Any]:
        """JSON-ready diagnostics, in the saved `bundle_adjustment` key order."""
        return {
            "accepted": self.accepted,
            "initial_robust_cost": self.initial_robust_cost,
            "final_robust_cost": self.final_robust_cost,
            "initial_rms_px": self.initial_rms_px,
            "final_rms_px": self.final_rms_px,
            "positive_depth": self.positive_depth,
            "solver_converged": self.solver_converged,
            "evaluations": self.evaluations,
            "fixed_pose_indices": list(range(FIXED_ANCHOR_POSES)),
            "limits": BUNDLE_ADJUSTMENT_LIMITS,
        }


@dataclass(frozen=True)
class _WindowProblem:
    """A validated window: poses, landmarks and indexed observations.

    The parameter vector is [rvec, t] for each optimized pose (6 values each,
    Rodrigues rotation vector and translation), followed by every landmark's
    XYZ in the segment's arbitrary-scale world frame.
    """

    poses: Array  # (N, 3, 4) world-to-camera [R | t]
    points: Array  # (M, 3) world landmarks
    cameras: Array  # (K,) pose index of each observation
    landmarks: Array  # (K,) landmark index of each observation
    observed: Array  # (K, 2) normalized image coordinates
    focal: float  # mean tracking focal length, px

    @property
    def pose_parameter_count(self) -> int:
        """Six values (rotation vector, translation) per optimized pose."""
        return 6 * (len(self.poses) - FIXED_ANCHOR_POSES)

    def pack(self) -> Array:
        """Initial parameter vector from the current poses and landmarks."""
        import cv2
        import numpy as np

        optimized = self.poses[FIXED_ANCHOR_POSES:]
        return np.concatenate(
            [
                np.concatenate((cv2.Rodrigues(p[:, :3])[0].ravel(), p[:, 3]))
                for p in optimized
            ]
            + [self.points.ravel()]
        )

    def unpack(self, parameters: Array) -> tuple[Array, Array]:
        """Poses (anchors unchanged) and landmarks for a parameter vector."""
        import cv2
        import numpy as np

        current_poses = self.poses.copy()
        for index in range(FIXED_ANCHOR_POSES, len(self.poses)):
            start = 6 * (index - FIXED_ANCHOR_POSES)
            current_poses[index] = np.column_stack(
                (
                    cv2.Rodrigues(parameters[start : start + 3])[0],
                    parameters[start + 3 : start + 6],
                )
            )
        landmarks = parameters[self.pose_parameter_count :].reshape(-1, 3)
        return current_poses, landmarks

    def camera_points(self, parameters: Array) -> Array:
        """Each observation's landmark in its observing camera's frame."""
        import numpy as np

        current_poses, current_points = self.unpack(parameters)
        selected = current_poses[self.cameras]
        return (
            np.einsum("nij,nj->ni", selected[:, :, :3], current_points[self.landmarks])
            + selected[:, :, 3]
        )

    def residuals(self, parameters: Array) -> Array:
        """Per observation: x/y reprojection error (px) and a depth penalty.

        The penalty is zero for points comfortably in front of the camera and
        grows linearly as a point approaches or crosses the image plane, so the
        solver is pushed away from the depth floor instead of exploiting it.
        """
        import numpy as np

        camera = self.camera_points(parameters)
        depth = np.maximum(camera[:, 2], DEPTH_FLOOR)
        errors = np.empty((len(camera), 3))
        errors[:, :2] = (camera[:, :2] / depth[:, None] - self.observed) * self.focal
        errors[:, 2] = np.maximum(DEPTH_PENALTY_MARGIN - camera[:, 2], 0) * self.focal
        return errors.ravel()

    def jacobian_sparsity(self, parameter_count: int) -> Any:
        """Each residual depends on one pose (if optimized) and one landmark.

        Declaring this keeps finite-difference Jacobians cheap even with many
        landmarks in the window.
        """
        from scipy.sparse import lil_matrix

        sparsity = lil_matrix((3 * len(self.cameras), parameter_count), dtype=int)
        for index, (camera, landmark) in enumerate(zip(self.cameras, self.landmarks)):
            rows = slice(3 * index, 3 * index + 3)
            if camera >= FIXED_ANCHOR_POSES:
                pose_start = 6 * (camera - FIXED_ANCHOR_POSES)
                sparsity[rows, pose_start : pose_start + 6] = 1
            start = self.pose_parameter_count + 3 * landmark
            sparsity[rows, start : start + 3] = 1
        return sparsity.tocsr()


def optimize_window(
    poses: Array,
    points: Array,
    observations: Sequence[WindowObservation | tuple[int, int, Any]],
    *,
    focal: float,
    max_evaluations: int = 25,
) -> WindowAdjustment:
    """Jointly refine camera poses 2..N and 3D points in an arbitrary-scale frame.

    `observations` holds (pose index, landmark index, normalized xy) entries.
    The first two poses are fixed, preserving their nonzero baseline and removing
    local similarity gauge freedom. This is stronger than minimal gauge fixing:
    anchor errors are retained. Inputs are never mutated.

    The step is accepted only if the robust cost decreases, every value stays
    finite and every point stays in front of its cameras; otherwise the result
    carries unchanged copies of the inputs. Raises `DegenerateAnchorBaseline`
    when the anchors cannot fix scale and ValueError for other invalid inputs.
    """
    problem = _build_problem(poses, points, observations, focal, max_evaluations)
    initial = problem.pack()
    solution = _solve(problem, initial, max_evaluations)
    return _judge_step(problem, initial, solution)


def _solve(problem: _WindowProblem, initial: Array, max_evaluations: int) -> Any:
    """Run SciPy's sparse trust-region solver with a soft-L1 robust loss."""
    least_squares = _load_solver()
    return least_squares(
        problem.residuals,
        initial,
        jac_sparsity=problem.jacobian_sparsity(len(initial)),
        loss="soft_l1",
        f_scale=SOFT_L1_SCALE_PX,
        max_nfev=max_evaluations,
        ftol=SOLVER_TOLERANCE,
        xtol=SOLVER_TOLERANCE,
        gtol=SOLVER_TOLERANCE,
        x_scale="jac",
    )


def _judge_step(
    problem: _WindowProblem, initial: Array, solution: Any
) -> WindowAdjustment:
    """Accept the solver's step only if it is finite, cheaper and in front.

    A bounded step is accepted even when the solver has not converged: the
    evaluation budget is deliberately small, and any cost decrease is useful.
    """
    import numpy as np

    initial_residual = problem.residuals(initial)
    final_residual = problem.residuals(solution.x)
    before = _robust_cost(initial_residual)
    after = _robust_cost(final_residual)
    positive_depth = bool((problem.camera_points(solution.x)[:, 2] > DEPTH_FLOOR).all())
    accepted = bool(
        np.isfinite(solution.x).all()
        and np.isfinite(after)
        and after < before
        and positive_depth
    )
    refined_poses, refined_points = (
        problem.unpack(solution.x)
        if accepted
        else (problem.poses.copy(), problem.points.copy())
    )
    return WindowAdjustment(
        poses=refined_poses,
        points=refined_points,
        accepted=accepted,
        initial_robust_cost=before,
        final_robust_cost=after,
        initial_rms_px=_reprojection_rms(initial_residual),
        final_rms_px=_reprojection_rms(final_residual),
        positive_depth=positive_depth,
        solver_converged=bool(solution.success),
        evaluations=solution.nfev,
    )


def _load_solver() -> Callable[..., Any]:
    """SciPy's least_squares, with an install hint if SciPy is missing."""
    try:
        from scipy.optimize import least_squares
    except ImportError as exc:
        raise ValueError(
            'Install bundle adjustment support: pip install -e ".[odometry]"'
        ) from exc
    return least_squares


def _build_problem(
    poses: Array,
    points: Array,
    observations: Sequence[Any],
    focal: float,
    max_evaluations: int,
) -> _WindowProblem:
    """Validate inputs and index the observations; never mutates the inputs."""
    import numpy as np

    poses = np.asarray(poses, dtype=float)
    points = np.asarray(points, dtype=float)
    _validate_window(poses, points, focal, max_evaluations)
    cameras, landmarks, observed = _index_observations(
        observations, len(poses), len(points)
    )
    return _WindowProblem(poses, points, cameras, landmarks, observed, focal)


def _validate_window(
    poses: Array, points: Array, focal: float, max_evaluations: int
) -> None:
    """Shapes, finiteness, budget and a nonzero anchor baseline."""
    import numpy as np

    if poses.ndim != 3 or poses.shape[1:] != (3, 4) or len(poses) < MIN_WINDOW_POSES:
        raise ValueError("Need at least four 3x4 world-to-camera poses")
    if points.ndim != 2 or points.shape[1:] != (3,) or not len(points):
        raise ValueError("Need N by 3 landmarks")
    if (
        not np.isfinite(poses).all()
        or not np.isfinite(points).all()
        or not np.isfinite(focal)
        or focal <= 0
    ):
        raise ValueError(
            "Bundle adjustment inputs must be finite, with positive focal length"
        )
    if max_evaluations < 1:
        raise ValueError("Evaluation budget must be positive")
    centers = [-pose[:, :3].T @ pose[:, 3] for pose in poses[:FIXED_ANCHOR_POSES]]
    if np.linalg.norm(centers[1] - centers[0]) < MIN_ANCHOR_BASELINE:
        raise DegenerateAnchorBaseline(
            "Anchor poses need a nonzero translation baseline"
        )


def _index_observations(
    observations: Sequence[Any], pose_count: int, point_count: int
) -> tuple[Array, Array, Array]:
    """Split (camera, landmark, xy) entries into arrays and check coverage.

    Every landmark needs MIN_LANDMARK_VIEWS distinct poses: with fewer, its
    depth along the ray is too weakly constrained to refine alongside poses.
    """
    import numpy as np

    cameras = np.asarray([o[0] for o in observations], dtype=int)
    landmarks = np.asarray([o[1] for o in observations], dtype=int)
    observed = np.asarray([o[2] for o in observations], dtype=float)
    if (
        observed.shape != (len(observations), 2)
        or not len(observations)
        or not np.isfinite(observed).all()
    ):
        raise ValueError("Invalid normalized observations")
    if (
        (cameras < 0).any()
        or (cameras >= pose_count).any()
        or (landmarks < 0).any()
        or (landmarks >= point_count).any()
    ):
        raise ValueError("Observation index out of bounds")
    for index in range(point_count):
        if len(set(cameras[landmarks == index])) < MIN_LANDMARK_VIEWS:
            raise ValueError(
                "Each optimized landmark needs three distinct observed poses"
            )
    return cameras, landmarks, observed


def _robust_cost(errors: Array) -> float:
    """SciPy's soft-L1 cost at f_scale s: sum of s² (sqrt(1 + (e/s)²) - 1).

    This is 0.5 · s² · ρ(e²/s²) per residual with ρ(z) = 2 (sqrt(1 + z) - 1),
    so it matches the objective least_squares minimizes.
    """
    import numpy as np

    scale = SOFT_L1_SCALE_PX
    return float((scale**2 * (np.sqrt(1 + (errors / scale) ** 2) - 1)).sum())


def _reprojection_rms(residuals: Array) -> float:
    """RMS of the x/y reprojection residuals (px), ignoring depth penalties."""
    import numpy as np

    return float(np.sqrt(np.mean(residuals.reshape(-1, 3)[:, :2] ** 2)))
