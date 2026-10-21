"""Bounded local bundle adjustment with two fixed camera-pose anchors."""

# Method caveats saved with every report (what the numbers do not show).
BUNDLE_ADJUSTMENT_LIMITS = (
    "Bounded cost-decreasing step may be accepted at evaluation limit; convergence "
    "is reported separately. Two complete anchor poses remain fixed."
)


def optimize_window(poses, points, observations, *, focal, max_evaluations=25):
    """Jointly refine camera poses 2..N and 3D points in an arbitrary-scale frame.

    observations holds (camera index, landmark index, normalized XY) tuples.
    The first two poses are fixed, preserving their nonzero baseline and removing
    local similarity gauge freedom. This is stronger than minimal gauge fixing:
    anchor errors are retained. Inputs are never mutated.
    """
    import cv2
    import numpy as np

    try:
        from scipy.optimize import least_squares
        from scipy.sparse import lil_matrix
    except ImportError as exc:
        raise ValueError(
            'Install bundle adjustment support: pip install -e ".[odometry]"'
        ) from exc

    poses = np.asarray(poses, dtype=float)
    points = np.asarray(points, dtype=float)
    if poses.ndim != 3 or poses.shape[1:] != (3, 4) or len(poses) < 4:
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
    centers = [-pose[:, :3].T @ pose[:, 3] for pose in poses[:2]]
    if np.linalg.norm(centers[1] - centers[0]) < 1e-6:
        raise ValueError("Anchor poses need a nonzero translation baseline")
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
        or (cameras >= len(poses)).any()
        or (landmarks < 0).any()
        or (landmarks >= len(points)).any()
    ):
        raise ValueError("Observation index out of bounds")
    for index in range(len(points)):
        if len(set(cameras[landmarks == index])) < 3:
            raise ValueError(
                "Each optimized landmark needs three distinct observed poses"
            )
    parameter_count = 6 * (len(poses) - 2)
    initial = np.concatenate(
        [
            np.concatenate((cv2.Rodrigues(p[:, :3])[0].ravel(), p[:, 3]))
            for p in poses[2:]
        ]
        + [points.ravel()]
    )

    def unpack(parameters):
        current_poses = poses.copy()
        for index in range(2, len(poses)):
            start = 6 * (index - 2)
            current_poses[index] = np.column_stack(
                (
                    cv2.Rodrigues(parameters[start : start + 3])[0],
                    parameters[start + 3 : start + 6],
                )
            )
        return current_poses, parameters[parameter_count:].reshape(-1, 3)

    def project(parameters):
        current_poses, current_points = unpack(parameters)
        selected = current_poses[cameras]
        camera = (
            np.einsum("nij,nj->ni", selected[:, :, :3], current_points[landmarks])
            + selected[:, :, 3]
        )
        return camera

    def residual(parameters):
        camera = project(parameters)
        depth = np.maximum(camera[:, 2], 1e-6)
        errors = np.empty((len(camera), 3))
        errors[:, :2] = (camera[:, :2] / depth[:, None] - observed) * focal
        errors[:, 2] = np.maximum(1e-4 - camera[:, 2], 0) * focal
        return errors.ravel()

    # Each residual depends on one pose and one point. Sparse differentiation
    # keeps work bounded even with many landmarks in the local window.
    sparsity = lil_matrix((3 * len(observations), len(initial)), dtype=int)
    for index, (camera, landmark) in enumerate(zip(cameras, landmarks)):
        if camera >= 2:
            sparsity[3 * index : 3 * index + 3, 6 * (camera - 2) : 6 * (camera - 1)] = 1
        start = parameter_count + 3 * landmark
        sparsity[3 * index : 3 * index + 3, start : start + 3] = 1

    def robust_cost(errors):
        return float((4 * (np.sqrt(1 + (errors / 2) ** 2) - 1)).sum())

    initial_residual = residual(initial)
    solution = least_squares(
        residual,
        initial,
        jac_sparsity=sparsity.tocsr(),
        loss="soft_l1",
        f_scale=2.0,
        max_nfev=max_evaluations,
        ftol=1e-4,
        xtol=1e-4,
        gtol=1e-4,
        x_scale="jac",
    )
    final_residual = residual(solution.x)
    before = robust_cost(initial_residual)
    after = robust_cost(final_residual)
    positive_depth = bool((project(solution.x)[:, 2] > 1e-6).all())
    accepted = bool(
        np.isfinite(solution.x).all()
        and np.isfinite(after)
        and after < before
        and positive_depth
    )
    refined_poses, refined_points = (
        unpack(solution.x) if accepted else (poses.copy(), points.copy())
    )
    diagnostics = {
        "accepted": accepted,
        "initial_robust_cost": before,
        "final_robust_cost": after,
        "initial_rms_px": float(
            np.sqrt(np.mean(initial_residual.reshape(-1, 3)[:, :2] ** 2))
        ),
        "final_rms_px": float(
            np.sqrt(np.mean(final_residual.reshape(-1, 3)[:, :2] ** 2))
        ),
        "positive_depth": positive_depth,
        "solver_converged": bool(solution.success),
        "evaluations": solution.nfev,
        "fixed_pose_indices": [0, 1],
        "limits": BUNDLE_ADJUSTMENT_LIMITS,
    }
    return refined_poses, refined_points, diagnostics
