"""Small fixed-pose landmark refinement, separate from bundle adjustment."""


def refine_landmark(point, observations, *, iterations=6):
    """Minimize normalized reprojection error over fixed world-to-camera poses.

    Only the three landmark coordinates change. A damped Gauss–Newton step is
    accepted only if every depth stays positive and the total cost decreases.
    Returns a copy of the original point if no improving step is found.
    """
    import numpy as np

    current = np.asarray(point, dtype=float).copy()
    if current.shape != (3,) or not np.isfinite(current).all() or len(observations) < 3:
        raise ValueError("Need a finite 3D point and at least three observations")

    def residual_and_jacobian(candidate):
        residuals = []
        jacobians = []
        for pose, observed in observations:
            camera = pose[:, :3] @ candidate + pose[:, 3]
            x, y, z = camera
            if z <= 1e-8 or not np.isfinite(camera).all():
                return None, None
            residuals.extend(camera[:2] / z - observed)
            projection = np.array([[1 / z, 0, -x / (z * z)], [0, 1 / z, -y / (z * z)]])
            jacobians.extend(projection @ pose[:, :3])
        return np.asarray(residuals), np.asarray(jacobians)

    residual, jacobian = residual_and_jacobian(current)
    if residual is None:
        return current
    for _ in range(iterations):
        step = np.linalg.lstsq(
            jacobian.T @ jacobian + 1e-9 * np.eye(3), -jacobian.T @ residual, rcond=None
        )[0]
        if not np.isfinite(step).all() or np.linalg.norm(step) < 1e-9:
            break
        improved = False
        for fraction in (1.0, 0.5, 0.25, 0.125):
            trial = current + fraction * step
            trial_residual, trial_jacobian = residual_and_jacobian(trial)
            if (
                trial_residual is not None
                and trial_residual @ trial_residual < residual @ residual
            ):
                current, residual, jacobian = trial, trial_residual, trial_jacobian
                improved = True
                break
        if not improved:
            break
    return current


def reprojection_errors(point, observations, focal):
    """Per-view approximate tracking-pixel errors; invalid depth is infinity."""
    import numpy as np

    errors = []
    for pose, observed in observations:
        camera = pose[:, :3] @ point + pose[:, 3]
        errors.append(
            float(np.linalg.norm(camera[:2] / camera[2] - observed) * focal)
            if camera[2] > 1e-8 and np.isfinite(camera).all()
            else float("inf")
        )
    return np.asarray(errors)
