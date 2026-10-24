"""Timestamp-associated, per-segment monocular evaluation against reference poses.

Pipeline:
1. Load the reference trajectory: body poses on integer-ns timestamps and the
   fixed camera mounting.
2. Associate every saved VO frame with the reference camera pose at its source
   timestamp: an exact sample or an interpolation between two samples at most
   `max_gap_ms` apart, never an extrapolation.
3. Group tracked, matched frames by VO segment and fit one similarity (Sim(3):
   scale, rotation, translation) per segment, because each monocular segment
   has its own arbitrary frame and scale.
4. Report aligned position and rotation errors, coverage and continuity.

Frames: the reference gives T_world_body (4×4); the camera pose is
T_world_camera = T_world_body T_body_camera. VO saves world-to-camera [R | t]
(3×4) in each segment's frame, with camera center -R.T t.
"""

from __future__ import annotations

import bisect
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fusionlab._typing import Array

# Method caveats saved with every report (what the numbers do not show).
TRAJECTORY_SCORE_LIMITS = (
    "Per-segment fit measures local consistency, not whole-route drift or metric "
    "scale recovery. Short segments may fit well despite fragmentation. No "
    "extrapolation across missing ground truth. Frame fractions describe processed "
    "frames, not time coverage."
)
ALIGNMENT_DESCRIPTION = (
    "Independent positive-scale Sim(3) per segment; fitted on scored samples"
)
# Alignment needs this many matched positions, spanning more than a line.
MIN_ALIGNMENT_POSITIONS = 3


@dataclass(frozen=True)
class Similarity:
    """y ≈ scale · R x + t, mapping estimated positions onto the reference."""

    scale: float  # reference units (m) per estimated segment unit
    rotation: Array  # 3×3, proper (det +1)
    translation: Array  # (3,), reference frame

    def apply(self, points: Array) -> Array:
        """Transform (N, 3) estimated positions into the reference frame."""
        return self.scale * (points @ self.rotation.T) + self.translation


def similarity_alignment(
    estimated: Array | Sequence[Sequence[float]],
    reference: Array | Sequence[Sequence[float]],
) -> Similarity:
    """Least-squares proper Sim(3), with positive scale and noncollinear support.

    Umeyama's closed form: the rotation comes from the SVD of the
    cross-covariance of the centred point sets (with a sign fix so det R = +1),
    and the scale from the singular values over the estimated set's variance.
    Raises ValueError when the fit is underconstrained or degenerate.
    """
    import numpy as np

    x, y = np.asarray(estimated, dtype=float), np.asarray(reference, dtype=float)
    if (
        x.shape != y.shape
        or x.ndim != 2
        or x.shape[1] != 3
        or len(x) < MIN_ALIGNMENT_POSITIONS
    ):
        raise ValueError("Alignment needs at least three matching 3D positions")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite trajectory")
    xc, yc = x - x.mean(axis=0), y - y.mean(axis=0)
    if np.linalg.matrix_rank(xc) < 2 or np.linalg.matrix_rank(yc) < 2:
        raise ValueError("Alignment is underconstrained (stationary or collinear)")
    u, singular, vt = np.linalg.svd(yc.T @ xc / len(x))
    sign = np.ones(3)
    sign[-1] = np.linalg.det(u @ vt)
    rotation = u @ np.diag(sign) @ vt
    scale = float(singular @ sign / np.mean(np.sum(xc * xc, axis=1)))
    if scale <= 0:
        raise ValueError("Nonpositive alignment scale")
    translation = y.mean(axis=0) - scale * rotation @ x.mean(axis=0)
    return Similarity(scale, rotation, translation)


@dataclass(frozen=True)
class ReferenceTrajectory:
    """Reference body poses and the camera mounting, on integer-ns timestamps.

    Orientations are interpolated by SLERP and positions linearly. Times are
    converted to seconds only after subtracting the first timestamp, so
    nanosecond precision survives the float conversion.
    """

    stamps_ns: list[int]  # strictly increasing
    world_from_body: Array  # (N, 4, 4) T_world_body
    body_from_camera: Array  # 4×4 T_body_camera
    rotations: Any = field(repr=False)  # SciPy Slerp over seconds since start

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ReferenceTrajectory:
        """Parse a `groundtruth.json` dict ({"states": [...], "body_from_camera"})."""
        import numpy as np
        from scipy.spatial.transform import Rotation, Slerp

        from fusionlab.recordings.euroc import rigid_transform

        states = data["states"]
        stamps = [int(state["source_time_ns"]) for state in states]
        if len(stamps) < 2 or any(b <= a for a, b in zip(stamps, stamps[1:])):
            raise ValueError("Ground-truth timestamps must increase strictly")
        transforms = np.asarray(
            [rigid_transform(state["world_from_body"]) for state in states]
        )
        body_from_camera = rigid_transform(data["body_from_camera"])
        times = np.array([(stamp - stamps[0]) / 1e9 for stamp in stamps])
        rotations = Slerp(times, Rotation.from_matrix(transforms[:, :3, :3]))
        return cls(stamps, transforms, body_from_camera, rotations)

    def camera_pose_at(self, stamp_ns: int, max_gap_ns: float) -> Array | None:
        """T_world_camera at `stamp_ns`, or None outside or across a long gap."""
        stamps = self.stamps_ns
        index = bisect.bisect_left(stamps, stamp_ns)
        if index < len(stamps) and stamps[index] == stamp_ns:
            return self.world_from_body[index] @ self.body_from_camera
        if index == 0 or index == len(stamps):
            return None
        a, b = index - 1, index
        if stamps[b] - stamps[a] > max_gap_ns:
            return None
        return self._interpolated_body(stamp_ns, a, b) @ self.body_from_camera

    def _interpolated_body(self, stamp_ns: int, a: int, b: int) -> Array:
        """T_world_body between samples a and b: SLERP rotation, linear position."""
        import numpy as np

        stamps = self.stamps_ns
        fraction = (stamp_ns - stamps[a]) / (stamps[b] - stamps[a])
        pose = np.eye(4)
        pose[:3, :3] = self.rotations([(stamp_ns - stamps[0]) / 1e9]).as_matrix()[0]
        positions = self.world_from_body[:, :3, 3]
        pose[:3, 3] = (1 - fraction) * positions[a] + fraction * positions[b]
        return pose

    def camera_positions(self) -> Array:
        """(N, 3) reference camera centers at every sample, world frame."""
        return (self.world_from_body @ self.body_from_camera)[:, :3, 3]


@dataclass(frozen=True)
class _MatchedFrame:
    """A tracked VO frame and its reference camera pose."""

    row: dict[str, Any]  # saved `poses.jsonl` row
    reference: Array  # 4×4 T_world_camera


@dataclass
class _Association:
    """Frame counts and matched frames grouped by VO segment (in first-seen order)."""

    segments: dict[int, list[_MatchedFrame]] = field(default_factory=dict)
    reference_available: int = 0  # processed frames with a reference pose
    tracked: int = 0  # frames the estimator tracked
    matched: int = 0  # tracked frames with a reference pose


def evaluate_run(
    run_dir: Path | str,
    groundtruth: Path | str,
    output: Path | str,
    *,
    max_gap_ms: float = 50.0,
) -> dict[str, Any]:
    """Score a saved VO run against reference poses, one alignment per segment.

    Writes `summary.json`, `aligned.jsonl`, `report.md` and, when any segment
    aligns, `trajectory.svg` into the new directory `output`. Returns the summary.
    """
    import numpy as np

    from fusionlab.evaluation.vo_evaluation import summarize_run

    if not np.isfinite(max_gap_ms) or max_gap_ms <= 0:
        raise ValueError("max_gap_ms must be positive and finite")
    run_dir, groundtruth, output = map(Path, (run_dir, groundtruth, output))
    reference = ReferenceTrajectory.from_json(json.loads(groundtruth.read_text()))
    rows = _read_poses(run_dir / "poses.jsonl")
    association = _associate(rows, reference, max_gap_ms * 1e6)
    segments: list[dict[str, Any]] = []
    plot_rows: list[dict[str, Any]] = []
    for segment, samples in association.segments.items():
        record, aligned_rows = _score_segment(segment, samples)
        segments.append(record)
        plot_rows += aligned_rows
    summary = {
        "run": str(run_dir.resolve()),
        "groundtruth": str(groundtruth.resolve()),
        "continuity": summarize_run(run_dir),
        **_coverage(association, len(rows)),
        "max_reference_gap_ms": max_gap_ms,
        "segments": segments,
        "alignment": ALIGNMENT_DESCRIPTION,
        "limits": TRAJECTORY_SCORE_LIMITS,
    }
    _write_outputs(output, summary, plot_rows, reference)
    return summary


def _read_poses(path: Path) -> list[dict[str, Any]]:
    """Every saved VO row, in processing order; an empty run is an error."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("Empty estimated trajectory")
    return rows


def _associate(
    rows: list[dict[str, Any]], reference: ReferenceTrajectory, max_gap_ns: float
) -> _Association:
    """Look up the reference pose of every row; group tracked matches by segment.

    Non-tracking frames still count towards reference availability, so the
    matched fraction measures how much of the scorable time was tracked.
    """
    association = _Association()
    for row in rows:
        pose = reference.camera_pose_at(int(row["source_time_ns"]), max_gap_ns)
        association.reference_available += pose is not None
        if row["status"] != "tracking":
            continue
        association.tracked += 1
        if pose is None:
            continue
        association.matched += 1
        association.segments.setdefault(row["segment"], []).append(
            _MatchedFrame(row, pose)
        )
    return association


def _coverage(association: _Association, processed: int) -> dict[str, Any]:
    """Frame counts and fractions, in the saved summary key order."""
    tracked, matched = association.tracked, association.matched
    available = association.reference_available
    return {
        "processed_frames": processed,
        "tracked_frames": tracked,
        "reference_available_frames": available,
        "matched_tracking_frames": matched,
        "tracking_frame_fraction": tracked / processed,
        "matched_tracking_fraction_of_reference_frames": (
            matched / available if available else None
        ),
    }


def _score_segment(
    segment: int, samples: list[_MatchedFrame]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fit this segment's Sim(3) and measure errors after alignment.

    Returns the segment record and its per-frame `aligned.jsonl` rows; a
    segment that cannot be aligned is reported with the reason and no rows.
    """
    import numpy as np

    estimated = np.asarray([s.row["camera_center_arbitrary"] for s in samples])
    reference = np.asarray([s.reference[:3, 3] for s in samples])
    record: dict[str, Any] = {
        "segment": segment,
        "matched_frames": len(samples),
        "span_s": (samples[-1].row["source_time_ns"] - samples[0].row["source_time_ns"])
        / 1e9,
    }
    try:
        alignment = similarity_alignment(estimated, reference)
    except ValueError as exc:
        record.update(alignment_status="unavailable", reason=str(exc))
        return record, []
    aligned = alignment.apply(estimated)
    errors = np.linalg.norm(aligned - reference, axis=1)
    angles = _rotation_errors_deg(samples, alignment.rotation)
    record.update(
        alignment_status="aligned",
        scale=alignment.scale,
        rotation=alignment.rotation.tolist(),
        translation=alignment.translation.tolist(),
        position_rmse_m=float(np.sqrt(np.mean(errors**2))),
        position_median_m=float(np.median(errors)),
        position_max_m=float(errors.max()),
        rotation_rmse_deg=float(np.sqrt(np.mean(np.square(angles)))),
    )
    rows = [
        {
            "source_time_ns": sample.row["source_time_ns"],
            "segment": segment,
            "estimated_aligned": position.tolist(),
            "reference": sample.reference[:3, 3].tolist(),
            "position_error_m": float(error),
        }
        for sample, position, error in zip(samples, aligned, errors)
    ]
    return record, rows


def _rotation_errors_deg(
    samples: list[_MatchedFrame], alignment_rotation: Array
) -> list[float]:
    """Angle between each aligned estimated and reference camera orientation.

    VO saves world-to-camera [R | t], so R.T rotates camera to segment world;
    the alignment rotation then maps segment world to reference world, and the
    residual R_ref.T R_align R.T is the orientation error.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation

    angles = []
    for sample in samples:
        world_to_camera = np.asarray(sample.row["world_to_camera"], dtype=float)
        camera_rotation = world_to_camera[:, :3].T
        residual = sample.reference[:3, :3].T @ alignment_rotation @ camera_rotation
        angles.append(float(Rotation.from_matrix(residual).magnitude() * 180 / np.pi))
    return angles


def _write_outputs(
    output: Path,
    summary: dict[str, Any],
    plot_rows: list[dict[str, Any]],
    reference: ReferenceTrajectory,
) -> None:
    """Create `output` and write the summary, aligned rows, report and plot."""
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    (output / "aligned.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in plot_rows)
    )
    lines = _report_lines(summary)
    if plot_rows:
        segment_order = [record["segment"] for record in summary["segments"]]
        _plot_trajectories(
            output / "trajectory.svg",
            reference.camera_positions(),
            plot_rows,
            segment_order,
        )
        lines += ["", "![Segment trajectories and aligned errors](trajectory.svg)"]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def _report_lines(summary: dict[str, Any]) -> list[str]:
    """Markdown report: caveats, coverage, continuity and a per-segment table."""
    continuity = summary["continuity"]
    lines = [
        "# Ground-truth evaluation",
        "",
        summary["limits"],
        "",
        f"Tracked: {summary['tracked_frames']}/{summary['processed_frames']} processed frames. Matched tracking: {summary['matched_tracking_frames']}/{summary['reference_available_frames']} reference-eligible frames.",
        f"Tracking losses: {continuity['tracking_losses']}. Segments: {continuity['segments']}.",
        "",
        "| Segment | Matched frames | Span (s) | Aligned position RMSE (m) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for s in summary["segments"]:
        value = (
            f"{s['position_rmse_m']:.4f}"
            if s["alignment_status"] == "aligned"
            else s["reason"]
        )
        lines.append(
            f"| {s['segment']} | {s['matched_frames']} | {s['span_s']:.3f} | {value} |"
        )
    return lines


def _plot_trajectories(
    path: Path,
    reference_positions: Array,
    plot_rows: list[dict[str, Any]],
    segment_order: list[int],
) -> None:
    """3D aligned segment paths over the reference, and error against time."""
    import numpy as np
    from matplotlib.backends.backend_svg import FigureCanvasSVG
    from matplotlib.figure import Figure

    figure = Figure(figsize=(10, 5))
    axes = figure.add_subplot(121, projection="3d")
    error_axes = figure.add_subplot(122)
    axes.plot(
        *reference_positions.T, color="black", alpha=0.4, label="Reference camera"
    )
    origin = min(row["source_time_ns"] for row in plot_rows)
    for segment in segment_order:
        selected = [row for row in plot_rows if row["segment"] == segment]
        if not selected:
            continue
        positions = np.asarray([row["estimated_aligned"] for row in selected])
        (line,) = axes.plot(*positions.T, label=f"Segment {segment}")
        error_axes.plot(
            [(row["source_time_ns"] - origin) / 1e9 for row in selected],
            [row["position_error_m"] for row in selected],
            color=line.get_color(),
        )
    axes.set(
        xlabel="X (m)",
        ylabel="Y (m)",
        zlabel="Z (m)",
        title="Independent segment alignments",
    )
    error_axes.set(
        xlabel="Time since first match (s)",
        ylabel="Position error (m)",
        title="Error after per-segment Sim(3) fit",
    )
    figure.tight_layout()
    FigureCanvasSVG(figure).print_svg(str(path))
