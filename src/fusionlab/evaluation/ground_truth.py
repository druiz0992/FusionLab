"""Timestamp-associated, per-segment monocular evaluation against EuRoC poses."""

import json
from pathlib import Path


def similarity_alignment(estimated, reference):
    """Least-squares proper Sim(3), with positive scale and noncollinear support."""
    import numpy as np

    x, y = np.asarray(estimated, dtype=float), np.asarray(reference, dtype=float)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3 or len(x) < 3:
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
    return scale, rotation, translation


def evaluate_run(run_dir, groundtruth, output, *, max_gap_ms=50.0):
    import bisect
    import numpy as np
    from scipy.spatial.transform import Rotation, Slerp
    from fusionlab.recordings.euroc import rigid_transform
    from fusionlab.evaluation.vo_evaluation import summarize_run

    if not np.isfinite(max_gap_ms) or max_gap_ms <= 0:
        raise ValueError("max_gap_ms must be positive and finite")
    run_dir, groundtruth, output = map(Path, (run_dir, groundtruth, output))
    truth = json.loads(groundtruth.read_text())
    stamps = [int(s["source_time_ns"]) for s in truth["states"]]
    if len(stamps) < 2 or any(b <= a for a, b in zip(stamps, stamps[1:])):
        raise ValueError("Ground-truth timestamps must increase strictly")
    transforms = np.asarray(
        [rigid_transform(s["world_from_body"]) for s in truth["states"]]
    )
    body_from_camera = rigid_transform(truth["body_from_camera"])
    # Subtract integer epoch before conversion to seconds to retain ns precision.
    times = np.array([(s - stamps[0]) / 1e9 for s in stamps])
    rotations = Slerp(times, Rotation.from_matrix(transforms[:, :3, :3]))

    def reference(stamp):
        index = bisect.bisect_left(stamps, stamp)
        if index < len(stamps) and stamps[index] == stamp:
            return transforms[index] @ body_from_camera
        if index == 0 or index == len(stamps):
            return None
        a, b = index - 1, index
        if stamps[b] - stamps[a] > max_gap_ms * 1e6:
            return None
        fraction = (stamp - stamps[a]) / (stamps[b] - stamps[a])
        pose = np.eye(4)
        pose[:3, :3] = rotations([(stamp - stamps[0]) / 1e9]).as_matrix()[0]
        pose[:3, 3] = (1 - fraction) * transforms[a, :3, 3] + fraction * transforms[
            b, :3, 3
        ]
        return pose @ body_from_camera

    rows = [
        json.loads(line)
        for line in (run_dir / "poses.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("Empty estimated trajectory")
    groups = {}
    eligible = matched = tracked = 0
    for row in rows:
        pose = reference(int(row["source_time_ns"]))
        eligible += pose is not None
        if row["status"] != "tracking":
            continue
        tracked += 1
        if pose is None:
            continue
        matched += 1
        groups.setdefault(row["segment"], []).append((row, pose))
    segments = []
    plot_rows = []
    for segment, samples in groups.items():
        x = np.asarray([s[0]["camera_center_arbitrary"] for s in samples])
        y = np.asarray([s[1][:3, 3] for s in samples])
        record = dict(
            segment=segment,
            matched_frames=len(samples),
            span_s=(samples[-1][0]["source_time_ns"] - samples[0][0]["source_time_ns"])
            / 1e9,
        )
        try:
            scale, rotation, translation = similarity_alignment(x, y)
        except ValueError as exc:
            record.update(alignment_status="unavailable", reason=str(exc))
            segments.append(record)
            continue
        aligned = scale * (x @ rotation.T) + translation
        errors = np.linalg.norm(aligned - y, axis=1)
        angles = []
        for row, gt in samples:
            camera_rotation = np.asarray(row["world_to_camera"]).reshape(3, 3).T
            angles.append(
                float(
                    Rotation.from_matrix(
                        gt[:3, :3].T @ rotation @ camera_rotation
                    ).magnitude()
                    * 180
                    / np.pi
                )
            )
        record.update(
            alignment_status="aligned",
            scale=scale,
            rotation=rotation.tolist(),
            translation=translation.tolist(),
            position_rmse_m=float(np.sqrt(np.mean(errors**2))),
            position_median_m=float(np.median(errors)),
            position_max_m=float(errors.max()),
            rotation_rmse_deg=float(np.sqrt(np.mean(np.square(angles)))),
        )
        segments.append(record)
        for (row, gt), position, error in zip(samples, aligned, errors):
            plot_rows.append(
                dict(
                    source_time_ns=row["source_time_ns"],
                    segment=segment,
                    estimated_aligned=position.tolist(),
                    reference=gt[:3, 3].tolist(),
                    position_error_m=float(error),
                )
            )
    summary = dict(
        run=str(run_dir.resolve()),
        groundtruth=str(groundtruth.resolve()),
        continuity=summarize_run(run_dir),
        processed_frames=len(rows),
        tracked_frames=tracked,
        reference_available_frames=eligible,
        matched_tracking_frames=matched,
        tracking_frame_fraction=tracked / len(rows),
        matched_tracking_fraction_of_reference_frames=(
            matched / eligible if eligible else None
        ),
        max_reference_gap_ms=max_gap_ms,
        segments=segments,
        alignment="Independent positive-scale Sim(3) per segment; fitted on scored samples",
        limits="Per-segment fit measures local consistency, not whole-route drift or metric scale recovery. Short segments may fit well despite fragmentation. No extrapolation across missing ground truth. Frame fractions describe processed frames, not time coverage.",
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    (output / "aligned.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in plot_rows)
    )
    lines = [
        "# Ground-truth evaluation",
        "",
        summary["limits"],
        "",
        f"Tracked: {tracked}/{len(rows)} processed frames. Matched tracking: {matched}/{eligible} reference-eligible frames.",
        f"Tracking losses: {summary['continuity']['tracking_losses']}. Segments: {summary['continuity']['segments']}.",
        "",
        "| Segment | Matched frames | Span (s) | Aligned position RMSE (m) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for s in segments:
        value = (
            f"{s['position_rmse_m']:.4f}"
            if s["alignment_status"] == "aligned"
            else s["reason"]
        )
        lines.append(
            f"| {s['segment']} | {s['matched_frames']} | {s['span_s']:.3f} | {value} |"
        )
    if plot_rows:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_svg import FigureCanvasSVG

        figure = Figure(figsize=(10, 5))
        axes = figure.add_subplot(121, projection="3d")
        error_axes = figure.add_subplot(122)
        reference_positions = (transforms @ body_from_camera)[:, :3, 3]
        axes.plot(
            *reference_positions.T, color="black", alpha=0.4, label="Reference camera"
        )
        origin = min(row["source_time_ns"] for row in plot_rows)
        for segment in groups:
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
        FigureCanvasSVG(figure).print_svg(str(output / "trajectory.svg"))
        lines += ["", "![Segment trajectories and aligned errors](trajectory.svg)"]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return summary
