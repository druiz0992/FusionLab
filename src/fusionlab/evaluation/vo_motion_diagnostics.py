"""Offline visual-pose/CV/gyro diagnostics. No estimator acceptance changes."""

import bisect
import html
import json
import shutil
from pathlib import Path

from fusionlab.vision.tracks import TrackFrame


def rotation_angle(rotation):
    import numpy as np

    return float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))))


def predict_constant_velocity(history, timestamp_ns):
    """Extrapolate camera center and SO(3) motion using source-time intervals.

    history contains two accepted states from ONE contiguous segment. Calling
    code resets it at loss, initialization, source-time discontinuity or stream
    changes. Predictions never feed back into the estimator.
    """
    import cv2
    import numpy as np

    if len(history) < 2:
        return None
    first, last = history[-2:]
    if first.get("segment") != last.get("segment"):
        return None
    dt = (last["time"] - first["time"]) / 1e9
    horizon = (timestamp_ns - last["time"]) / 1e9
    if dt <= 0 or horizon <= 0 or max(dt, horizon) > 0.5:
        return None
    ratio = horizon / dt
    poses = [np.asarray(state["pose"]) for state in (first, last)]
    centers = [-pose[:, :3].T @ pose[:, 3] for pose in poses]
    center = centers[1] + (centers[1] - centers[0]) * ratio
    increment = cv2.Rodrigues(poses[1][:, :3] @ poses[0][:, :3].T)[0]
    rotation = cv2.Rodrigues(increment * ratio)[0] @ poses[1][:, :3]
    return np.column_stack((rotation, -rotation @ center))


def spatial_distribution(pixels, size):
    """Coverage and clustering in original tracking pixels, independent of pose."""
    import cv2
    import numpy as np

    points = np.asarray(pixels, dtype=float).reshape(-1, 2)
    width, height = size
    if not len(points):
        return {
            "points": 0,
            "occupied_grid_cells": 0,
            "hull_fraction": 0.0,
            "pca_minor_major_ratio": None,
        }
    cells = np.floor(points / np.array([width, height]) * 4).astype(int)
    cells = np.clip(cells, 0, 3)
    hull = (
        float(cv2.contourArea(cv2.convexHull(points.astype("float32"))))
        / (width * height)
        if len(points) >= 3
        else 0.0
    )
    eigenvalues = (
        np.linalg.eigvalsh(np.cov((points / np.array([width, height])).T))
        if len(points) >= 3
        else [0.0, 0.0]
    )
    return {
        "points": len(points),
        "occupied_grid_cells": len(set(map(tuple, cells))),
        "hull_fraction": hull,
        "pca_minor_major_ratio": (
            float(eigenvalues[0] / eigenvalues[-1]) if eigenvalues[-1] > 1e-15 else None
        ),
    }


def evaluate_pose(pose, snapshot, pixels, size):
    """Identical pre-update landmarks and gates for every pose hypothesis."""
    import numpy as np

    camera = snapshot["objects"] @ pose[:, :3].T + pose[:, 3]
    valid = (camera[:, 2] > 1e-8) & np.isfinite(camera).all(axis=1)
    focal = float((snapshot["camera"][0, 0] + snapshot["camera"][1, 1]) / 2)
    errors = np.full(len(camera), np.inf)
    errors[valid] = (
        np.linalg.norm(
            camera[valid, :2] / camera[valid, 2, None] - snapshot["normalized"][valid],
            axis=1,
        )
        * focal
    )
    inliers = valid & (errors <= 3.0)
    finite = errors[np.isfinite(errors)]
    depths = camera[inliers, 2]
    return {
        "candidates": len(camera),
        "positive_depth": int(valid.sum()),
        "inliers_3px": int(inliers.sum()),
        "inlier_ratio": float(inliers.mean()) if len(inliers) else 0.0,
        "median_error_px": float(np.median(finite)) if len(finite) else None,
        "p90_error_px": float(np.percentile(finite, 90)) if len(finite) else None,
        "inlier_median_error_px": (
            float(np.median(errors[inliers])) if inliers.any() else None
        ),
        "inlier_spatial": spatial_distribution(pixels[inliers], size),
        "inlier_depth_quantiles_arbitrary": (
            np.percentile(depths, [10, 50, 90]).tolist() if len(depths) else None
        ),
    }


def gyro_interval(
    samples,
    start_ns,
    end_ns,
    *,
    camera_from_imu=None,
    visual_rotation=None,
    offset_ns=0,
):
    """Midpoint gyro integration with interpolated endpoints on source timestamps.

    Offset is added to camera times to index IMU time. Without extrinsics, only
    rotation magnitudes can be compared. Exact clock alignment is NOT established.
    """
    import cv2
    import numpy as np

    start_ns += offset_ns
    end_ns += offset_ns
    if end_ns <= start_ns:
        return {"status": "invalid_interval"}
    times = [sample["source_time_ns"] for sample in samples]
    if any(b <= a for a, b in zip(times, times[1:])):
        return {"status": "imu_clock_not_monotonic"}
    left = bisect.bisect_right(times, start_ns) - 1
    right = bisect.bisect_left(times, end_ns)
    if left < 0 or right >= len(times):
        return {"status": "no_bracketing_imu"}
    rows = samples[left : right + 1]
    intervals = np.diff([sample["source_time_ns"] for sample in rows]) / 1e9
    if len(rows) < 2 or (intervals <= 0).any() or (intervals > 0.05).any():
        return {"status": "imu_gap_or_regression"}
    relative_times = np.array(
        [row["source_time_ns"] - start_ns for row in rows], dtype=float
    )
    rates = np.array([row["angular_velocity"] for row in rows], dtype=float)
    if not np.isfinite(rates).all():
        return {"status": "nonfinite_gyro"}
    knots = (
        [0]
        + [int(t) for t in relative_times if 0 < t < end_ns - start_ns]
        + [end_ns - start_ns]
    )
    interpolated = np.column_stack(
        [np.interp(knots, relative_times, rates[:, axis]) for axis in range(3)]
    )
    body_start_from_end = np.eye(3)
    for i, dt in enumerate(np.diff(knots) / 1e9):
        body_start_from_end = (
            body_start_from_end
            @ cv2.Rodrigues(0.5 * (interpolated[i] + interpolated[i + 1]) * dt)[0]
        )
    report = {
        "status": "available_assuming_source_clock_alignment",
        "gyro_angle_deg": rotation_angle(body_start_from_end),
        "interval_s": (end_ns - start_ns) / 1e9,
        "camera_to_imu_offset_ns": offset_ns,
        "comparison": "angle magnitude only; camera–IMU extrinsics unavailable",
    }
    if visual_rotation is not None:
        report["visual_angle_deg"] = rotation_angle(visual_rotation)
        report["angle_magnitude_difference_deg"] = abs(
            report["visual_angle_deg"] - report["gyro_angle_deg"]
        )
        if camera_from_imu is not None:
            expected = camera_from_imu @ body_start_from_end.T @ camera_from_imu.T
            report["axis_aware_rotation_error_deg"] = rotation_angle(
                visual_rotation @ expected.T
            )
            report["comparison"] = (
                "Full relative rotation using explicitly supplied camera-from-IMU rotation"
            )
    return report


def diagnose_run(run_dir, capture, output, *, camera_from_imu=None, imu_offset_ms=0.0):
    """Reproduce a saved estimator run to access pre-update landmark snapshots."""
    import cv2
    import numpy as np
    from fusionlab.vision.odometry.visual_odometry import MonocularOdometry
    from fusionlab.evaluation.vo_evaluation import inspect_translation_jumps

    if not np.isfinite(imu_offset_ms):
        raise ValueError("IMU offset must be finite")
    if camera_from_imu is not None:
        camera_from_imu = np.asarray(camera_from_imu, dtype=float)
        if (
            camera_from_imu.shape != (3, 3)
            or not np.isfinite(camera_from_imu).all()
            or not np.allclose(
                camera_from_imu.T @ camera_from_imu, np.eye(3), atol=1e-6
            )
            or not np.isclose(np.linalg.det(camera_from_imu), 1.0, atol=1e-6)
        ):
            raise ValueError("camera-from-IMU must be a proper 3x3 rotation")
    config = json.loads((run_dir / "summary.json").read_text())
    tracks = Path(config["tracks"])
    calibration = Path(config["calibration"])
    tracker = MonocularOdometry(
        json.loads(calibration.read_text()),
        pose_acceptance=config.get("pose_acceptance", "baseline"),
        keyframe_policy=config.get("keyframe_policy", "baseline"),
        landmark_policy=config.get("landmark_policy", "baseline"),
        bundle_adjustment=config.get("bundle_adjustment", False),
    )
    cv2.setRNGSeed(config.get("ransac_seed", 0))
    with (run_dir / "poses.jsonl").open() as file:
        saved = {r["frame_index"]: r for line in file if (r := json.loads(line))}
    with (capture / "events.jsonl").open() as file:
        events = [json.loads(line) for line in file]
    imu = {}
    cameras = {}
    for event in events:
        if event["type"] == "ImuSample":
            imu.setdefault((event["session_id"], event["stream_id"]), []).append(event)
        elif event["type"] == "CameraSample":
            cameras[
                (
                    event["stream_id"],
                    event["source_time_ns"],
                    event["received_monotonic_ns"],
                )
            ] = event["session_id"]
    output.mkdir(parents=True, exist_ok=False)
    history = []
    last_geometry = None
    rows = []
    mismatches = []
    origin = None
    with tracks.open() as file, (output / "frames.jsonl").open("x") as report_file:
        for index, line in enumerate(file):
            if index not in saved:
                continue
            row = TrackFrame.from_json(json.loads(line))
            geometry = row.geometry_key
            if geometry != last_geometry:
                history = []
            last_geometry = geometry
            result = tracker.process(row)
            old = saved[index]
            if (
                result.status != old["status"]
                or result.segment != old["segment"]
                or (
                    result.status == "tracking"
                    and not np.allclose(
                        result.world_to_camera,
                        old["world_to_camera"],
                        atol=1e-8,
                        rtol=1e-8,
                    )
                )
            ):
                mismatches.append(index)
            if origin is None:
                origin = row.received_monotonic_ns
            item = {
                "frame_index": index,
                "status": result.status,
                "segment": result.segment,
                "receive_elapsed_s": (row.received_monotonic_ns - origin) / 1e9,
                "source_time_ns": row.source_time_ns,
                "failure_category": result.failure_category,
                "overlay_path": row.overlay_path,
            }
            if history and (
                history[-1]["segment"] != result.segment
                or not 0 < row.source_time_ns - history[-1]["time"] <= 500_000_000
            ):
                history = []
            snapshot = tracker.last_pnp_snapshot
            candidate = (
                np.asarray(result.world_to_camera)
                if result.status == "tracking"
                else (snapshot.get("candidate_pose") if snapshot else None)
            )
            prediction = predict_constant_velocity(history, row.source_time_ns)
            if snapshot is not None:
                lookup = row.positions()
                pixels = np.array(
                    [lookup[key] for key in snapshot["ids"]], dtype=float
                ).reshape(-1, 2)
                item["candidate_spatial"] = spatial_distribution(
                    pixels, row.tracking_size
                )
                item["map_basis"] = (
                    "Pre-update 3D landmarks, shared by all hypotheses. Accepted online poses may include BA corrections; these support counts need not equal the estimator acceptance counts."
                )
                if candidate is not None:
                    item["pnp"] = evaluate_pose(
                        candidate, snapshot, pixels, row.tracking_size
                    )
                    item["pnp_pose_kind"] = (
                        "accepted online pose after optional BA"
                        if result.status == "tracking"
                        else "rejected raw RANSAC candidate"
                    )
                if prediction is not None:
                    item["constant_velocity"] = evaluate_pose(
                        prediction, snapshot, pixels, row.tracking_size
                    )
                    item["predicted_center_arbitrary"] = (
                        -prediction[:, :3].T @ prediction[:, 3]
                    ).tolist()
                    if candidate is not None:
                        center = -candidate[:, :3].T @ candidate[:, 3]
                        innovation = float(
                            np.linalg.norm(
                                center - np.asarray(item["predicted_center_arbitrary"])
                            )
                        )
                        last_center = (
                            -history[-1]["pose"][:, :3].T @ history[-1]["pose"][:, 3]
                        )
                        expected_step = float(
                            np.linalg.norm(
                                np.asarray(item["predicted_center_arbitrary"])
                                - last_center
                            )
                        )
                        item["translation_innovation_arbitrary"] = innovation
                        item["innovation_over_predicted_step"] = (
                            innovation / expected_step if expected_step > 1e-6 else None
                        )
                        item["predicted_step_arbitrary"] = expected_step
                        item["rotation_prediction_error_deg"] = rotation_angle(
                            candidate[:, :3] @ prediction[:, :3].T
                        )
                        # Isolate translation: hold the PnP rotation fixed for both.
                        translation_only = np.column_stack(
                            (
                                candidate[:, :3],
                                -candidate[:, :3]
                                @ np.asarray(item["predicted_center_arbitrary"]),
                            )
                        )
                        item["cv_translation_with_pnp_rotation"] = evaluate_pose(
                            translation_only, snapshot, pixels, row.tracking_size
                        )
            if history and candidate is not None:
                session = cameras.get(
                    (
                        row.stream_id,
                        row.source_time_ns,
                        row.received_monotonic_ns,
                    )
                )
                streams = [
                    samples for (sid, _), samples in imu.items() if sid == session
                ]
                visual = candidate[:, :3] @ history[-1]["pose"][:, :3].T
                item["gyro"] = (
                    gyro_interval(
                        streams[0],
                        history[-1]["time"],
                        row.source_time_ns,
                        camera_from_imu=camera_from_imu,
                        visual_rotation=visual,
                        offset_ns=round(imu_offset_ms * 1e6),
                    )
                    if len(streams) == 1
                    else {"status": "missing_or_ambiguous_session_imu"}
                )
            report_file.write(json.dumps(item, allow_nan=False) + "\n")
            rows.append(item)
            if result.status == "tracking":
                history.append(
                    {
                        "pose": candidate.copy(),
                        "time": row.source_time_ns,
                        "segment": result.segment,
                    }
                )
                history = history[-2:]
            else:
                history = []
            if len(rows) % 100 == 0:
                print(f"Diagnosed {len(rows)}/{len(saved)} frames", flush=True)
    # Rank CV innovations, but separately flag old step ratios whose denominator
    # can be small during stops. Neither is treated as ground truth.
    innovations = sorted(
        [r for r in rows if r.get("innovation_over_predicted_step") is not None],
        key=lambda r: r["innovation_over_predicted_step"],
        reverse=True,
    )
    losses = [r for r in rows if r["status"] == "tracking_lost"]
    summary = {
        "frames": len(rows),
        "reproduced_saved_run": not mismatches,
        "mismatch_frames": mismatches,
        "losses": len(losses),
        "frames_with_cv": sum("constant_velocity" in r for r in rows),
        "gyro_intervals": sum(
            r.get("gyro", {}).get("status")
            == "available_assuming_source_clock_alignment"
            for r in rows
        ),
        "camera_from_imu_supplied": camera_from_imu is not None,
        "imu_offset_ms": imu_offset_ms,
        "limits": "Diagnostic only; no acceptance changes. Constant velocity is a hypothesis. Shared pre-update landmarks make pose comparisons conditional on map accuracy. CV uses online past poses, not retroactively smoothed history. Source-clock alignment/offset and gyro units rad/s are assumptions. Without extrinsics, gyro comparisons are angle magnitudes only. No exposure timing, bias or rolling-shutter correction.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "ranked_events.json").write_text(
        json.dumps(
            {
                "translation_innovations": innovations[:20],
                "tracking_losses": losses,
                "segment_step_ratio_flags": inspect_translation_jumps(
                    run_dir, limit=20
                ),
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    _write_html(output, tracks.parent, innovations[:10], losses, summary)
    write_findings(output)
    return summary


def _write_html(output, tracks_root, jumps, losses, summary):
    """Self-contained index with local frame images and exact replay timestamps."""
    (output / "frames").mkdir(exist_ok=True)
    body = [
        '<!doctype html><meta charset="utf-8"><title>VO motion diagnostics</title>',
        "<style>body{font:16px sans-serif;max-width:1100px;margin:30px auto}img{max-width:280px}pre{white-space:pre-wrap}article{border-top:1px solid #aaa;padding:15px}</style>",
        "<h1>VO motion diagnostics</h1><p>Use receive times to seek the existing replay. These are inspection flags, not proof of pose errors.</p>",
        "<pre>" + html.escape(json.dumps(summary, indent=2)) + "</pre>",
    ]
    for title, events in [
        ("Largest CV innovations", jumps),
        ("Tracking losses", losses),
    ]:
        body.append("<h2>" + title + "</h2>")
        for item in events:
            body.append(
                f"<article><h3>Frame {item['frame_index']} · {item['receive_elapsed_s']:.2f} s · segment {item['segment']}</h3>"
            )
            if item.get("overlay_path"):
                source = (tracks_root / item["overlay_path"]).resolve()
                if not source.is_relative_to(tracks_root.resolve()):
                    raise ValueError("Overlay escapes tracking folder")
                if source.exists():
                    target = f"frames/{item['frame_index']:06d}.jpg"
                    shutil.copyfile(source, output / target)
                    body.append(f'<img src="{target}" alt="Tracked camera frame">')
            body.append(
                '<table border="1" cellpadding="5"><tr><th>Hypothesis</th><th>Inliers / candidates</th><th>Median error (px)</th><th>Coverage</th><th>Grid cells / 16</th></tr>'
            )
            for key, label in (
                ("pnp", "Visual pose"),
                ("constant_velocity", "Constant velocity"),
                (
                    "cv_translation_with_pnp_rotation",
                    "CV translation + visual rotation",
                ),
            ):
                metrics = item.get(key)
                if metrics is None:
                    continue
                spatial = metrics["inlier_spatial"]
                error = metrics["median_error_px"]
                error_text = f"{error:.2f}" if error is not None else "unavailable"
                body.append(
                    f"<tr><td>{label}</td><td>{metrics['inliers_3px']} / {metrics['candidates']}</td><td>{error_text}</td><td>{100*spatial['hull_fraction']:.1f}%</td><td>{spatial['occupied_grid_cells']}</td></tr>"
                )
            body.append("</table>")
            gyro = item.get("gyro", {})
            if "angle_magnitude_difference_deg" in gyro:
                body.append(
                    f"<p>Visual rotation {gyro['visual_angle_deg']:.2f}°; gyro rotation {gyro['gyro_angle_deg']:.2f}°. Magnitudes only; source-time alignment assumed.</p>"
                )
            body.append(
                "<details><summary>Full diagnostic record</summary><pre>"
                + html.escape(json.dumps(item, indent=2))
                + "</pre></details></article>"
            )
    (output / "index.html").write_text("\n".join(body))


def write_findings(output):
    """Summarize observed disagreements; predictions are not ground truth."""
    from statistics import median

    with (output / "frames.jsonl").open() as file:
        rows = [json.loads(line) for line in file]
    compared = [r for r in rows if "pnp" in r and "constant_velocity" in r]
    gyro = [
        r
        for r in rows
        if r.get("gyro", {}).get("angle_magnitude_difference_deg") is not None
    ]
    disagreements = sorted(
        gyro, key=lambda r: r["gyro"]["angle_magnitude_difference_deg"], reverse=True
    )
    losses = [r for r in rows if r["status"] == "tracking_lost"]
    recovered = [
        r
        for r in losses
        if r.get("constant_velocity", {}).get("inliers_3px", 0) >= 20
        and r["constant_velocity"]["inlier_ratio"] >= 0.5
    ]
    findings = {
        "compared_pose_frames": len(compared),
        "visual_more_inliers": sum(
            r["pnp"]["inliers_3px"] > r["constant_velocity"]["inliers_3px"]
            for r in compared
        ),
        "cv_more_inliers": sum(
            r["pnp"]["inliers_3px"] < r["constant_velocity"]["inliers_3px"]
            for r in compared
        ),
        "losses_with_cv": sum("constant_velocity" in r for r in losses),
        "losses_where_cv_passes_geometric_gates": len(recovered),
        "median_gyro_visual_angle_difference_deg": (
            median(r["gyro"]["angle_magnitude_difference_deg"] for r in gyro)
            if gyro
            else None
        ),
        "largest_gyro_visual_angle_disagreements": [
            {
                "frame_index": r["frame_index"],
                "receive_elapsed_s": r["receive_elapsed_s"],
                "status": r["status"],
                **r["gyro"],
            }
            for r in disagreements[:10]
        ],
        "interpretation": "Visual poses fit these visual landmarks by construction, so this is not an independent accuracy comparison. CV failures can reflect genuine acceleration or incorrect previous poses. Gyro magnitude differences assume aligned source clocks; magnitude agreement cannot verify rotation axes.",
    }
    (output / "findings.json").write_text(
        json.dumps(findings, indent=2, allow_nan=False) + "\n"
    )
    lines = [
        "# Visual motion diagnostics findings",
        "",
        findings["interpretation"],
        "",
        f"Comparable visual/CV frames: {len(compared)}.",
        f"Visual pose has more geometric inliers: {findings['visual_more_inliers']} frames; CV has more: {findings['cv_more_inliers']}.",
        f"Tracking losses with CV available: {findings['losses_with_cv']}; CV passing the same geometric gates: {len(recovered)}.",
        f"Median visual/gyro rotation-magnitude difference: {findings['median_gyro_visual_angle_difference_deg']} degrees.",
        "",
        "Inspect `index.html` for ranked innovations and every tracking loss, with matching overlay images. `frames.jsonl` preserves every frame; `ranked_events.json` includes segment-normalized step flags. Spatial coverage uses convex-hull image fraction, a 4x4 grid and normalized-coordinate PCA minor/major eigenvalue ratio. Depth quantiles have independent arbitrary scale per segment.",
        "",
        "No constant-velocity or gyro constraints were added to estimation. A low-error visual fit does not establish a correct trajectory. Prioritize events with weak spatial support and gyro disagreement, then test timing/extrinsics before treating IMU as a rotation acceptance gate.",
    ]
    (output / "findings.md").write_text("\n".join(lines) + "\n")
    try:
        import numpy as np
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError:
        return findings
    figure = Figure(figsize=(12, 13))
    FigureCanvasAgg(figure)
    axes = figure.subplots(5, 1, sharex=True)
    times = [r["receive_elapsed_s"] for r in rows]
    for key, label in (("pnp", "Visual pose"), ("constant_velocity", "CV")):
        for axis, metric in ((axes[0], "inliers_3px"), (axes[1], "median_error_px")):
            axis.plot(
                times,
                [
                    (
                        r.get(key, {}).get(metric, np.nan)
                        if r.get(key, {}).get(metric) is not None
                        else np.nan
                    )
                    for r in rows
                ],
                label=label,
            )
    axes[1].set_yscale("symlog", linthresh=1.0)
    axes[2].plot(
        times,
        [
            100
            * r.get("pnp", {}).get("inlier_spatial", {}).get("hull_fraction", np.nan)
            for r in rows
        ],
    )
    axes[3].plot(
        times,
        [r.get("gyro", {}).get("visual_angle_deg", np.nan) for r in rows],
        label="Visual",
    )
    axes[3].plot(
        times,
        [r.get("gyro", {}).get("gyro_angle_deg", np.nan) for r in rows],
        label="Gyro (assumed timing)",
    )
    axes[4].plot(times, [r.get("innovation_over_predicted_step", np.nan) for r in rows])
    axes[4].set_yscale("symlog", linthresh=1.0)
    for axis, label in zip(
        axes,
        (
            "3 px inlier count",
            "Median error (px)",
            "Visual inlier hull (%)",
            "Relative rotation (deg)",
            "Innovation / CV step",
        ),
    ):
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)
        for row in losses:
            axis.axvline(row["receive_elapsed_s"], color="red", alpha=0.12)
    for axis in (axes[0], axes[1], axes[3]):
        axis.legend()
    axes[-1].set_xlabel("Receive-time playback seconds; red markers = tracking losses")
    figure.suptitle("Diagnostic comparisons only — no prediction or gyro fusion")
    figure.tight_layout()
    figure.savefig(output / "motion_checks.png", dpi=140)
    return findings
