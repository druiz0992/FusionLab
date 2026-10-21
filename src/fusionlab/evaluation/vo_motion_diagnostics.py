"""Offline visual-pose/CV/gyro diagnostics. No estimator acceptance changes.

`diagnose_run` replays a saved VO run with the same settings, checks that it
reproduces the saved poses, and at every frame evaluates competing pose
hypotheses on the SAME pre-update landmark snapshot: the visual (PnP) pose, a
constant-velocity (CV) prediction from the two previous accepted poses, and CV
translation with the visual rotation. Where IMU data exist, the visual rotation
between accepted frames is compared with integrated gyro rotation. Poses are
world-to-camera in each segment's arbitrary-scale frame.
"""

from __future__ import annotations

import bisect
import html
import json
import shutil
from pathlib import Path
from typing import Any

from fusionlab.vision.tracks import TrackFrame

# Method caveats saved with every report (what the numbers do not show).
DIAGNOSTIC_LIMITS = (
    "Diagnostic only; no acceptance changes. Constant velocity is a hypothesis. "
    "Shared pre-update landmarks make pose comparisons conditional on map accuracy. "
    "CV uses online past poses, not retroactively smoothed history. Source-clock "
    "alignment/offset and gyro units rad/s are assumptions. Without extrinsics, "
    "gyro comparisons are angle magnitudes only. No exposure timing, bias or "
    "rolling-shutter correction."
)

FINDINGS_INTERPRETATION = (
    "Visual poses fit these visual landmarks by construction, so this is not an "
    "independent accuracy comparison. CV failures can reflect genuine acceleration "
    "or incorrect previous poses. Gyro magnitude differences assume aligned source "
    "clocks; magnitude agreement cannot verify rotation axes."
)

FINDINGS_OUTPUT_GUIDE = (
    "Inspect `index.html` for ranked innovations and every tracking loss, with "
    "matching overlay images. `frames.jsonl` preserves every frame; "
    "`ranked_events.json` includes segment-normalized step flags. Spatial coverage "
    "uses convex-hull image fraction, a 4x4 grid and normalized-coordinate PCA "
    "minor/major eigenvalue ratio. Depth quantiles have independent arbitrary scale "
    "per segment."
)

FINDINGS_SCOPE = (
    "No constant-velocity or gyro constraints were added to estimation. A low-error "
    "visual fit does not establish a correct trajectory."
)


MAP_BASIS = (
    "Pre-update 3D landmarks, shared by all hypotheses. Accepted online poses may "
    "include BA corrections; these support counts need not equal the estimator "
    "acceptance counts."
)

# Same support gate as the estimator: 3 px, at least 20 inliers and 50%.
INLIER_THRESHOLD_PX = 3.0
MIN_SUPPORT_INLIERS = 20
MIN_SUPPORT_RATIO = 0.5
# Points must be at least this far in front of the camera to count.
MIN_DEPTH = 1e-8
# CV needs two accepted poses at most this far apart, extrapolated at most this far.
CV_MAX_INTERVAL_S = 0.5
# History is dropped across source-time steps longer than this (or non-positive).
HISTORY_MAX_GAP_NS = 500_000_000
# Gyro integration refuses IMU sample gaps longer than this.
MAX_IMU_GAP_S = 0.05


def rotation_angle(rotation) -> float:
    """Rotation magnitude in degrees, from trace(R) = 1 + 2 cos(theta)."""
    import numpy as np

    return float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))))


def predict_constant_velocity(history: list[dict[str, Any]], timestamp_ns: int):
    """Extrapolate camera center and SO(3) motion using source-time intervals.

    history contains two accepted states from ONE contiguous segment. Calling
    code resets it at loss, initialization, source-time discontinuity or stream
    changes. Predictions never feed back into the estimator. Returns a 3×4
    world-to-camera pose, or None when no valid prediction exists.
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
    if dt <= 0 or horizon <= 0 or max(dt, horizon) > CV_MAX_INTERVAL_S:
        return None
    ratio = horizon / dt
    poses = [np.asarray(state["pose"]) for state in (first, last)]
    centers = [-pose[:, :3].T @ pose[:, 3] for pose in poses]
    center = centers[1] + (centers[1] - centers[0]) * ratio
    # Scale the last rotation increment (as a rotation vector) to the horizon.
    increment = cv2.Rodrigues(poses[1][:, :3] @ poses[0][:, :3].T)[0]
    rotation = cv2.Rodrigues(increment * ratio)[0] @ poses[1][:, :3]
    return np.column_stack((rotation, -rotation @ center))


def spatial_distribution(pixels, size) -> dict[str, Any]:
    """Coverage and clustering in original tracking pixels, independent of pose.

    Reports occupied cells of a 4×4 grid, the convex hull as a fraction of the
    image, and the minor/major PCA eigenvalue ratio of normalized coordinates
    (near 0 means points lie along a line).
    """
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


def evaluate_pose(pose, snapshot: dict[str, Any], pixels, size) -> dict[str, Any]:
    """Identical pre-update landmarks and gates for every pose hypothesis.

    Reprojection errors are normalized-coordinate errors times the mean tracking
    focal length (approximate pixels); depth quantiles are in segment units.
    """
    import numpy as np

    camera = snapshot["objects"] @ pose[:, :3].T + pose[:, 3]
    valid = (camera[:, 2] > MIN_DEPTH) & np.isfinite(camera).all(axis=1)
    focal = float((snapshot["camera"][0, 0] + snapshot["camera"][1, 1]) / 2)
    errors = np.full(len(camera), np.inf)
    errors[valid] = (
        np.linalg.norm(
            camera[valid, :2] / camera[valid, 2, None] - snapshot["normalized"][valid],
            axis=1,
        )
        * focal
    )
    inliers = valid & (errors <= INLIER_THRESHOLD_PX)
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
    samples: list[dict[str, Any]],
    start_ns: int,
    end_ns: int,
    *,
    camera_from_imu=None,
    visual_rotation=None,
    offset_ns: int = 0,
) -> dict[str, Any]:
    """Midpoint gyro integration with interpolated endpoints on source timestamps.

    Offset is added to camera times to index IMU time. Without extrinsics, only
    rotation magnitudes can be compared. Exact clock alignment is NOT established.
    Gyro rates are assumed to be rad/s in the IMU body frame.
    """
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
    if len(rows) < 2 or (intervals <= 0).any() or (intervals > MAX_IMU_GAP_S).any():
        return {"status": "imu_gap_or_regression"}
    relative_times = np.array(
        [row["source_time_ns"] - start_ns for row in rows], dtype=float
    )
    rates = np.array([row["angular_velocity"] for row in rows], dtype=float)
    if not np.isfinite(rates).all():
        return {"status": "nonfinite_gyro"}
    body_start_from_end = _integrate_gyro(relative_times, rates, end_ns - start_ns)
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


def _integrate_gyro(relative_times, rates, duration_ns: int):
    """Compose midpoint rotations over the interval, with interpolated endpoints."""
    import cv2
    import numpy as np

    knots = (
        [0] + [int(t) for t in relative_times if 0 < t < duration_ns] + [duration_ns]
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
    return body_start_from_end


class _ReplayContext:
    """Inputs shared by every frame of a diagnostic replay."""

    def __init__(self, imu, camera_sessions, camera_from_imu, offset_ns: int):
        self.imu = imu  # (session, stream) -> IMU events
        self.camera_sessions = camera_sessions  # (stream, source, receive) -> session
        self.camera_from_imu = camera_from_imu  # 3×3 or None
        self.offset_ns = offset_ns


def diagnose_run(
    run_dir: Path,
    capture: Path,
    output: Path,
    *,
    camera_from_imu=None,
    imu_offset_ms: float = 0.0,
) -> dict[str, Any]:
    """Reproduce a saved estimator run to access pre-update landmark snapshots."""
    import numpy as np
    from fusionlab.evaluation.vo_evaluation import inspect_translation_jumps

    if not np.isfinite(imu_offset_ms):
        raise ValueError("IMU offset must be finite")
    camera_from_imu = _checked_camera_from_imu(camera_from_imu)
    config = json.loads((run_dir / "summary.json").read_text())
    tracks = Path(config["tracks"])
    tracker = _rebuild_estimator(config)
    saved = _saved_results(run_dir)
    imu, camera_sessions = _index_capture(capture)
    output.mkdir(parents=True, exist_ok=False)
    context = _ReplayContext(
        imu, camera_sessions, camera_from_imu, round(imu_offset_ms * 1e6)
    )
    rows, mismatches = _replay_frames(tracker, tracks, saved, output, context)
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
        "limits": DIAGNOSTIC_LIMITS,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    rankings = {
        "translation_innovations": innovations[:20],
        "tracking_losses": losses,
        "segment_step_ratio_flags": inspect_translation_jumps(run_dir, limit=20),
    }
    (output / "ranked_events.json").write_text(
        json.dumps(rankings, indent=2, allow_nan=False) + "\n"
    )
    _write_html(output, tracks.parent, innovations[:10], losses, summary)
    write_findings(output)
    return summary


def _checked_camera_from_imu(camera_from_imu):
    """A proper 3×3 rotation array, or None; never an assumed identity."""
    import numpy as np

    if camera_from_imu is None:
        return None
    camera_from_imu = np.asarray(camera_from_imu, dtype=float)
    if (
        camera_from_imu.shape != (3, 3)
        or not np.isfinite(camera_from_imu).all()
        or not np.allclose(camera_from_imu.T @ camera_from_imu, np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(camera_from_imu), 1.0, atol=1e-6)
    ):
        raise ValueError("camera-from-IMU must be a proper 3x3 rotation")
    return camera_from_imu


def _rebuild_estimator(config: dict[str, Any]):
    """A fresh estimator with the saved run's calibration, policies and seed."""
    import cv2
    from fusionlab.vision.odometry.visual_odometry import MonocularOdometry

    tracker = MonocularOdometry(
        json.loads(Path(config["calibration"]).read_text()),
        pose_acceptance=config.get("pose_acceptance", "baseline"),
        keyframe_policy=config.get("keyframe_policy", "baseline"),
        landmark_policy=config.get("landmark_policy", "baseline"),
        bundle_adjustment=config.get("bundle_adjustment", False),
    )
    cv2.setRNGSeed(config.get("ransac_seed", 0))
    return tracker


def _saved_results(run_dir: Path) -> dict[int, dict[str, Any]]:
    """Saved poses.jsonl rows keyed by frame index."""
    with (run_dir / "poses.jsonl").open() as file:
        return {r["frame_index"]: r for line in file if (r := json.loads(line))}


def _index_capture(capture: Path):
    """IMU events per (session, stream) and the session of every camera frame."""
    with (capture / "events.jsonl").open() as file:
        events = [json.loads(line) for line in file]
    imu: dict[tuple[str, str], list[dict[str, Any]]] = {}
    camera_sessions: dict[tuple[str, int, int], str] = {}
    for event in events:
        if event["type"] == "ImuSample":
            imu.setdefault((event["session_id"], event["stream_id"]), []).append(event)
        elif event["type"] == "CameraSample":
            key = (
                event["stream_id"],
                event["source_time_ns"],
                event["received_monotonic_ns"],
            )
            camera_sessions[key] = event["session_id"]
    return imu, camera_sessions


def _replay_frames(tracker, tracks: Path, saved, output: Path, context):
    """Replay saved frames, writing one diagnostic record per frame.

    Returns (records, frame indices whose replay differs from the saved run).
    """
    history: list[dict[str, Any]] = []
    last_geometry = None
    rows: list[dict[str, Any]] = []
    mismatches: list[int] = []
    origin = None
    with tracks.open() as file, (output / "frames.jsonl").open("x") as report_file:
        for index, line in enumerate(file):
            if index not in saved:
                continue
            row = TrackFrame.from_json(json.loads(line))
            if row.geometry_key != last_geometry:
                history = []
            last_geometry = row.geometry_key
            result = tracker.process(row)
            if not _reproduces(result, saved[index]):
                mismatches.append(index)
            if origin is None:
                origin = row.received_monotonic_ns
            item = _frame_record(index, row, result, origin)
            history = _continuous_history(history, result, row)
            snapshot = tracker.last_pnp_snapshot
            candidate = _candidate_pose(result, snapshot)
            prediction = predict_constant_velocity(history, row.source_time_ns)
            if snapshot is not None:
                _compare_hypotheses(
                    item, row, result, snapshot, candidate, prediction, history
                )
            if history and candidate is not None:
                item["gyro"] = _gyro_comparison(context, row, candidate, history)
            report_file.write(json.dumps(item, allow_nan=False) + "\n")
            rows.append(item)
            history = _advance_history(history, result, candidate, row)
            if len(rows) % 100 == 0:
                print(f"Diagnosed {len(rows)}/{len(saved)} frames", flush=True)
    return rows, mismatches


def _reproduces(result, saved: dict[str, Any]) -> bool:
    """Same status and segment, and (when tracking) the same pose to 1e-8."""
    import numpy as np

    if result.status != saved["status"] or result.segment != saved["segment"]:
        return False
    if result.status != "tracking":
        return True
    return bool(
        np.allclose(
            result.world_to_camera, saved["world_to_camera"], atol=1e-8, rtol=1e-8
        )
    )


def _frame_record(index: int, row: TrackFrame, result, origin: int) -> dict[str, Any]:
    """Identity, status and timing fields that start every frame record."""
    return {
        "frame_index": index,
        "status": result.status,
        "segment": result.segment,
        "receive_elapsed_s": (row.received_monotonic_ns - origin) / 1e9,
        "source_time_ns": row.source_time_ns,
        "failure_category": result.failure_category,
        "overlay_path": row.overlay_path,
    }


def _continuous_history(history, result, row: TrackFrame):
    """Drop history across a segment change or a source-time discontinuity."""
    if history and (
        history[-1]["segment"] != result.segment
        or not 0 < row.source_time_ns - history[-1]["time"] <= HISTORY_MAX_GAP_NS
    ):
        return []
    return history


def _candidate_pose(result, snapshot):
    """The accepted pose when tracking, else the raw RANSAC candidate if any."""
    import numpy as np

    if result.status == "tracking":
        return np.asarray(result.world_to_camera)
    return snapshot.get("candidate_pose") if snapshot else None


def _compare_hypotheses(item, row, result, snapshot, candidate, prediction, history):
    """Score PnP, CV and CV-translation hypotheses on the shared snapshot."""
    import numpy as np

    lookup = row.positions()
    pixels = np.array([lookup[key] for key in snapshot["ids"]], dtype=float).reshape(
        -1, 2
    )
    item["candidate_spatial"] = spatial_distribution(pixels, row.tracking_size)
    item["map_basis"] = MAP_BASIS
    if candidate is not None:
        item["pnp"] = evaluate_pose(candidate, snapshot, pixels, row.tracking_size)
        item["pnp_pose_kind"] = (
            "accepted online pose after optional BA"
            if result.status == "tracking"
            else "rejected raw RANSAC candidate"
        )
    if prediction is None:
        return
    item["constant_velocity"] = evaluate_pose(
        prediction, snapshot, pixels, row.tracking_size
    )
    item["predicted_center_arbitrary"] = (
        -prediction[:, :3].T @ prediction[:, 3]
    ).tolist()
    if candidate is not None:
        item.update(
            _innovation(item, candidate, prediction, history, snapshot, pixels, row)
        )


def _innovation(item, candidate, prediction, history, snapshot, pixels, row):
    """How far the PnP pose departs from the CV prediction.

    The translation innovation is normalized by the predicted step so it is
    comparable across segments of different arbitrary scale. The last entry
    holds the PnP rotation fixed to isolate the CV translation's support.
    """
    import numpy as np

    predicted_center = np.asarray(item["predicted_center_arbitrary"])
    center = -candidate[:, :3].T @ candidate[:, 3]
    innovation = float(np.linalg.norm(center - predicted_center))
    last_center = -history[-1]["pose"][:, :3].T @ history[-1]["pose"][:, 3]
    expected_step = float(np.linalg.norm(predicted_center - last_center))
    translation_only = np.column_stack(
        (candidate[:, :3], -candidate[:, :3] @ predicted_center)
    )
    return {
        "translation_innovation_arbitrary": innovation,
        "innovation_over_predicted_step": (
            innovation / expected_step if expected_step > 1e-6 else None
        ),
        "predicted_step_arbitrary": expected_step,
        "rotation_prediction_error_deg": rotation_angle(
            candidate[:, :3] @ prediction[:, :3].T
        ),
        "cv_translation_with_pnp_rotation": evaluate_pose(
            translation_only, snapshot, pixels, row.tracking_size
        ),
    }


def _gyro_comparison(context, row: TrackFrame, candidate, history):
    """Gyro vs. visual rotation since the previous accepted frame."""
    session = context.camera_sessions.get(
        (row.stream_id, row.source_time_ns, row.received_monotonic_ns)
    )
    streams = [samples for (sid, _), samples in context.imu.items() if sid == session]
    if len(streams) != 1:
        return {"status": "missing_or_ambiguous_session_imu"}
    visual = candidate[:, :3] @ history[-1]["pose"][:, :3].T
    return gyro_interval(
        streams[0],
        history[-1]["time"],
        row.source_time_ns,
        camera_from_imu=context.camera_from_imu,
        visual_rotation=visual,
        offset_ns=context.offset_ns,
    )


def _advance_history(history, result, candidate, row: TrackFrame):
    """Keep the two latest accepted poses; any non-tracking frame clears them."""
    if result.status != "tracking":
        return []
    history.append(
        {
            "pose": candidate.copy(),
            "time": row.source_time_ns,
            "segment": result.segment,
        }
    )
    return history[-2:]


def _write_html(output: Path, tracks_root: Path, jumps, losses, summary) -> None:
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


def write_findings(output: Path) -> dict[str, Any]:
    """Summarize observed disagreements; predictions are not ground truth.

    Reads `frames.jsonl` from `output` and writes `findings.json`, `findings.md`
    and, when Matplotlib is available, `motion_checks.png`.
    """
    with (output / "frames.jsonl").open() as file:
        rows = [json.loads(line) for line in file]
    findings = _findings(rows)
    (output / "findings.json").write_text(
        json.dumps(findings, indent=2, allow_nan=False) + "\n"
    )
    (output / "findings.md").write_text(_findings_markdown(findings))
    try:
        import numpy  # noqa: F401
        from matplotlib.figure import Figure  # noqa: F401
    except ImportError:
        return findings
    _plot_motion_checks(rows, output / "motion_checks.png")
    return findings


def _passes_support_gates(metrics: dict[str, Any]) -> bool:
    """Would this hypothesis pass the estimator's 20-inlier / 50% gates?"""
    return (
        metrics.get("inliers_3px", 0) >= MIN_SUPPORT_INLIERS
        and metrics["inlier_ratio"] >= MIN_SUPPORT_RATIO
    )


def _findings(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts comparing visual, CV and gyro evidence over all frames."""
    from statistics import median

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
        if "constant_velocity" in r and _passes_support_gates(r["constant_velocity"])
    ]
    return {
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
        "interpretation": FINDINGS_INTERPRETATION,
    }


def _findings_markdown(findings: dict[str, Any]) -> str:
    """Human-readable view of the findings counts plus the method caveats."""
    lines = [
        "# Visual motion diagnostics findings",
        "",
        findings["interpretation"],
        "",
        f"Comparable visual/CV frames: {findings['compared_pose_frames']}.",
        f"Visual pose has more geometric inliers: {findings['visual_more_inliers']} "
        f"frames; CV has more: {findings['cv_more_inliers']}.",
        f"Tracking losses with CV available: {findings['losses_with_cv']}; CV passing "
        f"the same geometric gates: {findings['losses_where_cv_passes_geometric_gates']}.",
        "Median visual/gyro rotation-magnitude difference: "
        f"{findings['median_gyro_visual_angle_difference_deg']} degrees.",
        "",
        FINDINGS_OUTPUT_GUIDE,
        "",
        FINDINGS_SCOPE,
    ]
    return "\n".join(lines) + "\n"


def _plot_motion_checks(rows: list[dict[str, Any]], path: Path) -> None:
    """Five stacked time series with tracking losses marked in red."""
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    losses = [r for r in rows if r["status"] == "tracking_lost"]
    figure = Figure(figsize=(12, 13))
    FigureCanvasAgg(figure)
    axes = figure.subplots(5, 1, sharex=True)
    times = [r["receive_elapsed_s"] for r in rows]

    def metric(key, name):
        value = [r.get(key, {}).get(name) for r in rows]
        return [np.nan if v is None else v for v in value]

    for key, label in (("pnp", "Visual pose"), ("constant_velocity", "CV")):
        axes[0].plot(times, metric(key, "inliers_3px"), label=label)
        axes[1].plot(times, metric(key, "median_error_px"), label=label)
    axes[1].set_yscale("symlog", linthresh=1.0)
    axes[2].plot(
        times,
        [
            100
            * r.get("pnp", {}).get("inlier_spatial", {}).get("hull_fraction", np.nan)
            for r in rows
        ],
    )
    axes[3].plot(times, metric("gyro", "visual_angle_deg"), label="Visual")
    axes[3].plot(times, metric("gyro", "gyro_angle_deg"), label="Gyro (assumed timing)")
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
    figure.savefig(path, dpi=140)
