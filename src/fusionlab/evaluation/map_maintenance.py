"""Map-maintenance diagnostics for a saved monocular VO run.

Questions this answers from `poses.jsonl` and `tracks.jsonl`, without rerunning
or changing the estimator:

1. At each tracking loss, were image tracks missing (frontend) or present but
   unmapped (map maintenance)? How fast did visible mapped landmarks decline?
2. How old was the reference keyframe that new landmarks were triangulated
   against? (Inferred; see `reference_ages`.)
3. How much parallax did each track offer against its own first observation,
   compared with what the map used?
4. At losses where PnP support failed with landmarks still visible, did recent
   and older landmarks fit the candidate pose differently?
5. Optionally: was an external reference tracker (ORB-SLAM3) tracking then?
"""

import json
from pathlib import Path
from statistics import median
from typing import Any

from fusionlab.vision.odometry.frame_result import (
    FrameResult,
    TrackingStatus,
    read_frame_results,
)
from fusionlab.vision.rays import track_rays
from fusionlab.vision.tracks import read_track_frames

# Method caveats saved with every report (what the numbers do not show).
MAP_DIAGNOSTIC_LIMITS = (
    "Diagnostic only; nothing here changes estimation. Parallax angles use the "
    "run's own estimated rotations, so they inherit its rotation error, and a "
    "large angle does not prove a correct match. Reference age is inferred from "
    "the saved policy and landmark additions, not logged by the estimator. "
    "External tracker states are matched by nearest source timestamp."
)

# Frames before each loss at which the visible mapped landmark count is sampled.
LOOKBACK_FRAMES = (1, 3, 5, 10, 20)
# Parallax thresholds (degrees) for per-track available-parallax statistics.
PARALLAX_THRESHOLDS_DEG = (1.0, 2.0, 3.0, 5.0)
# The threshold used for the per-frame "could already be triangulated" count.
USEFUL_PARALLAX_DEG = 2.0
# ORB-SLAM3 tracking-state codes counted as tracking (see benchmarks/orbslam3).
ORB_TRACKING_STATES = (2, 5)
GEOMETRIC_FAILURES = ("low_pnp_inlier_ratio", "too_few_pnp_inliers")


def _median(values):
    values = list(values)
    return median(values) if values else None


def _percentile(values, fraction):
    values = sorted(values)
    return values[min(int(fraction * len(values)), len(values) - 1)] if values else None


def _mapped_fraction(result: FrameResult):
    """Share of visible tracks that are mapped landmarks in this frame."""
    visible = result.diagnostics["visible_features"]
    return result.diagnostics["visible_landmarks"] / visible if visible else None


def _frame_counts(results: list[FrameResult]) -> dict[str, Any]:
    fractions = [f for f in map(_mapped_fraction, results) if f is not None]
    return {
        "median_visible_tracks": _median(
            r.diagnostics["visible_features"] for r in results
        ),
        "median_visible_mapped": _median(
            r.diagnostics["visible_landmarks"] for r in results
        ),
        "median_mapped_fraction": _median(fractions),
    }


def loss_context(results: list[FrameResult]) -> dict[str, Any]:
    """Tracks visible vs. mapped at each loss, and the decline beforehand."""
    losses = [
        i for i, r in enumerate(results) if r.status == TrackingStatus.TRACKING_LOST
    ]
    categories: dict[str, int] = {}
    for i in losses:
        category = results[i].failure_category
        categories[category] = categories.get(category, 0) + 1
    decline = {}
    for k in LOOKBACK_FRAMES:
        earlier = [
            results[i - k].diagnostics["visible_landmarks"]
            for i in losses
            if i >= k
            and results[i - k].is_tracking
            and results[i - k].segment == results[i].segment
        ]
        decline[str(k)] = {
            "median_visible_mapped": _median(earlier),
            "losses": len(earlier),
        }
    at_loss = [results[i] for i in losses]
    return {
        "losses": len(losses),
        "failure_categories": categories,
        "at_loss": _frame_counts(at_loss)
        | {
            "min_visible_tracks": min(
                (r.diagnostics["visible_features"] for r in at_loss), default=None
            ),
            "median_stored_landmarks": _median(
                r.diagnostics["landmarks_before"] for r in at_loss
            ),
        },
        "while_tracking": _frame_counts([r for r in results if r.is_tracking]),
        "visible_mapped_before_loss": decline,
    }


def segment_spans(results: list[FrameResult]) -> dict[str, Any]:
    """Tracked frames per VO segment, to expose very short segments."""
    frames: dict[int, int] = {}
    for result in results:
        if result.is_tracking:
            frames[result.segment] = frames.get(result.segment, 0) + 1
    counts = sorted(frames.values())
    return {
        "segments": len(counts),
        "tracked_frames_per_segment": counts,
        "segments_with_one_tracked_frame": sum(c == 1 for c in counts),
    }


def triangulation_totals(results: list[FrameResult]) -> dict[str, Any]:
    """Sum triangulation outcomes and summarize admitted angles."""
    totals: dict[str, int] = {}
    angles: list[float] = []
    added = []
    for result in results:
        if not result.is_tracking:
            continue
        added.append(result.diagnostics.get("landmarks_added", 0))
        triangulation = result.diagnostics.get("triangulation", {})
        for key, value in triangulation.items():
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
        angles.extend(triangulation.get("accepted_parallax_deg", []))
    candidates = totals.get("candidates", 0)
    return {
        "totals": totals,
        "low_parallax_share": (
            totals.get("low_parallax", 0) / candidates if candidates else None
        ),
        "accepted_share": (
            totals.get("accepted", 0) / candidates if candidates else None
        ),
        "mean_landmarks_added_per_tracking_frame": (
            sum(added) / len(added) if added else None
        ),
        "median_admitted_parallax_deg": _median(angles),
        "admitted_below_2_deg_share": (
            sum(a < 2 for a in angles) / len(angles) if angles else None
        ),
    }


def reference_ages(results: list[FrameResult], summary: dict[str, Any]):
    """Processed frames since the reference keyframe was set, per frame, or None.

    Inference: with the baseline keyframe policy and no third-view validation,
    the estimator moves the reference to the current frame exactly when a frame
    adds landmarks (and at bootstrap). Other policies move it for other reasons,
    so no age is inferred for them.
    """
    keyframes_ok = summary.get("keyframe_policy", "baseline") == "baseline"
    landmarks_ok = summary.get("landmark_policy", "baseline") in (
        "baseline",
        "refinement",
    )
    if not (keyframes_ok and landmarks_ok):
        return None
    ages: list[int | None] = []
    age = None
    for result in results:
        bootstrap = "bootstrap_origin_received_monotonic_ns" in result.diagnostics
        if not result.is_tracking:
            age = None
        elif age is None or bootstrap:
            age = 0
        else:
            age += 1
        ages.append(age)
        if result.is_tracking and result.diagnostics.get("landmarks_added", 0) > 0:
            age = 0
    return ages


def reference_age_summary(ages) -> dict[str, Any] | None:
    if ages is None:
        return None
    known = [a for a in ages if a is not None]
    return {"median_frames": _median(known), "p90_frames": _percentile(known, 0.9)}


def available_parallax(
    results: list[FrameResult], tracks: Path, calibration: dict[str, Any]
) -> dict[str, Any]:
    """Angle each track reaches against its own first observation in a segment.

    The first tracked observation of a track in a VO segment defines a world
    bearing b0 = R0.T [x, y, 1]; later bearings are compared with it. The angle
    between two bearings of the same point is its triangulation angle, so this is
    what a per-track scheme could use. Frame counts are in processed frames.
    """
    import numpy as np

    by_frame = {r.frame_index: r for r in results}
    processed = {index: n for n, index in enumerate(sorted(by_frame))}
    first: dict[tuple[int, int], tuple[int, Any]] = {}
    best: dict[tuple[int, int], float] = {}
    reached: dict[tuple[int, int], dict[float, int]] = {}
    lifetime: dict[tuple[int, int], int] = {}
    useful_per_frame, mapped_per_frame = [], []
    for index, frame in read_track_frames(tracks):
        result = by_frame.get(index)
        if result is None or not result.is_tracking:
            continue
        rotation = np.asarray(result.world_to_camera)[:, :3]
        rays, _ = track_rays(frame, calibration)
        useful = 0
        for track_id, ray in rays.items():
            bearing = rotation.T @ np.array([ray[0], ray[1], 1.0])
            bearing /= np.linalg.norm(bearing)
            key = (result.segment, track_id)
            if key not in first:
                first[key] = (processed[index], bearing)
                best[key], reached[key], lifetime[key] = 0.0, {}, 0
                continue
            lifetime[key] += 1
            cosine = np.clip(first[key][1] @ bearing, -1, 1)
            angle = float(np.degrees(np.arccos(cosine)))
            best[key] = max(best[key], angle)
            for threshold in PARALLAX_THRESHOLDS_DEG:
                if angle >= threshold and threshold not in reached[key]:
                    reached[key][threshold] = processed[index] - first[key][0]
            useful += best[key] >= USEFUL_PARALLAX_DEG
        useful_per_frame.append(useful)
        mapped_per_frame.append(result.diagnostics["visible_landmarks"])
    keys = list(best)
    reach = {}
    for threshold in PARALLAX_THRESHOLDS_DEG:
        frames = [reached[k][threshold] for k in keys if threshold in reached[k]]
        reach[f"{threshold:g}"] = {
            "track_share": len(frames) / len(keys) if keys else None,
            "median_processed_frames_needed": _median(frames),
        }
    return {
        "tracks": len(keys),
        "median_lifetime_processed_frames": _median(lifetime.values()),
        "p90_lifetime_processed_frames": _percentile(lifetime.values(), 0.9),
        "reach": reach,
        "per_tracking_frame": {
            "median_visible_with_useful_parallax": _median(useful_per_frame),
            "useful_parallax_deg": USEFUL_PARALLAX_DEG,
            "median_visible_mapped": _median(mapped_per_frame),
        },
    }


def geometric_losses(results: list[FrameResult]) -> list[dict[str, Any]]:
    """PnP-support losses: candidate errors of recent vs. older landmarks."""
    rows = []
    for result in results:
        if result.failure_category not in GEOMETRIC_FAILURES:
            continue
        d = result.diagnostics
        rows.append(
            {
                "frame_index": result.frame_index,
                "category": result.failure_category,
                "pnp_candidates": d.get("pnp_candidates"),
                "pnp_inlier_ratio": d.get("pnp_inlier_ratio"),
                "recent_landmarks": d.get("pnp_recent_landmarks"),
                "recent_median_error_px": d.get("pnp_recent_median_error_px"),
                "older_landmarks": d.get("pnp_older_landmarks"),
                "older_median_error_px": d.get("pnp_older_median_error_px"),
                "refinement_would_pass_same_gates": d.get(
                    "diagnostic_refinement_passes_same_gates"
                ),
            }
        )
    return rows


def reference_tracker_overlap(results: list[FrameResult], states_csv: Path):
    """External tracker state (nearest source time) at each FusionLab loss."""
    import csv
    from bisect import bisect_left

    with Path(states_csv).open() as file:
        states = list(csv.DictReader(file))
    times = [int(s["source_time_ns"]) for s in states]
    losses = []
    for result in results:
        if result.status != TrackingStatus.TRACKING_LOST:
            continue
        i = min(bisect_left(times, result.source_time_ns), len(times) - 1)
        losses.append(
            {
                "frame_index": result.frame_index,
                "source_elapsed_s": (result.source_time_ns - times[0]) / 1e9,
                "category": result.failure_category,
                "reference_tracking": int(states[i]["state"]) in ORB_TRACKING_STATES,
            }
        )
    return {
        "losses_while_reference_tracking": sum(
            loss["reference_tracking"] for loss in losses
        ),
        "losses": losses,
    }


def diagnose_map(
    run_dir: Path,
    output: Path,
    *,
    tracks: Path | None = None,
    calibration: Path | None = None,
    reference_states: Path | None = None,
) -> dict[str, Any]:
    """Write `summary.json` and `report.md` for one saved VO run.

    `tracks`/`calibration` override the paths recorded in the run's summary,
    which may be historical. `output` must be a new directory.
    """
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    tracks = Path(tracks or summary["tracks"])
    calibration = Path(calibration or summary["calibration"])
    results = read_frame_results(run_dir / "poses.jsonl")
    configuration_keys = (
        "stride",
        "pose_acceptance",
        "keyframe_policy",
        "landmark_policy",
        "bundle_adjustment",
        "ransac_seed",
    )
    report = {
        "run": str(run_dir.resolve()),
        "tracks": str(tracks.resolve()),
        "calibration": str(calibration.resolve()),
        "configuration": {key: summary.get(key) for key in configuration_keys},
        "status_counts": summary.get("status_counts"),
        "segments": segment_spans(results),
        "loss_context": loss_context(results),
        "triangulation": triangulation_totals(results),
        "reference_age": reference_age_summary(reference_ages(results, summary)),
        "available_parallax": available_parallax(
            results, tracks, json.loads(calibration.read_text())
        ),
        "geometric_losses": geometric_losses(results),
        "reference_tracker": (
            reference_tracker_overlap(results, reference_states)
            if reference_states
            else None
        ),
        "limits": MAP_DIAGNOSTIC_LIMITS,
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "report.md").write_text(render_report(report))
    return report


def render_report(report: dict[str, Any]) -> str:
    """A compact Markdown view of the numbers in `summary.json`."""
    loss = report["loss_context"]
    tri = report["triangulation"]
    parallax = report["available_parallax"]
    age = report["reference_age"]
    segments = report["segments"]
    before = ", ".join(
        f"{k}: {v['median_visible_mapped']}"
        for k, v in loss["visible_mapped_before_loss"].items()
    )
    lines = [
        "# Map-maintenance diagnostics",
        "",
        MAP_DIAGNOSTIC_LIMITS,
        "",
        f"Configuration: `{json.dumps(report['configuration'])}`",
        "",
        "## Losses and segments",
        "",
        f"- Losses: {loss['losses']} {loss['failure_categories']}",
        f"- Segments: {segments['segments']}; with a single tracked frame: "
        f"{segments['segments_with_one_tracked_frame']}",
        f"- At loss (median): {loss['at_loss']['median_visible_tracks']} visible "
        f"tracks, {loss['at_loss']['median_visible_mapped']} visible mapped "
        f"(fraction {_fmt(loss['at_loss']['median_mapped_fraction'])})",
        f"- While tracking (median): {loss['while_tracking']['median_visible_tracks']}"
        f" visible tracks, {loss['while_tracking']['median_visible_mapped']} mapped",
        f"- Visible mapped k frames before a loss (median): {before}",
        "",
        "## Triangulation",
        "",
        f"- Candidates {tri['totals'].get('candidates', 0)}; low-parallax share "
        f"{_fmt(tri['low_parallax_share'])}; accepted share "
        f"{_fmt(tri['accepted_share'])}",
        f"- Landmarks added per tracking frame "
        f"{_fmt(tri['mean_landmarks_added_per_tracking_frame'])}; admitted angles "
        f"below 2°: {_fmt(tri['admitted_below_2_deg_share'])}",
        "- Reference keyframe age (processed frames): "
        + (
            f"median {age['median_frames']}, 90th pct {age['p90_frames']}"
            if age
            else "not inferred for this policy"
        ),
        "",
        "## Parallax available per track (vs. its first observation)",
        "",
        f"- Tracks: {parallax['tracks']}; lifetime median "
        f"{parallax['median_lifetime_processed_frames']}, 90th pct "
        f"{parallax['p90_lifetime_processed_frames']} processed frames",
    ]
    for threshold, values in parallax["reach"].items():
        lines.append(
            f"- ≥ {threshold}°: {_fmt(values['track_share'])} of tracks, after "
            f"{values['median_processed_frames_needed']} frames (median)"
        )
    per_frame = parallax["per_tracking_frame"]
    lines += [
        f"- Per tracking frame (median): {per_frame['median_visible_with_useful_parallax']}"
        f" visible tracks already offer ≥ {per_frame['useful_parallax_deg']:g}°; "
        f"{per_frame['median_visible_mapped']} are mapped",
        "",
        "## Geometric losses (PnP support)",
        "",
        "| Frame | Category | Candidates | Inlier ratio | Recent (n @ px) "
        "| Older (n @ px) | Refinement passes |",
        "| ---: | --- | ---: | ---: | --- | --- | --- |",
    ]
    for row in report["geometric_losses"]:
        lines.append(
            f"| {row['frame_index']} | {row['category']} | {row['pnp_candidates']} | "
            f"{_fmt(row['pnp_inlier_ratio'])} | {row['recent_landmarks']} @ "
            f"{_fmt(row['recent_median_error_px'])} | {row['older_landmarks']} @ "
            f"{_fmt(row['older_median_error_px'])} | "
            f"{row['refinement_would_pass_same_gates']} |"
        )
    reference = report["reference_tracker"]
    if reference:
        lines += [
            "",
            "## External reference tracker",
            "",
            "Losses while the reference tracker was tracking: "
            f"{reference['losses_while_reference_tracking']} / "
            f"{len(reference['losses'])}.",
        ]
    return "\n".join(lines) + "\n"


def _fmt(value):
    """Short numeric formatting for the Markdown report."""
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.3f}" if abs(value) < 1 else f"{value:.2f}"
    return str(value)
