"""Comparison diagnostics for segmented, arbitrary-scale monocular trajectories."""

import json
from math import acos, degrees, sqrt
from pathlib import Path
from statistics import median


def summarize_run(directory: Path):
    """Describe continuity and pose stability without claiming ground-truth accuracy."""
    with (directory / "poses.jsonl").open() as file:
        rows = [json.loads(line) for line in file]
    groups = {}
    for row in rows:
        if row["status"] == "tracking":
            groups.setdefault(row["segment"], []).append(row)
    durations = [
        (group[-1]["source_time_ns"] - group[0]["source_time_ns"]) / 1e9
        for group in groups.values()
    ]
    rotation_steps = []
    large_step_ratios = []
    for group in groups.values():
        translation_steps = []
        for previous, current in zip(group, group[1:]):
            a = previous["camera_center_arbitrary"]
            b = current["camera_center_arbitrary"]
            translation_steps.append(sqrt(sum((x - y) ** 2 for x, y in zip(a, b))))
            # trace(R_new R_old.T) is the Frobenius inner product of the rotations.
            a = previous["world_to_camera"]
            b = current["world_to_camera"]
            trace = sum(a[i][j] * b[i][j] for i in range(3) for j in range(3))
            rotation_steps.append(degrees(acos(max(-1.0, min(1.0, (trace - 1) / 2)))))
        if len(translation_steps) >= 5 and median(translation_steps) > 1e-12:
            large_step_ratios.extend(
                step / median(translation_steps) for step in translation_steps
            )
    errors = [
        r["refined_median_inlier_error_px"]
        for r in rows
        if r["status"] == "tracking"
        and r.get("refined_median_inlier_error_px") is not None
    ]
    return {
        "processed_frames": len(rows),
        "tracked_frames": sum(r["status"] == "tracking" for r in rows),
        "tracking_losses": sum(r["status"] == "tracking_lost" for r in rows),
        "segments": len(groups),
        "median_tracked_segment_span_s": median(durations) if durations else None,
        "longest_tracked_segment_span_s": max(durations) if durations else None,
        "median_refined_inlier_error_px": median(errors) if errors else None,
        "largest_within_segment_rotation_step_deg": (
            max(rotation_steps) if rotation_steps else None
        ),
        "largest_step_over_segment_median": (
            max(large_step_ratios) if large_step_ratios else None
        ),
        "steps_above_5_times_segment_median": sum(
            ratio > 5 for ratio in large_step_ratios
        ),
        "limits": "Segment span excludes bootstrap waiting. Step ratios use segments with at least five increments; they flag variability, not proven errors. No aligned position ground truth or common scale; accuracy and metric drift are not measured.",
    }


def inspect_translation_jumps(directory: Path, *, limit=5):
    """Rank within-segment steps relative to that segment's median step.

    Large ratios are inspection candidates, not proof of an erroneous pose.
    Absolute distances cannot be compared across independent segment scales.
    """
    with (directory / "poses.jsonl").open() as file:
        rows = [json.loads(line) for line in file]
    groups = {}
    for row in rows:
        if row["status"] == "tracking":
            groups.setdefault(row["segment"], []).append(row)
    events = []
    origin = rows[0]["received_monotonic_ns"] if rows else 0
    for segment, group in groups.items():
        steps = [
            sqrt(
                sum(
                    (x - y) ** 2
                    for x, y in zip(
                        a["camera_center_arbitrary"], b["camera_center_arbitrary"]
                    )
                )
            )
            for a, b in zip(group, group[1:])
        ]
        if len(steps) < 5 or median(steps) <= 1e-12:
            continue
        scale = median(steps)
        for index, step in enumerate(steps):
            row = group[index + 1]
            events.append(
                {
                    "frame_index": row["frame_index"],
                    "segment": segment,
                    "receive_elapsed_s": (row["received_monotonic_ns"] - origin) / 1e9,
                    "step_over_segment_median": step / scale,
                    "step_arbitrary_units": step,
                    "preceding_frames": [
                        {
                            key: r.get(key)
                            for key in (
                                "frame_index",
                                "landmarks",
                                "landmarks_added",
                                "pnp_candidates",
                                "pnp_inliers",
                                "pnp_inlier_ratio",
                                "pnp_median_candidate_error_px",
                                "refined_median_inlier_error_px",
                                "multiview_update",
                            )
                        }
                        for r in group[max(0, index - 1) : index + 2]
                    ],
                }
            )
    events.sort(key=lambda event: event["step_over_segment_median"], reverse=True)
    return events[:limit]
