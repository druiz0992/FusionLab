"""Failure and geometry audit reports written next to a VO run's poses.jsonl.

These summarize saved results for inspection; they do not change estimation.
"""

import json
from pathlib import Path


def write_failure_report(output: Path):
    """Summarize observed failure conditions without asserting visual causes."""
    from collections import Counter

    with (output / "poses.jsonl").open() as file:
        rows = [json.loads(line) for line in file]
    origin = rows[0]["received_monotonic_ns"]
    failures = []
    for index, row in enumerate(rows):
        if row["status"] != "tracking_lost":
            continue
        history = [
            previous
            for previous in rows[max(0, index - 5) : index]
            if previous["segment"] == row["segment"]
            and previous["status"] == "tracking"
        ]
        failures.append(
            {
                "frame_index": row["frame_index"],
                "receive_elapsed_s": (row["received_monotonic_ns"] - origin) / 1e9,
                "segment": row["segment"],
                "category": row["failure_category"],
                "landmarks_before": row["landmarks_before"],
                "visible_landmarks": row["visible_landmarks"],
                "landmarks_no_longer_visible": row["landmarks_no_longer_visible"],
                "pnp_inliers": row.get("pnp_inliers", 0),
                "pnp_inlier_ratio": row.get("pnp_inlier_ratio", 0),
                "pnp_median_candidate_error_px": row.get(
                    "pnp_median_candidate_error_px"
                ),
                "preceding_frames": [
                    {
                        key: previous.get(key)
                        for key in (
                            "frame_index",
                            "landmarks",
                            "visible_landmarks",
                            "landmarks_added",
                            "pnp_inliers",
                            "pnp_median_inlier_error_px",
                        )
                    }
                    for previous in history
                ],
            }
        )
    report = {
        "failure_count": len(failures),
        "failure_categories": dict(Counter(f["category"] for f in failures)),
        "failures": failures,
        "interpretation": "Categories describe failed gates, not root causes. Pixel errors approximate normalized reprojection error times mean tracking focal length. Inspect video before attributing blur, calibration mismatch or bad landmarks.",
    }
    (output / "failures.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    text = [
        "# VO tracking failure report",
        "",
        report["interpretation"],
        "",
        "| Frame | Receive time (s) | Segment | Failed condition | Visible / stored landmarks | PnP inliers |",
        "| --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for failure in failures:
        text.append(
            f"| {failure['frame_index']} | {failure['receive_elapsed_s']:.2f} | {failure['segment']} | {failure['category']} | {failure['visible_landmarks']} / {failure['landmarks_before']} | {failure['pnp_inliers']} |"
        )
    text.extend(
        [
            "",
            "See `failures.json` for up to five preceding tracked frames per failure.",
            "Video uses these receive times; frame indices refer to rows in the tracking file.",
        ]
    )
    (output / "failures.md").write_text("\n".join(text) + "\n")
    return report


def write_geometry_report(output: Path):
    """Summarize triangulation sensitivity and counterfactual pose refinement."""
    from statistics import median

    with (output / "poses.jsonl").open() as file:
        rows = [json.loads(line) for line in file]
    angles = [
        angle
        for row in rows
        for angle in row.get("triangulation", {}).get("accepted_parallax_deg", [])
    ]
    rejected = [row for row in rows if row["status"] == "tracking_lost"]
    salvage = [
        row["frame_index"]
        for row in rejected
        if row.get("diagnostic_refinement_passes_same_gates", False)
    ]
    refined = [
        row for row in rows if row.get("refined_median_inlier_error_px") is not None
    ]
    report = {
        "triangulated_landmark_observations": len(angles),
        "median_admitted_parallax_deg": median(angles) if angles else None,
        "fraction_admitted_below_2_deg": (
            sum(angle < 2 for angle in angles) / len(angles) if angles else None
        ),
        "failure_count": len(rejected),
        "counterfactual_same_gate_recoveries": len(salvage),
        "counterfactual_frame_indices": salvage,
        "median_pre_refinement_inlier_error_px": (
            median(r["pnp_median_inlier_error_px"] for r in refined)
            if refined
            else None
        ),
        "median_post_refinement_inlier_error_px": (
            median(r["refined_median_inlier_error_px"] for r in refined)
            if refined
            else None
        ),
        "interpretation": "Diagnostic only. Recovered failed frames are independent counterfactuals on the original map, not a rerun with fewer resets. Counts do not predict the number of segments after a fix. Narrow-angle triangulation is sensitive to errors but this audit does not measure true landmark depth accuracy.",
    }
    (output / "geometry_audit.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    text = [
        "# Landmark and pose-refinement audit",
        "",
        f"Rejected frames passing the same gates after diagnostic refinement: **{len(salvage)} / {len(rejected)}**.",
        f"Admitted landmark observations: **{len(angles)}**.",
    ]
    if angles:
        text.extend(
            [
                f"Median admitted triangulation angle: **{median(angles):.2f}°**.",
                f"Fraction below 2°: **{100*sum(a<2 for a in angles)/len(angles):.1f}%**.",
            ]
        )
    text.extend(
        [
            "",
            report["interpretation"],
            "",
            "Next controlled experiments: refine and recompute geometric support before acceptance; evaluate longer-baseline keyframes and multiview landmark refinement. Do not merely relax thresholds or claim these changes will recover the whole walk.",
        ]
    )
    (output / "geometry_audit.md").write_text("\n".join(text) + "\n")
    return report
