"""Run monocular VO over a tracks.jsonl file and save poses and reports.

Orchestration only: it wires the estimator, optional viewer replay and
evaluation reports together. Estimation logic lives in visual_odometry.py.
"""

import json
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

from fusionlab.evaluation.vo_reports import write_failure_report, write_geometry_report
from fusionlab.vision.odometry.visual_odometry import MonocularOdometry
from fusionlab.vision.tracks import read_track_frames
from fusionlab.visualization.odometry_replay import OdometryReplay

# Method caveats saved with every report (what the numbers do not show).
CAMERA_CONFIGURATION_CAVEAT = (
    "Requires matching camera, zoom and image processing; resolution alone cannot "
    "verify compatibility."
)


def run_odometry(
    tracks: Path,
    calibration: Path,
    output: Path,
    *,
    stride=5,
    max_frames=None,
    viewer=None,
    viewer_factory=None,
    pose_acceptance="baseline",
    keyframe_policy="baseline",
    landmark_policy="baseline",
    bundle_adjustment=False,
    seed=0,
):
    """Run sequential pose estimation; save failures as well as successful poses."""
    if stride < 1 or (max_frames is not None and max_frames < 1):
        raise ValueError("Counts must be positive")
    tracker = MonocularOdometry(
        json.loads(calibration.read_text()),
        pose_acceptance=pose_acceptance,
        keyframe_policy=keyframe_policy,
        landmark_policy=landmark_policy,
        bundle_adjustment=bundle_adjustment,
    )
    tracker.cv.setRNGSeed(seed)
    output.mkdir(parents=True, exist_ok=False)
    with ExitStack() as resources:
        if viewer_factory is not None:
            viewer = resources.enter_context(viewer_factory())
        counts = {}
        processed = 0
        replay = OdometryReplay(viewer, tracks) if viewer else None
        with (output / "poses.jsonl").open("x") as out:
            for index, row in read_track_frames(tracks):
                if index % stride:
                    continue
                result = replace(tracker.process(row), frame_index=index)
                out.write(json.dumps(result.to_json(), allow_nan=False) + "\n")
                status = result.status.value
                counts[status] = counts.get(status, 0) + 1
                processed += 1
                if replay is not None:
                    replay.publish(row, result, index)
                if max_frames is not None and processed >= max_frames:
                    break
        if not processed:
            raise ValueError("No tracking frames")
        diagnosis = write_failure_report(output)
        write_geometry_report(output)
        report = {
            "failure_categories": diagnosis["failure_categories"],
            "pose_acceptance": pose_acceptance,
            "keyframe_policy": keyframe_policy,
            "landmark_policy": landmark_policy,
            "bundle_adjustment": bundle_adjustment,
            "ransac_seed": seed,
            "processed_frames": processed,
            "stride": stride,
            "status_counts": counts,
            "segments": tracker.segment,
            "tracks": str(tracks.resolve()),
            "calibration": str(calibration.resolve()),
            "calibration_status": tracker.calibration.get("status"),
            "scope": (
                "Experimental monocular VO, arbitrary scale per segment; "
                + (
                    "bounded local bundle adjustment enabled; "
                    if bundle_adjustment
                    else "no bundle adjustment; "
                )
                + "no loop closure, gravity alignment or metric scale."
            ),
            "camera_configuration": CAMERA_CONFIGURATION_CAVEAT,
        }
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
