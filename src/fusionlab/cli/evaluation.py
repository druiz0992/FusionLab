"""Evaluation commands: VO motion diagnostics and ground-truth scoring.

Diagnostics inspect saved runs; they never change estimator decisions.
"""

import argparse
import json
from pathlib import Path


def register(commands) -> None:
    diagnose = commands.add_parser(
        "diagnose-vo",
        help="Compare VO poses with CV prediction, spatial support and gyro",
    )
    diagnose.add_argument("run_dir", type=Path)
    diagnose.add_argument("--capture", type=Path, required=True)
    diagnose.add_argument("--output-dir", type=Path, required=True)
    diagnose.add_argument("--imu-offset-ms", type=float, default=0.0)
    diagnose.add_argument(
        "--camera-from-imu",
        type=Path,
        help="JSON 3x3 rotation matrix; omit for angle-magnitude-only comparisons",
    )
    diagnose.set_defaults(handler=run_diagnose_vo)

    mapping = commands.add_parser(
        "diagnose-map",
        help="Landmark creation, reference age and parallax use of a saved VO run",
    )
    mapping.add_argument("run_dir", type=Path)
    mapping.add_argument("--output-dir", type=Path, required=True)
    mapping.add_argument(
        "--tracks", type=Path, help="Override the tracks.jsonl path saved in the run"
    )
    mapping.add_argument(
        "--calibration",
        type=Path,
        help="Override the calibration path saved in the run",
    )
    mapping.add_argument(
        "--reference-states",
        type=Path,
        help="ORB-SLAM3 states.csv to report its state at each FusionLab loss",
    )
    mapping.set_defaults(handler=run_diagnose_map)

    evaluate = commands.add_parser(
        "evaluate-trajectory", help="Score segmented VO against imported ground truth"
    )
    evaluate.add_argument("run_dir", type=Path)
    evaluate.add_argument("--groundtruth", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--max-gap-ms", type=float, default=50.0)
    evaluate.set_defaults(handler=run_evaluate_trajectory)


def run_diagnose_vo(args: argparse.Namespace) -> None:
    from fusionlab.evaluation.vo_motion_diagnostics import diagnose_run

    # No camera-from-IMU rotation is assumed unless one is supplied explicitly.
    alignment = (
        json.loads(args.camera_from_imu.read_text()) if args.camera_from_imu else None
    )
    report = diagnose_run(
        args.run_dir,
        args.capture,
        args.output_dir,
        camera_from_imu=alignment,
        imu_offset_ms=args.imu_offset_ms,
    )
    print(json.dumps(report, indent=2))


def run_diagnose_map(args: argparse.Namespace) -> None:
    from fusionlab.evaluation.map_maintenance import diagnose_map

    report = diagnose_map(
        args.run_dir,
        args.output_dir,
        tracks=args.tracks,
        calibration=args.calibration,
        reference_states=args.reference_states,
    )
    # Per-loss rows stay in summary.json; print the aggregate sections.
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("geometric_losses",)},
            indent=2,
        )
    )


def run_evaluate_trajectory(args: argparse.Namespace) -> None:
    from fusionlab.evaluation.ground_truth import evaluate_run

    result = evaluate_run(
        args.run_dir, args.groundtruth, args.output_dir, max_gap_ms=args.max_gap_ms
    )
    print(json.dumps(result, indent=2))
