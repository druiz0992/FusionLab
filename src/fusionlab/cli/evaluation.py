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


def run_evaluate_trajectory(args: argparse.Namespace) -> None:
    from fusionlab.evaluation.ground_truth import evaluate_run

    result = evaluate_run(
        args.run_dir, args.groundtruth, args.output_dir, max_gap_ms=args.max_gap_ms
    )
    print(json.dumps(result, indent=2))
