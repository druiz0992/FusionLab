"""Camera commands: calibration, feature tracking, two-view and visual odometry."""

import argparse
import json
from pathlib import Path

from fusionlab.cli._common import print_json, rerun_viewer_factory
from fusionlab.vision.odometry.policies import (
    KEYFRAME_POLICIES,
    LANDMARK_POLICIES,
    POSE_ACCEPTANCE,
)

DISPLAY_ROTATIONS = (0, 90, 180, 270)


def register(commands) -> None:
    _register_calibration(commands)
    _register_tracking(commands)
    _register_two_view(commands)
    _register_visual_odometry(commands)


# -- Calibration --------------------------------------------------------------


def _register_calibration(commands) -> None:
    board = commands.add_parser(
        "calibration-target", help="Generate printable A4 checkerboard"
    )
    board.add_argument(
        "--output-dir", type=Path, default=Path("docs/calibration_target")
    )
    board.set_defaults(handler=run_calibration_target)

    calibration = commands.add_parser(
        "calibrate-camera", help="Estimate RGB intrinsics from checkerboard capture"
    )
    calibration.add_argument("capture", type=Path)
    calibration.add_argument(
        "--square-mm",
        type=float,
        required=True,
        help="Actual measured printed square size",
    )
    calibration.add_argument(
        "--columns", type=int, default=9, help="Inner corner columns, not squares"
    )
    calibration.add_argument(
        "--rows", type=int, default=6, help="Inner corner rows, not squares"
    )
    calibration.add_argument("--stride", type=int, default=15)
    calibration.add_argument("--max-views", type=int, default=40)
    calibration.add_argument("--camera-stream")
    calibration.add_argument("--output-dir", type=Path)
    calibration.set_defaults(handler=run_calibrate_camera)

    # Scores existing intrinsics on a separate capture; never refits them.
    validation = commands.add_parser(
        "validate-calibration",
        help="Score fixed RGB intrinsics on a separate checkerboard capture",
    )
    validation.add_argument("capture", type=Path)
    validation.add_argument("calibration", type=Path, help="Existing calibration.json")
    validation.add_argument(
        "--output-dir", type=Path, required=True, help="New directory for summary.json"
    )
    validation.add_argument(
        "--image-list",
        type=Path,
        help="detections.json whose accepted_images replace stride sampling",
    )
    validation.add_argument("--stride", type=int, default=60)
    validation.add_argument("--max-views", type=int, default=40)
    validation.set_defaults(handler=run_validate_calibration)


def run_calibration_target(args: argparse.Namespace) -> None:
    from fusionlab.vision.calibration import target

    target(args.output_dir)
    print(f"Printable PDF/SVG saved to {args.output_dir.resolve()}")


def run_calibrate_camera(args: argparse.Namespace) -> None:
    from fusionlab.vision.calibration import calibrate

    report = calibrate(
        args.capture,
        args.output_dir or args.capture / "calibration",
        square_mm=args.square_mm,
        columns=args.columns,
        rows=args.rows,
        stride=args.stride,
        max_views=args.max_views,
        camera_stream=args.camera_stream,
    )
    print(json.dumps(report, indent=2))


def run_validate_calibration(args: argparse.Namespace) -> None:
    from fusionlab.vision.calibration_validation import validate_calibration

    result = validate_calibration(
        args.capture,
        args.calibration,
        args.output_dir,
        image_list=args.image_list,
        stride=args.stride,
        max_views=args.max_views,
    )
    # Per-image rows stay in summary.json; print only the aggregate.
    print_json(result, omit=("images", "rejected_images"))


# -- Feature tracking ---------------------------------------------------------


def _register_tracking(commands) -> None:
    tracking = commands.add_parser(
        "track-features", help="Track camera image features without estimating pose"
    )
    tracking.add_argument("capture", type=Path)
    tracking.add_argument(
        "--output-dir", type=Path, required=True, help="New analysis directory"
    )
    tracking.add_argument("--max-frames", type=int)
    tracking.add_argument("--max-features", type=int, default=400)
    tracking.add_argument("--max-width", type=int, default=640)
    tracking.add_argument("--camera-stream")
    tracking.add_argument(
        "--display-rotation",
        type=int,
        choices=DISPLAY_ROTATIONS,
        default=90,
        help="Clockwise overlay rotation; tracking coordinates stay unchanged",
    )
    tracking.add_argument("--viewer", action="store_true")
    tracking.add_argument("--rrd", type=Path)
    tracking.set_defaults(handler=run_track_features)


def run_track_features(args: argparse.Namespace) -> None:
    from fusionlab.vision.feature_tracking import TrackingConfig, track_capture

    report = track_capture(
        args.capture,
        args.output_dir,
        config=TrackingConfig(max_features=args.max_features, max_width=args.max_width),
        max_frames=args.max_frames,
        stream_id=args.camera_stream,
        viewer_factory=rerun_viewer_factory(args.rrd, args.viewer),
        display_rotation=args.display_rotation,
    )
    print(json.dumps(report, indent=2))


# -- Two-frame geometry -------------------------------------------------------


def _register_two_view(commands) -> None:
    geometry = commands.add_parser(
        "two-view", help="Uncalibrated geometry from shared feature tracks"
    )
    geometry.add_argument("capture", type=Path)
    geometry.add_argument("--tracks", type=Path, required=True)
    geometry.add_argument("--output-dir", type=Path, required=True)
    geometry.add_argument("--frame-a", type=int, default=0)
    geometry.add_argument("--frame-b", type=int, default=10)
    geometry.add_argument(
        "--display-rotation", type=int, choices=DISPLAY_ROTATIONS, default=90
    )
    geometry.set_defaults(handler=run_two_view)

    motion = commands.add_parser(
        "camera-motion",
        help="Calibrated two-frame rotation and translation-direction diagnostics",
    )
    motion.add_argument("--tracks", type=Path, required=True)
    motion.add_argument("--calibration", type=Path, required=True)
    motion.add_argument("--output-dir", type=Path, required=True)
    motion.add_argument("--frame-a", type=int, default=1000)
    motion.add_argument("--frame-b", type=int, default=1010)
    motion.set_defaults(handler=run_camera_motion)


def run_two_view(args: argparse.Namespace) -> None:
    from fusionlab.vision.two_view import analyze_pair

    report = analyze_pair(
        args.capture,
        args.tracks,
        args.output_dir,
        frame_a=args.frame_a,
        frame_b=args.frame_b,
        display_rotation=args.display_rotation,
    )
    print_json(report, omit=("inlier_mask", "sampson_residual_px", "track_ids"))


def run_camera_motion(args: argparse.Namespace) -> None:
    from fusionlab.vision.calibrated_motion import analyze_motion

    report = analyze_motion(
        args.tracks,
        args.calibration,
        args.output_dir,
        frame_a=args.frame_a,
        frame_b=args.frame_b,
    )
    print_json(report, omit=("inlier_mask", "track_ids"))


# -- Visual odometry ----------------------------------------------------------


def _register_visual_odometry(commands) -> None:
    vo = commands.add_parser(
        "visual-odometry", help="Experimental multiframe monocular trajectory"
    )
    vo.add_argument("--tracks", type=Path, required=True)
    vo.add_argument("--calibration", type=Path, required=True)
    vo.add_argument("--output-dir", type=Path, required=True)
    vo.add_argument("--stride", type=int, default=5)
    # Variant names come from the policy registries (see odometry/policies.py).
    vo.add_argument(
        "--pose-acceptance", choices=tuple(POSE_ACCEPTANCE), default="baseline"
    )
    vo.add_argument(
        "--keyframe-policy", choices=tuple(KEYFRAME_POLICIES), default="baseline"
    )
    vo.add_argument(
        "--landmark-policy",
        choices=tuple(LANDMARK_POLICIES),
        default="baseline",
    )
    vo.add_argument(
        "--bundle-adjustment",
        action="store_true",
        help="Experimental bounded joint pose/landmark refinement",
    )
    vo.add_argument("--seed", type=int, default=0)
    vo.add_argument("--max-frames", type=int)
    vo.add_argument("--viewer", action="store_true")
    vo.add_argument("--rrd", type=Path)
    vo.set_defaults(handler=run_visual_odometry)


def run_visual_odometry(args: argparse.Namespace) -> None:
    from fusionlab.vision.odometry.runner import run_odometry

    report = run_odometry(
        args.tracks,
        args.calibration,
        args.output_dir,
        stride=args.stride,
        max_frames=args.max_frames,
        viewer_factory=rerun_viewer_factory(
            args.rrd, args.viewer, layout_profile="odometry"
        ),
        pose_acceptance=args.pose_acceptance,
        keyframe_policy=args.keyframe_policy,
        landmark_policy=args.landmark_policy,
        bundle_adjustment=args.bundle_adjustment,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))
