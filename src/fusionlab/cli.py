import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from fusionlab.recordings.replay import read_samples, replay
from fusionlab.inertial.orientation import integrate, plot, summary
from fusionlab.inertial.measurements import MeasurementConvention, convert_imu
from fusionlab.vision.odometry.policies import (
    KEYFRAME_POLICIES,
    LANDMARK_POLICIES,
    POSE_ACCEPTANCE,
)


async def _replay(
    path: Path, speed: float, acceleration_unit: str | None = None
) -> None:
    convention = MeasurementConvention(acceleration_unit) if acceleration_unit else None
    async for sample in replay(path, speed=speed):
        if convention:
            print(
                json.dumps(
                    {
                        "type": "ProcessedImuSample",
                        **asdict(convert_imu(sample, convention)),
                    }
                ),
                flush=True,
            )
        else:
            print(json.dumps({"type": "ImuSample", **asdict(sample)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fusion Lab offline experiments")
    # Subcommands share one entry point; each experiment declares its own
    # inputs and assumptions before dispatching to the processing modules.
    commands = parser.add_subparsers(dest="command", required=True)
    replay_parser = commands.add_parser(
        "replay", help="Replay IMU samples as JSON lines"
    )
    replay_parser.add_argument("capture", type=Path)
    replay_parser.add_argument(
        "--speed",
        type=float,
        default=0,
        help="0: immediate; 1: recorded receive timing; 2: twice as fast",
    )
    replay_parser.add_argument(
        "--acceleration-unit",
        choices=("g", "m/s2"),
        help="Explicitly interpret acceleration and emit processed SI samples alongside raw data; gyro assumed rad/s",
    )
    orientation = commands.add_parser(
        "orientation", help="Integrate gyro and compare with phone orientation"
    )
    orientation.add_argument("capture", type=Path)
    orientation.add_argument(
        "--max-gap",
        type=float,
        default=0.1,
        help="Reset integration across gaps larger than this many seconds",
    )
    orientation.add_argument(
        "--inverse-reference",
        action="store_true",
        help="Interpret phone quaternion as world-to-body",
    )
    orientation.add_argument("--plot", action="store_true")
    orientation.add_argument("--plot-output", type=Path)
    orientation.add_argument(
        "--output", type=Path, help="Save per-sample estimates as JSONL"
    )
    orientation.add_argument(
        "--viewer",
        action="store_true",
        help="Open linked gyro/phone orientation comparison",
    )
    orientation.add_argument(
        "--rrd", type=Path, help="Export orientation comparison to Rerun recording"
    )
    orientation.add_argument(
        "--export-log",
        type=Path,
        help="Export orientation comparison to portable viewer directory",
    )
    position = commands.add_parser(
        "dead-reckoning", help="Explore acceleration integration and drift"
    )
    position.add_argument("capture", type=Path)
    position.add_argument("--acceleration-unit", choices=("g", "m/s2"), required=True)
    position.add_argument(
        "--bias-interval",
        type=float,
        nargs=2,
        metavar=("START", "END"),
        help="Stationary interval in source seconds from first sample; user must verify stationarity",
    )
    position.add_argument("--max-gap", type=float, default=0.1)
    position.add_argument(
        "--stationary",
        type=float,
        nargs=2,
        action="append",
        default=[],
        metavar=("START", "END"),
        help="Repeat for explicit stationary intervals; hard-reset velocity and hold position",
    )
    position.add_argument("--plot", action="store_true")
    position.add_argument("--plot-output", type=Path)
    position.add_argument("--output", type=Path)
    position.add_argument(
        "--viewer", action="store_true", help="Open XY path and position diagnostics"
    )
    position.add_argument(
        "--rrd", type=Path, help="Export position viewer recording without a GUI"
    )
    position.add_argument(
        "--export-log", type=Path, help="Export portable position viewer log"
    )
    position.add_argument(
        "--route",
        type=Path,
        help="User route.json sketch, shown separately without inferred alignment",
    )
    position.add_argument(
        "--summary-output",
        type=Path,
        help="Save selected and uncorrected integration summary",
    )
    board = commands.add_parser(
        "calibration-target", help="Generate printable A4 checkerboard"
    )
    board.add_argument(
        "--output-dir", type=Path, default=Path("docs/calibration_target")
    )
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
    visual = commands.add_parser(
        "visualize", help="Explore a recorded capture with the standalone Rerun viewer"
    )
    visual.add_argument("capture", type=Path)
    visual.add_argument("--rrd", type=Path, help="Export .rrd without opening a GUI")
    visual.add_argument(
        "--export-log",
        type=Path,
        help="Export portable viewer log directory instead of Rerun",
    )
    visual.add_argument(
        "--depth-metres-per-unit",
        type=float,
        help="Explicit metric depth scale, e.g. 0.001 for tested rear millimetres",
    )
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
        choices=(0, 90, 180, 270),
        default=90,
        help="Clockwise overlay rotation; tracking coordinates stay unchanged",
    )
    tracking.add_argument("--viewer", action="store_true")
    tracking.add_argument("--rrd", type=Path)
    geometry = commands.add_parser(
        "two-view", help="Uncalibrated geometry from shared feature tracks"
    )
    geometry.add_argument("capture", type=Path)
    geometry.add_argument("--tracks", type=Path, required=True)
    geometry.add_argument("--output-dir", type=Path, required=True)
    geometry.add_argument("--frame-a", type=int, default=0)
    geometry.add_argument("--frame-b", type=int, default=10)
    geometry.add_argument(
        "--display-rotation", type=int, choices=(0, 90, 180, 270), default=90
    )
    motion = commands.add_parser(
        "camera-motion",
        help="Calibrated two-frame rotation and translation-direction diagnostics",
    )
    motion.add_argument("--tracks", type=Path, required=True)
    motion.add_argument("--calibration", type=Path, required=True)
    motion.add_argument("--output-dir", type=Path, required=True)
    motion.add_argument("--frame-a", type=int, default=1000)
    motion.add_argument("--frame-b", type=int, default=1010)
    vo = commands.add_parser(
        "visual-odometry", help="Experimental multiframe monocular trajectory"
    )
    vo.add_argument("--tracks", type=Path, required=True)
    vo.add_argument("--calibration", type=Path, required=True)
    vo.add_argument("--output-dir", type=Path, required=True)
    vo.add_argument("--stride", type=int, default=5)
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
    euroc = commands.add_parser(
        "import-euroc", help="Import EuRoC cam0 and ground truth"
    )
    euroc.add_argument("source", type=Path)
    euroc.add_argument("--output-dir", type=Path, required=True)
    evaluate = commands.add_parser(
        "evaluate-trajectory", help="Score segmented VO against imported ground truth"
    )
    evaluate.add_argument("run_dir", type=Path)
    evaluate.add_argument("--groundtruth", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--max-gap-ms", type=float, default=50.0)
    args = parser.parse_args()
    try:
        if args.command == "import-euroc":
            from fusionlab.recordings.euroc import import_sequence

            print(json.dumps(import_sequence(args.source, args.output_dir), indent=2))
            return
        if args.command == "evaluate-trajectory":
            from fusionlab.evaluation.ground_truth import evaluate_run

            result = evaluate_run(
                args.run_dir,
                args.groundtruth,
                args.output_dir,
                max_gap_ms=args.max_gap_ms,
            )
            print(json.dumps(result, indent=2))
            return
        if args.command == "diagnose-vo":
            from fusionlab.evaluation.vo_motion_diagnostics import diagnose_run

            alignment = (
                json.loads(args.camera_from_imu.read_text())
                if args.camera_from_imu
                else None
            )
            print(
                json.dumps(
                    diagnose_run(
                        args.run_dir,
                        args.capture,
                        args.output_dir,
                        camera_from_imu=alignment,
                        imu_offset_ms=args.imu_offset_ms,
                    ),
                    indent=2,
                )
            )
        elif args.command == "visual-odometry":
            from fusionlab.vision.odometry.runner import run_odometry

            factory = None
            if args.viewer or args.rrd:
                from fusion_viewer import ViewerClient
                from fusion_viewer.rerun_backend import RerunSink

                def factory():
                    return ViewerClient(
                        RerunSink(
                            output=args.rrd,
                            spawn=args.viewer,
                            layout_profile="odometry",
                        )
                    )

            print(
                json.dumps(
                    run_odometry(
                        args.tracks,
                        args.calibration,
                        args.output_dir,
                        stride=args.stride,
                        max_frames=args.max_frames,
                        viewer_factory=factory,
                        pose_acceptance=args.pose_acceptance,
                        keyframe_policy=args.keyframe_policy,
                        landmark_policy=args.landmark_policy,
                        bundle_adjustment=args.bundle_adjustment,
                        seed=args.seed,
                    ),
                    indent=2,
                )
            )
        elif args.command == "camera-motion":
            from fusionlab.vision.calibrated_motion import analyze_motion

            report = analyze_motion(
                args.tracks,
                args.calibration,
                args.output_dir,
                frame_a=args.frame_a,
                frame_b=args.frame_b,
            )
            print(
                json.dumps(
                    {
                        key: value
                        for key, value in report.items()
                        if key not in ("inlier_mask", "track_ids")
                    },
                    indent=2,
                )
            )
        elif args.command == "two-view":
            from fusionlab.vision.two_view import analyze_pair

            report = analyze_pair(
                args.capture,
                args.tracks,
                args.output_dir,
                frame_a=args.frame_a,
                frame_b=args.frame_b,
                display_rotation=args.display_rotation,
            )
            print(
                json.dumps(
                    {
                        key: value
                        for key, value in report.items()
                        if key
                        not in ("inlier_mask", "sampson_residual_px", "track_ids")
                    },
                    indent=2,
                )
            )
        elif args.command == "track-features":
            from fusionlab.vision.feature_tracking import track_capture, TrackingConfig

            viewer_factory = None
            if args.viewer or args.rrd:
                from fusion_viewer import ViewerClient
                from fusion_viewer.rerun_backend import RerunSink

                def viewer_factory():
                    return ViewerClient(RerunSink(output=args.rrd, spawn=args.viewer))

            report = track_capture(
                args.capture,
                args.output_dir,
                config=TrackingConfig(
                    max_features=args.max_features, max_width=args.max_width
                ),
                max_frames=args.max_frames,
                stream_id=args.camera_stream,
                viewer_factory=viewer_factory,
                display_rotation=args.display_rotation,
            )
            print(json.dumps(report, indent=2))
        elif args.command == "visualize":
            from fusion_viewer import ViewerClient
            from fusionlab.visualization.visualization_adapter import export_capture

            if args.export_log and args.rrd:
                raise ValueError("Choose --export-log or --rrd")
            if args.export_log:
                from fusion_viewer.recording import LogSink

                sink = LogSink(args.export_log)
            else:
                from fusion_viewer.rerun_backend import RerunSink

                sink = RerunSink(output=args.rrd, spawn=args.rrd is None)
            with ViewerClient(sink) as viewer:
                count = export_capture(
                    args.capture,
                    viewer,
                    depth_metres_per_unit=args.depth_metres_per_unit,
                )
            print(f"Loaded {count} sensor events")
        elif args.command == "calibration-target":
            from fusionlab.vision.calibration import target

            target(args.output_dir)
            print(f"Printable PDF/SVG saved to {args.output_dir.resolve()}")
        elif args.command == "calibrate-camera":
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
        elif args.command == "validate-calibration":
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
            per_image = ("images", "rejected_images")
            print(
                json.dumps(
                    {k: v for k, v in result.items() if k not in per_image}, indent=2
                )
            )
        elif args.command == "replay":
            asyncio.run(_replay(args.capture, args.speed, args.acceleration_unit))
        elif args.command == "dead-reckoning":
            from fusionlab.inertial.dead_reckoning import (
                integrate_position,
                summarize,
                plot as plot_position,
            )

            samples = list(read_samples(args.capture))
            convention = MeasurementConvention(args.acceleration_unit)
            baseline = integrate_position(samples, convention, max_gap_s=args.max_gap)
            result = integrate_position(
                samples,
                convention,
                max_gap_s=args.max_gap,
                bias_interval_s=(
                    tuple(args.bias_interval) if args.bias_interval else None
                ),
                stationary_intervals_s=tuple(tuple(pair) for pair in args.stationary),
            )
            report = {"selected": summarize(result), "uncorrected": summarize(baseline)}
            print(json.dumps(report, indent=2))
            if args.summary_output:
                args.summary_output.parent.mkdir(parents=True, exist_ok=True)
                args.summary_output.write_text(json.dumps(report, indent=2) + "\n")
            if args.viewer or args.rrd or args.export_log:
                from fusion_viewer import ViewerClient
                from fusionlab.visualization.visualization_adapter import (
                    export_position,
                )

                if args.export_log and (args.rrd or args.viewer):
                    raise ValueError("Choose portable export or Rerun output")
                if args.export_log:
                    from fusion_viewer.recording import LogSink

                    sink = LogSink(args.export_log)
                else:
                    from fusion_viewer.rerun_backend import RerunSink

                    sink = RerunSink(output=args.rrd, spawn=args.viewer)
                with ViewerClient(sink) as viewer:
                    export_position(result, viewer, route_path=args.route)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("w") as file:
                    for estimate in result.estimates:
                        file.write(json.dumps(asdict(estimate)) + "\n")
            if args.plot or args.plot_output:
                output = args.plot_output or args.capture / "dead_reckoning.png"
                plot_position(result, baseline, output)
                print(f"Plot saved to {output}")
        else:
            estimates = integrate(
                read_samples(args.capture),
                max_gap_s=args.max_gap,
                inverse_reference=args.inverse_reference,
            )
            print(json.dumps(summary(estimates), indent=2))
            if args.viewer or args.rrd or args.export_log:
                from fusion_viewer import ViewerClient
                from fusionlab.visualization.visualization_adapter import (
                    export_orientation,
                )

                if args.export_log and (args.rrd or args.viewer):
                    raise ValueError("Choose portable export or Rerun output")
                if args.export_log:
                    from fusion_viewer.recording import LogSink

                    sink = LogSink(args.export_log)
                else:
                    from fusion_viewer.rerun_backend import RerunSink

                    sink = RerunSink(output=args.rrd, spawn=args.viewer)
                with ViewerClient(sink) as viewer:
                    export_orientation(estimates, viewer)

            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("w") as file:
                    for estimate in estimates:
                        file.write(json.dumps(asdict(estimate)) + "\n")
            if args.plot or args.plot_output:
                output = args.plot_output or args.capture / "orientation.png"
                plot(estimates, output)
                print(f"Plot saved to {output}")
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(1, f"fusionlab: {exc}\n")
