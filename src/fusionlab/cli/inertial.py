"""IMU baseline commands: gyro orientation and exploratory dead reckoning.

These are separate baselines, not inputs to visual odometry.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from fusionlab.cli._common import open_viewer_sink, write_jsonl


def register(commands) -> None:
    _register_orientation(commands)
    _register_dead_reckoning(commands)


def _register_orientation(commands) -> None:
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
    orientation.set_defaults(handler=run_orientation)


def _register_dead_reckoning(commands) -> None:
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
    position.set_defaults(handler=run_dead_reckoning)


def _wants_viewer(args: argparse.Namespace) -> bool:
    return bool(args.viewer or args.rrd or args.export_log)


def run_orientation(args: argparse.Namespace) -> None:
    """Integrate gyro rates and compare with the phone's own orientation."""
    from fusionlab.inertial.orientation import integrate, plot, summary
    from fusionlab.recordings.replay import read_samples

    estimates = integrate(
        read_samples(args.capture),
        max_gap_s=args.max_gap,
        inverse_reference=args.inverse_reference,
    )
    print(json.dumps(summary(estimates), indent=2))
    if _wants_viewer(args):
        from fusion_viewer import ViewerClient
        from fusionlab.visualization.visualization_adapter import export_orientation

        sink = open_viewer_sink(args.export_log, args.rrd, args.viewer)
        with ViewerClient(sink) as viewer:
            export_orientation(estimates, viewer)
    if args.output:
        write_jsonl(args.output, (asdict(estimate) for estimate in estimates))
    if args.plot or args.plot_output:
        output = args.plot_output or args.capture / "orientation.png"
        plot(estimates, output)
        print(f"Plot saved to {output}")


def run_dead_reckoning(args: argparse.Namespace) -> None:
    """Integrate acceleration twice, with and without the chosen corrections."""
    from fusionlab.inertial.dead_reckoning import integrate_position
    from fusionlab.inertial.dead_reckoning import plot as plot_position
    from fusionlab.inertial.dead_reckoning import summarize
    from fusionlab.inertial.measurements import MeasurementConvention
    from fusionlab.recordings.replay import read_samples

    samples = list(read_samples(args.capture))
    convention = MeasurementConvention(args.acceleration_unit)
    # The uncorrected run is reported alongside to show what corrections change.
    baseline = integrate_position(samples, convention, max_gap_s=args.max_gap)
    result = integrate_position(
        samples,
        convention,
        max_gap_s=args.max_gap,
        bias_interval_s=tuple(args.bias_interval) if args.bias_interval else None,
        stationary_intervals_s=tuple(tuple(pair) for pair in args.stationary),
    )
    report = {"selected": summarize(result), "uncorrected": summarize(baseline)}
    print(json.dumps(report, indent=2))
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(report, indent=2) + "\n")
    if _wants_viewer(args):
        from fusion_viewer import ViewerClient
        from fusionlab.visualization.visualization_adapter import export_position

        sink = open_viewer_sink(args.export_log, args.rrd, args.viewer)
        with ViewerClient(sink) as viewer:
            export_position(result, viewer, route_path=args.route)
    if args.output:
        write_jsonl(args.output, (asdict(estimate) for estimate in result.estimates))
    if args.plot or args.plot_output:
        output = args.plot_output or args.capture / "dead_reckoning.png"
        plot_position(result, baseline, output)
        print(f"Plot saved to {output}")
