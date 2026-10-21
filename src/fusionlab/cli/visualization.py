"""Capture visualization in the standalone viewer."""

import argparse
from pathlib import Path


def register(commands) -> None:
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
    visual.set_defaults(handler=run_visualize)


def run_visualize(args: argparse.Namespace) -> None:
    """Send every recorded event to a portable log, an .rrd file, or the GUI."""
    from fusion_viewer import ViewerClient
    from fusionlab.visualization.visualization_adapter import export_capture

    if args.export_log and args.rrd:
        raise ValueError("Choose --export-log or --rrd")
    if args.export_log:
        from fusion_viewer.recording import LogSink

        sink = LogSink(args.export_log)
    else:
        from fusion_viewer.rerun_backend import RerunSink

        # Without an .rrd target the GUI opens; with one, nothing is spawned.
        sink = RerunSink(output=args.rrd, spawn=args.rrd is None)
    with ViewerClient(sink) as viewer:
        count = export_capture(
            args.capture, viewer, depth_metres_per_unit=args.depth_metres_per_unit
        )
    print(f"Loaded {count} sensor events")
