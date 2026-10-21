import argparse
from pathlib import Path
from .api import ViewerClient
from .models import RunSpec, FrameSpec
from .recording import read_log


def main():
    parser = argparse.ArgumentParser(description="Standalone localization viewer")
    commands = parser.add_subparsers(dest="command", required=True)
    opened = commands.add_parser("open", help="Open a portable viewer log with Rerun")
    opened.add_argument("recording", type=Path)
    opened.add_argument(
        "--rrd", type=Path, help="Export Rerun recording instead of opening GUI"
    )
    args = parser.parse_args()
    try:
        from .rerun_backend import RerunSink

        with ViewerClient(RerunSink(output=args.rrd, spawn=args.rrd is None)) as viewer:
            # Replay declarations before their messages, using the same validation path
            # as a live producer. The context manager flushes and closes the backend.
            for run, value in read_log(args.recording):
                if isinstance(value, RunSpec):
                    viewer.register_run(value)
                elif isinstance(value, FrameSpec):
                    viewer.register_frame(run, value)
                else:
                    viewer.publish(run, value)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"fusion-viewer: {exc}\n")
