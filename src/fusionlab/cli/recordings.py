"""Commands that read recordings: IMU replay and EuRoC import."""

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path


def register(commands) -> None:
    replay = commands.add_parser("replay", help="Replay IMU samples as JSON lines")
    replay.add_argument("capture", type=Path)
    replay.add_argument(
        "--speed",
        type=float,
        default=0,
        help="0: immediate; 1: recorded receive timing; 2: twice as fast",
    )
    replay.add_argument(
        "--acceleration-unit",
        choices=("g", "m/s2"),
        help="Explicitly interpret acceleration and emit processed SI samples alongside raw data; gyro assumed rad/s",
    )
    replay.set_defaults(handler=run_replay)

    euroc = commands.add_parser(
        "import-euroc", help="Import EuRoC cam0 and ground truth"
    )
    euroc.add_argument("source", type=Path)
    euroc.add_argument("--output-dir", type=Path, required=True)
    euroc.set_defaults(handler=run_import_euroc)


def run_replay(args: argparse.Namespace) -> None:
    asyncio.run(_print_replay(args.capture, args.speed, args.acceleration_unit))


async def _print_replay(
    path: Path, speed: float, acceleration_unit: str | None = None
) -> None:
    """Print raw IMU samples, or processed SI samples when a unit is given."""
    from fusionlab.inertial.measurements import MeasurementConvention, convert_imu
    from fusionlab.recordings.replay import replay

    convention = MeasurementConvention(acceleration_unit) if acceleration_unit else None
    async for sample in replay(path, speed=speed):
        if convention:
            processed = asdict(convert_imu(sample, convention))
            print(json.dumps({"type": "ProcessedImuSample", **processed}), flush=True)
        else:
            print(json.dumps({"type": "ImuSample", **asdict(sample)}), flush=True)


def run_import_euroc(args: argparse.Namespace) -> None:
    from fusionlab.recordings.euroc import import_sequence

    print(json.dumps(import_sequence(args.source, args.output_dir), indent=2))
