import argparse
import asyncio
import json
import sys
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from math import isfinite

from . import (
    ImuSample,
    GpsSample,
    CameraSample,
    DepthSample,
    SensorEvent,
    ReceiverConfig,
    SensorStreamError,
    SensorStreamSource,
)
from .recording import SessionRecorder


async def _write_recording(recorder: SessionRecorder, event: SensorEvent) -> None:
    # Keep disk I/O off the network event loop. Await completion even when
    # Ctrl+C cancels consumption, so a dequeued sample cannot be lost.
    writing = asyncio.create_task(asyncio.to_thread(recorder.write, event))
    try:
        await asyncio.shield(writing)
    except asyncio.CancelledError:
        await writing
        raise


async def _listen(args: argparse.Namespace) -> None:
    config = ReceiverConfig(
        bind_host=args.host,
        bind_port=args.port,
        queue_capacity=args.queue_capacity,
        overflow_policy=args.overflow_policy,
        transport=getattr(args, "transport", "wifi"),
        usb_port=getattr(args, "usb_port", 2345),
        usb_udid=getattr(args, "usb_udid", None),
    )
    recording = (
        SessionRecorder(args.output_dir, config) if args.record else nullcontext(None)
    )
    with recording as recorder:
        await _receive(args, config, recorder)


async def _receive(
    args: argparse.Namespace, config: ReceiverConfig, recorder: SessionRecorder | None
) -> None:
    source = SensorStreamSource(config)
    try:
        async with source:
            try:
                deadline = asyncio.timeout(args.duration)
                try:
                    async with deadline:
                        await _consume(args, source, recorder)
                except TimeoutError:
                    if not deadline.expired():
                        raise
                    print(
                        f"Duration reached ({args.duration:g} seconds); stopping.",
                        flush=True,
                    )
            finally:
                # Stop producers before draining. Context exit otherwise clears
                # queued measurements before the recorder can persist them.
                pending = await source.aclose(drain=True)
                for event in pending:
                    if recorder:
                        recorder.write(event)
    finally:
        if recorder:
            recorder.set_stats(source.stats())


async def _consume(
    args: argparse.Namespace,
    source: SensorStreamSource,
    recorder: SessionRecorder | None,
) -> None:
    preview = None
    if getattr(args, "live", False):
        from .live import LiveCameraPreview

        preview = LiveCameraPreview(source.stats, args.display_rotation)
    if getattr(args, "transport", "wifi") == "usb":
        print(f"Waiting for SensorStream USB device on port {source.port}", flush=True)
    else:
        print(
            f"Listening on {args.host}:{source.port}; waiting for SensorStream",
            flush=True,
        )
    if recorder:
        print(f"Recording to {recorder.path.resolve()}", flush=True)

    async def report() -> None:
        while True:
            await asyncio.sleep(1)
            print(
                json.dumps({"stats": asdict(source.stats())}),
                file=sys.stderr,
                flush=True,
            )

    async def consume() -> None:
        async for event in source.events():
            if preview and isinstance(event, CameraSample):
                preview.submit(event)
            if recorder:
                await _write_recording(recorder, event)
            if (
                not isinstance(event, (ImuSample, GpsSample, CameraSample, DepthSample))
                or args.print_samples
            ):
                data = asdict(event)
                if isinstance(event, CameraSample):
                    data["image_size_bytes"] = len(data.pop("image_data"))
                if isinstance(event, DepthSample):
                    data["depth_size_bytes"] = len(data.pop("depth_data"))
                print(json.dumps({"type": type(event).__name__, **data}), flush=True)

    # Run status reporting, event consumption and optional preview concurrently.
    # Their cleanup below cancels and awaits each task before closing the window.
    reporter = asyncio.create_task(report())
    consumer = asyncio.create_task(consume())
    rendering = asyncio.create_task(preview.run()) if preview else None
    try:
        if rendering:
            done, _ = await asyncio.wait(
                (consumer, rendering), return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                task.result()
        else:
            await consumer
    finally:
        reporter.cancel()
        consumer.cancel()
        if rendering:
            rendering.cancel()
        await asyncio.gather(
            reporter,
            consumer,
            *([rendering] if rendering else []),
            return_exceptions=True,
        )
        if preview:
            preview.close()


def _positive_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Duration must be a positive number of seconds"
        ) from exc
    if not isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError(
            "Duration must be finite and greater than zero"
        )
    return seconds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Receive SensorStream IMU and GPS data without ROS2"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    listen = commands.add_parser("listen", help="Start the Wi-Fi TCP receiver")
    listen.add_argument("--host", default="0.0.0.0")
    listen.add_argument("--transport", choices=("wifi", "usb"), default="wifi")
    listen.add_argument(
        "--usb-port",
        type=int,
        default=2345,
        help="SensorStream device port in USB mode",
    )
    listen.add_argument(
        "--usb-udid", help="Select a USB device when multiple are connected"
    )
    listen.add_argument("--port", type=int, default=5678)
    listen.add_argument("--queue-capacity", type=int, default=4096)
    listen.add_argument(
        "--overflow-policy", choices=("raise", "drop_oldest"), default="raise"
    )
    listen.add_argument("--print-samples", action="store_true")
    listen.add_argument(
        "--live", action="store_true", help="Open a live RGB camera preview window"
    )
    listen.add_argument(
        "--display-rotation",
        type=int,
        choices=(0, 90, 180, 270),
        default=0,
        help="Clockwise camera display rotation; recorded images are unchanged",
    )
    listen.add_argument(
        "--duration",
        type=_positive_seconds,
        metavar="SECONDS",
        help="Stop after this many seconds from listener startup (default: unlimited)",
    )
    listen.add_argument(
        "--record",
        action="store_true",
        help="Save decoded events in a new session folder",
    )
    listen.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/sessions"),
        help="Recording root (default: data/sessions; requires --record)",
    )
    inspect = commands.add_parser("inspect", help="Summarize a recorded capture")
    inspect.add_argument(
        "capture", type=Path, help="Folder containing metadata.json and events.jsonl"
    )
    inspect.add_argument(
        "--plot", action="store_true", help="Save analysis.png in the capture folder"
    )
    inspect.add_argument(
        "--plot-output", type=Path, help="Save plot at this path (implies --plot)"
    )
    replay_camera = commands.add_parser(
        "replay-camera",
        help="Create a browser replay comparing source and receive timing",
    )
    replay_camera.add_argument("capture", type=Path)
    replay_camera.add_argument(
        "--output", type=Path, help="HTML output (default: capture/camera_replay.html)"
    )
    depth_analysis = commands.add_parser(
        "analyze-depth",
        help="Analyze raw depth validity and timestamp-paired RGB edges",
    )
    depth_analysis.add_argument("capture", type=Path)
    depth_analysis.add_argument("--output-dir", type=Path)
    depth_analysis.add_argument("--tolerance-ms", type=float, default=20.0)
    depth_analysis.add_argument(
        "--camera-stream", help="Select RGB stream ID if multiple exist in one session"
    )
    depth_analysis.add_argument(
        "--edge-threshold",
        type=_positive_seconds,
        default=100.0,
        help="Depth discontinuity threshold in raw units (default 100)",
    )
    depth_analysis.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "analyze-depth":
            from .inspection.depth_analysis import analyze

            output = args.output_dir or args.capture / "depth_analysis"
            report = analyze(
                args.capture,
                output,
                tolerance_ms=args.tolerance_ms,
                camera_stream=args.camera_stream,
                edge_threshold=args.edge_threshold,
                plots=args.plot,
            )
            print(json.dumps(report, indent=2))
            print(f"Analysis saved to {output.resolve()}", file=sys.stderr)
        elif args.command == "replay-camera":
            from .inspection.camera_replay import write_replay

            output = args.output or args.capture / "camera_replay.html"
            count = write_replay(args.capture, output)
            print(
                f"Replay created for {count} frames: {output.resolve()}\nOpen this HTML file in your browser."
            )
        elif args.command == "inspect":
            from .inspection import load_capture, summarize, plot_capture

            metadata, events = load_capture(args.capture)
            print(json.dumps(summarize(metadata, events), indent=2))
            if args.plot or args.plot_output:
                output = args.plot_output or args.capture / "analysis.png"
                plot_capture(events, output, capture=args.capture)
                print(f"Plot saved to {output.resolve()}", file=sys.stderr)
        else:
            asyncio.run(_listen(args))
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError, SensorStreamError) as exc:
        parser.exit(1, f"sensor-stream: {exc}\n")
