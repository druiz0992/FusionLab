"""Viewer contracts, portable logs and Rerun sink."""

import tempfile
import unittest
from pathlib import Path

from fusion_viewer import (
    ViewerClient,
    RunSpec,
    FrameSpec,
    SampleTime,
    ScalarSample,
    OrientationSample,
    ImageSample,
    DepthSample,
    Map2D,
    Pose2D,
    Trajectory2D,
    Particles2D,
)
from fusion_viewer.recording import LogSink, read_log

from support import require


class ViewerTests(unittest.TestCase):
    def test_round_trip_and_contract_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.fvl"
            stamp = SampleTime("s", "clock", 0)
            image = ImageSample(
                entity_id="rgb/frame",
                time=stamp,
                width=1,
                height=1,
                encoding="image/jpeg",
                data=b"encoded",
            )
            with ViewerClient(LogSink(path)) as client:
                run = client.register_run(RunSpec("test", "test"))
                client.register_frame(run, FrameSpec("reference", "test axes"))
                client.publish(run, image)
                with self.assertRaises(ValueError):
                    client.publish(
                        run,
                        ScalarSample(
                            entity_id="rgb/frame", time=stamp, value=1, units="m"
                        ),
                    )
                with self.assertRaises(ValueError):
                    client.publish(
                        run,
                        OrientationSample(
                            entity_id="orientation/body",
                            time=stamp,
                            frame_id="missing",
                            quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                        ),
                    )
                with self.assertRaises(ValueError):
                    client.publish(
                        run,
                        DepthSample(
                            entity_id="depth/frame",
                            time=stamp,
                            width=2,
                            height=2,
                            data=b"xx",
                            encoding="16UC1",
                            byte_order="little",
                        ),
                    )
            records = list(read_log(path))
            self.assertEqual(records[-1][1], image)
            with self.assertRaises(RuntimeError):
                client.publish(run, image)
            asset = next((path / "assets").iterdir())
            asset.write_bytes(b"bad")
            with self.assertRaises(ValueError):
                list(read_log(path))

    def test_nonfinite_and_path_ids_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with ViewerClient(LogSink(Path(directory) / "run")) as client:
                with self.assertRaises(ValueError):
                    client.register_run(RunSpec("../bad", "bad"))
                run = client.register_run(RunSpec("run", "run"))
                with self.assertRaises(ValueError):
                    client.publish(
                        run,
                        ScalarSample(
                            entity_id="value",
                            time=SampleTime("s", "clock", 0),
                            value=float("nan"),
                            units="m",
                        ),
                    )

    def test_localization_roundtrip_and_validation(self):
        stamp = SampleTime("s", "clock", 0)
        common = dict(time=stamp, frame_id="map")
        messages = [
            Map2D(entity_id="room", walls=(((0.0, 0.0), (2.0, 0.0)),), **common),
            Pose2D(entity_id="pose", xy=(1.0, 2.0), heading_rad=0.5, **common),
            Trajectory2D(entity_id="path", points=((0.0, 0.0), (1.0, 2.0)), **common),
            Particles2D(
                entity_id="particles",
                poses=((1.0, 2.0, 0.5), (2.0, 3.0, 0.1)),
                weights=(2.0, 1.0),
                **common,
            ),
            Particles2D(entity_id="particles", poses=(), weights=(), **common),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene"
            with ViewerClient(LogSink(path)) as client:
                run = client.register_run(RunSpec("test", "test"))
                with self.assertRaises(ValueError):
                    client.publish(run, messages[0])
                client.register_frame(run, FrameSpec("map", "XY, heading CCW"))
                for message in messages:
                    client.publish(run, message)
                invalid = [
                    Map2D(entity_id="bad", walls=(((0.0, 0.0),),), **common),
                    Pose2D(
                        entity_id="bad",
                        xy=(0.0, 0.0),
                        heading_rad=float("nan"),
                        **common,
                    ),
                    Trajectory2D(
                        entity_id="bad", points=((0.0, float("inf")),), **common
                    ),
                    Particles2D(
                        entity_id="bad", poses=((0.0, 0.0, 0.0),), weights=(), **common
                    ),
                    Particles2D(
                        entity_id="bad",
                        poses=((0.0, 0.0, 0.0),),
                        weights=(-1.0,),
                        **common,
                    ),
                    Particles2D(
                        entity_id="bad",
                        poses=((0.0, 0.0, 0.0),),
                        weights=(0.0,),
                        **common,
                    ),
                ]
                for message in invalid:
                    with self.subTest(message=message):
                        with self.assertRaises(ValueError):
                            client.publish(run, message)
                client.register_frame(run, FrameSpec("other", "XY"))
                with self.assertRaises(ValueError):
                    client.publish(
                        run,
                        Pose2D(
                            entity_id="pose",
                            time=stamp,
                            frame_id="other",
                            xy=(0.0, 0.0),
                            heading_rad=0.0,
                        ),
                    )
            self.assertEqual(
                [
                    value
                    for _, value in read_log(path)
                    if isinstance(value, (Map2D, Pose2D, Trajectory2D, Particles2D))
                ],
                messages,
            )

    def test_arbitrary_frame_units_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arbitrary"
            with ViewerClient(LogSink(path)) as client:
                run = client.register_run(RunSpec("vo", "vo"))
                client.register_frame(
                    run, FrameSpec("segment", "Initial camera XZ", units="arbitrary")
                )
            frame = list(read_log(path))[-1][1]
            self.assertEqual(frame.units, "arbitrary")

    def test_rerun_layout_selects_explicit_run_subtrees(self):
        try:
            import rerun
            import rerun.blueprint as rb
        except ImportError:
            self.skipTest("Optional viewer SDK unavailable")
        from unittest.mock import patch
        from fusion_viewer.rerun_backend import RerunSink

        with (
            patch.object(rerun, "RecordingStream") as stream,
            patch.object(rb, "Spatial2DView", wraps=rb.Spatial2DView) as view,
            patch.object(rb, "TimePanel", wraps=rb.TimePanel) as time_panel,
        ):
            sink = RerunSink(output=Path("/tmp/unused-viewer-test.rrd"))
            sink.register_run(RunSpec("first", "First"))
            self.assertEqual(
                view.call_args_list[0].kwargs["contents"], ["/runs/first/rgb/**"]
            )
            sink.register_run(RunSpec("second", "Second"))
            self.assertEqual(
                view.call_args_list[2].kwargs["contents"],
                ["/runs/first/rgb/**", "/runs/second/rgb/**"],
            )
            self.assertEqual(
                view.call_args_list[3].kwargs["contents"],
                ["/runs/first/depth/**", "/runs/second/depth/**"],
            )
            time_panel.assert_called_with(timeline="playback")
            self.assertTrue(
                stream.return_value.send_blueprint.call_args.kwargs["make_active"]
            )
            sink.close()

    def test_odometry_layout_has_one_path_panel(self):
        try:
            import rerun
            import rerun.blueprint as rb
        except ImportError:
            self.skipTest("Optional viewer SDK unavailable")
        from unittest.mock import patch
        from fusion_viewer.rerun_backend import RerunSink

        with (
            patch.object(rerun, "RecordingStream"),
            patch.object(rb, "Spatial2DView", wraps=rb.Spatial2DView) as views,
        ):
            sink = RerunSink(
                output=Path("/tmp/unused-vo-layout.rrd"), layout_profile="odometry"
            )
            sink.register_run(RunSpec("vo", "vo"))
            names = [call.kwargs.get("name", "") for call in views.call_args_list]
            self.assertEqual(
                sum(name.startswith("Active segment") for name in names), 1
            )
            self.assertIn("Camera / tracking status", names)
            sink.close()

    def test_rerun_export_if_available(self):
        require("rerun")
        from fusion_viewer.rerun_backend import RerunSink

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.rrd"
            with ViewerClient(RerunSink(output=path)) as client:
                run = client.register_run(RunSpec("run", "run"))
                client.register_frame(run, FrameSpec("reference", "axes"))
                client.publish(
                    run,
                    OrientationSample(
                        entity_id="orientation/body",
                        time=SampleTime("s", "clock", 0),
                        frame_id="reference",
                        quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                    ),
                )
                client.publish(
                    run,
                    DepthSample(
                        entity_id="depth/frame",
                        time=SampleTime("s", "clock", 1),
                        width=1,
                        height=1,
                        data=b"\xe8\x03",
                        encoding="16UC1",
                        byte_order="little",
                        metres_per_unit=0.001,
                    ),
                )
                client.register_frame(run, FrameSpec("map", "XY"))
                stamp = SampleTime("s", "clock", 2)
                for message in (
                    Map2D(
                        entity_id="room",
                        time=stamp,
                        frame_id="map",
                        walls=(((0.0, 0.0), (1.0, 0.0)),),
                    ),
                    Pose2D(
                        entity_id="pose",
                        time=stamp,
                        frame_id="map",
                        xy=(0.0, 0.0),
                        heading_rad=0.2,
                    ),
                    Trajectory2D(
                        entity_id="path",
                        time=stamp,
                        frame_id="map",
                        points=((0.0, 0.0), (1.0, 1.0)),
                    ),
                    Particles2D(
                        entity_id="particles",
                        time=stamp,
                        frame_id="map",
                        poses=((0.0, 0.0, 0.0),),
                        weights=(1.0,),
                    ),
                    Particles2D(
                        entity_id="particles",
                        time=stamp,
                        frame_id="map",
                        poses=(),
                        weights=(),
                    ),
                ):
                    client.publish(run, message)
            self.assertGreater(path.stat().st_size, 0)
