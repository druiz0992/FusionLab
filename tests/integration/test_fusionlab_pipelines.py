"""Multi-stage fusionlab paths: import, tracking, VO, evaluation, CLI."""

import contextlib
import io
import json
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

from fusionlab.vision.tracks import TrackFrame

from support.vision import calibration_json, track_row, translating_camera_rows
from support import require

# Synthetic camera: 1280×720 pixels, no distortion, 9×6 inner corners of 20 mm.
CAMERA_MATRIX = [[900.0, 0.0, 640.0], [0.0, 920.0, 360.0], [0.0, 0.0, 1.0]]


CALIBRATION_JSON = {
    "camera_matrix": CAMERA_MATRIX,
    "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
    "distortion_order": ["k1", "k2", "p1", "p2", "k3"],
    "inner_corners": [9, 6],
    "measured_square_mm": 20,
    "image_size": [1280, 720],
}


def camera_event(image_path: str, data: bytes, stream_id: str = "rgb") -> dict:
    """A CameraSample event with the integrity fields the validator checks."""
    return {
        "type": "CameraSample",
        "stream_id": stream_id,
        "image_path": image_path,
        "image_size_bytes": len(data),
        "image_sha256": sha256(data).hexdigest(),
    }


def write_capture(root: Path, images: dict[str, bytes], stream_id: str = "rgb"):
    """Write a minimal capture: image files plus their events.jsonl rows."""
    for relative, data in images.items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_bytes(data)
    rows = [
        camera_event(relative, data, stream_id) for relative, data in images.items()
    ]
    (root / "events.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


class EurocBenchmarkTests(unittest.TestCase):
    def setUp(self):
        try:
            import cv2
            import numpy as np
            import yaml
            from scipy.spatial.transform import Rotation
        except ImportError:
            self.skipTest("benchmark extras unavailable")
        self.np, self.cv, self.yaml, self.Rotation = np, cv2, yaml, Rotation

    def fixture(self, root):
        np = self.np
        mav = root / "source/mav0"
        camera = mav / "cam0"
        (camera / "data").mkdir(parents=True)
        gt = mav / "state_groundtruth_estimate0"
        gt.mkdir()
        transform = np.eye(4)
        transform[0, 3] = 0.2
        sensor = dict(
            camera_model="pinhole",
            distortion_model="radial-tangential",
            resolution=[96, 64],
            intrinsics=[80, 80, 48, 32],
            distortion_coefficients=[0, 0, 0, 0],
            T_BS=dict(data=transform.ravel().tolist()),
        )
        (camera / "sensor.yaml").write_text(self.yaml.safe_dump(sensor))
        epoch = 1400000000000000000
        frames = []
        states = []
        for i in range(8):
            stamp = epoch + i * 10000000
            filename = f"{stamp}.png"
            image = np.random.default_rng(4).integers(0, 255, (64, 96), dtype=np.uint8)
            self.cv.imwrite(str(camera / "data" / filename), image)
            frames.append(f"{stamp},{filename}\n")
            states.append(f"{stamp},{i*0.1},{(i*0.1)**2},0,1,0,0,0,0,0,0\n")
        (camera / "data.csv").write_text("#timestamp,filename\n" + "".join(frames))
        (gt / "data.csv").write_text("#timestamp,p,q\n" + "".join(states))
        return mav, epoch

    def test_import_tracks_and_odometry_keep_source_and_timestamps(self):
        from fusionlab.recordings.euroc import import_sequence
        from fusionlab.vision.feature_tracking import track_capture
        from fusionlab.vision.odometry.runner import run_odometry

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, epoch = self.fixture(root)
            output = root / "imported"
            meta = import_sequence(source, output)
            self.assertEqual(meta["frames"], 8)
            events = [
                json.loads(line)
                for line in (output / "events.jsonl").read_text().splitlines()
            ]
            self.assertEqual(events[0]["source_time_ns"], epoch)
            self.assertEqual(events[1]["received_monotonic_ns"], 10000000)
            filename = f"{epoch}.png"
            self.assertEqual(
                (source / "cam0/data" / filename).read_bytes(),
                (output / "camera" / filename).read_bytes(),
            )
            track_capture(output, root / "tracks", display_rotation=0)
            run_odometry(
                root / "tracks/tracks.jsonl",
                output / "calibration.json",
                root / "run",
                stride=1,
            )
            self.assertEqual(
                len((root / "run/poses.jsonl").read_text().splitlines()), 8
            )
            with self.assertRaises(FileExistsError):
                import_sequence(source, output)

    def test_interpolated_camera_reference_sim3_and_coverage(self):
        from fusionlab.recordings.euroc import import_sequence
        from fusionlab.evaluation.ground_truth import evaluate_run

        np = self.np
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, epoch = self.fixture(root)
            output = root / "imported"
            import_sequence(source, output)
            run = root / "run"
            run.mkdir()
            rotation = self.Rotation.from_euler("z", 35, degrees=True).as_matrix()
            records = []
            for i in range(6):
                t = (i + 0.5) * 0.01
                # Linear interpolation of body position, then apply camera lever arm.
                a = i * 0.1
                b = (i + 1) * 0.1
                camera = np.array([(a + b) / 2 + 0.2, (a * a + b * b) / 2, 0])
                estimate = rotation.T @ (camera - np.array([1, 2, 3])) / 2.5
                records.append(
                    dict(
                        source_time_ns=epoch + int(round(t * 1e9)),
                        status="tracking",
                        segment=1,
                        camera_center_arbitrary=estimate.tolist(),
                        world_to_camera=rotation.tolist(),
                        refined_median_inlier_error_px=None,
                    )
                )
            records.append(
                dict(source_time_ns=epoch + 65000000, status="tracking_lost", segment=1)
            )
            records.append(
                dict(source_time_ns=epoch + 90000000, status="tracking_lost", segment=1)
            )
            (run / "poses.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in records)
            )
            result = evaluate_run(run, output / "groundtruth.json", root / "score")
            self.assertEqual(result["matched_tracking_frames"], 6)
            self.assertEqual(result["reference_available_frames"], 7)
            self.assertEqual(result["continuity"]["tracking_losses"], 2)
            self.assertAlmostEqual(result["segments"][0]["scale"], 2.5)
            self.assertLess(result["segments"][0]["position_rmse_m"], 1e-10)
            self.assertLess(result["segments"][0]["rotation_rmse_deg"], 1e-8)
            rejected = evaluate_run(
                run, output / "groundtruth.json", root / "gap", max_gap_ms=1
            )
            self.assertEqual(rejected["matched_tracking_frames"], 0)

    def test_import_cli_dispatch(self):
        import contextlib
        import io
        from unittest.mock import patch
        from fusionlab.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, _ = self.fixture(root)
            with patch(
                "sys.argv",
                [
                    "fusionlab",
                    "import-euroc",
                    str(source),
                    "--output-dir",
                    str(root / "capture"),
                ],
            ):
                with contextlib.redirect_stdout(io.StringIO()) as stdout:
                    main()
            self.assertEqual(json.loads(stdout.getvalue())["frames"], 8)


class FeatureTrackingTests(unittest.TestCase):
    def setUp(self):
        self.cv, self.np = require("cv2", "numpy")

    def test_capture_export_and_asset_integrity(self):
        import json
        import tempfile
        from pathlib import Path
        from hashlib import sha256
        from fusionlab.vision.feature_tracking import track_capture
        from fusion_viewer import ViewerClient, ImageSample
        from fusion_viewer.recording import LogSink, read_log

        cv, np = self.cv, self.np
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = root / "capture"
            capture.mkdir()
            events = []
            for i in range(2):
                image = np.random.default_rng(8).integers(
                    0, 256, (120, 160, 3), dtype=np.uint8
                )
                _, encoded = cv.imencode(".jpg", image)
                raw = encoded.tobytes()
                (capture / f"{i}.jpg").write_bytes(raw)
                events.append(
                    dict(
                        type="CameraSample",
                        stream_id="s",
                        source_time_ns=i * 33_000_000,
                        received_monotonic_ns=i * 33_000_000,
                        image_path=f"{i}.jpg",
                        image_size_bytes=len(raw),
                        image_sha256=sha256(raw).hexdigest(),
                    )
                )
            (capture / "events.jsonl").write_text(
                "".join(json.dumps(e) + "\n" for e in events)
            )
            with ViewerClient(LogSink(root / "viewer")) as viewer:
                report = track_capture(capture, root / "analysis", viewer=viewer)
            self.assertEqual(report["frames"], 2)
            values = [
                v for _, v in read_log(root / "viewer") if isinstance(v, ImageSample)
            ]
            self.assertEqual(len(values), 2)
            self.assertEqual((values[0].width, values[0].height), (120, 160))
            track_capture(capture, root / "unrotated", display_rotation=0)
            rotated = json.loads(
                (root / "analysis" / "tracks.jsonl").read_text().splitlines()[0]
            )
            unrotated = json.loads(
                (root / "unrotated" / "tracks.jsonl").read_text().splitlines()[0]
            )
            self.assertEqual(rotated["tracks"], unrotated["tracks"])
            self.assertEqual(rotated["display_rotation_clockwise_deg"], 90)
            self.assertEqual(unrotated["display_rotation_clockwise_deg"], 0)
            # A viewer that writes inside the analysis directory must open only
            # after the tracker reserves that directory.
            from contextlib import contextmanager

            opened = []

            @contextmanager
            def nested_export():
                opened.append(True)
                (root / "nested" / "replay.rrd").write_bytes(b"test export")
                yield None

            track_capture(capture, root / "nested", viewer_factory=nested_export)
            self.assertEqual(opened, [True])
            with self.assertRaises(FileExistsError):
                track_capture(capture, root / "nested", viewer_factory=nested_export)
            self.assertEqual(opened, [True])
            (capture / "0.jpg").write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                track_capture(capture, root / "bad")


class ValidateCalibrationIntegrationTests(unittest.TestCase):
    """Render a board with known intrinsics, then run the CLI end to end."""

    @classmethod
    def setUpClass(cls):
        cls.cv, cls.np = require("cv2", "numpy")

    def render_board_png(self) -> bytes:
        """Warp a 10×7-square checkerboard texture into the synthetic camera.

        Texture pixel centers map to board metres; inner corner (0, 0) lies on
        the edge between texture pixels, half a pixel before the first square.
        """
        cv, np = self.cv, self.np
        square_px, margin_px = 40, 40
        texture = np.full(
            (7 * square_px + 2 * margin_px, 10 * square_px + 2 * margin_px),
            255,
            np.uint8,
        )
        for row in range(7):
            for column in range(10):
                if (row + column) % 2 == 0:
                    top = margin_px + row * square_px
                    left = margin_px + column * square_px
                    texture[top : top + square_px, left : left + square_px] = 0
        # Board metres from texture pixels: corner (0, 0) is one square in.
        square_m = CALIBRATION_JSON["measured_square_mm"] / 1000
        origin_px = margin_px + square_px - 0.5
        texture_to_board = np.array(
            [
                [square_m / square_px, 0, -origin_px * square_m / square_px],
                [0, square_m / square_px, -origin_px * square_m / square_px],
                [0, 0, 1],
            ]
        )
        # Plane homography K [r1 r2 t] for a slightly tilted board 0.5 m away.
        rotation, _ = cv.Rodrigues(np.array([0.2, -0.15, 0.05]))
        translation = np.array([-0.08, -0.05, 0.5])
        board_to_image = np.array(CAMERA_MATRIX) @ np.column_stack(
            (rotation[:, 0], rotation[:, 1], translation)
        )
        image = cv.warpPerspective(
            texture,
            board_to_image @ texture_to_board,
            (1280, 720),
            flags=cv.INTER_LINEAR,
            borderValue=255,
        )
        ok, encoded = cv.imencode(".png", image)
        self.assertTrue(ok)
        return encoded.tobytes()

    def run_cli(self, *arguments: str) -> dict:
        from fusionlab.cli import main

        with patch("sys.argv", ["fusionlab", "validate-calibration", *arguments]):
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                main()
        return json.loads(stdout.getvalue())

    def test_cli_scores_rendered_board_and_writes_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture"
            board = self.render_board_png()
            write_capture(capture, {"camera/0.png": board, "camera/1.png": board})
            calibration = root / "calibration.json"
            calibration.write_text(json.dumps(CALIBRATION_JSON))
            output = root / "validation"

            printed = self.run_cli(
                str(capture),
                str(calibration),
                "--output-dir",
                str(output),
                "--stride",
                "1",
            )

            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["detected_views"], 2)
            self.assertFalse(summary["intrinsics_refitted"])
            # Subpixel detection on a rendered board; the true model fits closely.
            self.assertLess(summary["median_heldout_corner_rms_px"], 0.3)
            self.assertNotIn("images", printed)
            self.assertEqual(printed["detected_views"], 2)
            # Inputs are never modified.
            self.assertEqual(json.loads(calibration.read_text()), CALIBRATION_JSON)

    def test_cli_refuses_existing_output_directory(self):
        from fusionlab.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_capture(root / "capture", {"camera/0.png": self.render_board_png()})
            (root / "calibration.json").write_text(json.dumps(CALIBRATION_JSON))
            (root / "existing").mkdir()
            arguments = [
                "fusionlab",
                "validate-calibration",
                str(root / "capture"),
                str(root / "calibration.json"),
                "--output-dir",
                str(root / "existing"),
            ]
            with patch("sys.argv", arguments):
                with contextlib.redirect_stdout(io.StringIO()):
                    with contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as exit_info:
                            main()
            self.assertEqual(exit_info.exception.code, 1)


class MotionDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.cv, self.np = require("cv2", "numpy")

    def test_initializing_replay_does_not_fabricate_predictions_or_gyro(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from fusionlab.evaluation.vo_motion_diagnostics import diagnose_run
        from fusionlab.vision.odometry.visual_odometry import MonocularOdometry

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            run.mkdir()
            capture = root / "capture"
            capture.mkdir()
            calibration = calibration_json()
            (root / "calibration.json").write_text(json.dumps(calibration))
            tracker = MonocularOdometry(calibration)
            rows = [track_row(i * 100_000_000) for i in range(3)]
            (root / "tracks.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rows)
            )
            results = [
                {
                    **tracker.process(TrackFrame.from_json(row)).to_json(),
                    "frame_index": i,
                }
                for i, row in enumerate(rows)
            ]
            (run / "poses.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in results)
            )
            (run / "summary.json").write_text(
                json.dumps(
                    {
                        "tracks": str(root / "tracks.jsonl"),
                        "calibration": str(root / "calibration.json"),
                    }
                )
            )
            (capture / "events.jsonl").write_text("")
            with patch("fusionlab.evaluation.vo_motion_diagnostics.write_findings"):
                summary = diagnose_run(run, capture, root / "report")
            self.assertTrue(summary["reproduced_saved_run"])
            self.assertEqual(summary["frames_with_cv"], 0)
            self.assertEqual(summary["gyro_intervals"], 0)
            self.assertTrue((root / "report/index.html").exists())


class DiagnoseMapIntegrationTests(unittest.TestCase):
    """Run VO on a synthetic scene, then `fusionlab diagnose-map` on the saved run."""

    def setUp(self):
        require("cv2", "numpy", "scipy")

    def run_cli(self, *arguments: str) -> dict:
        from fusionlab.cli import main

        with patch("sys.argv", ["fusionlab", "diagnose-map", *arguments]):
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                main()
        return json.loads(stdout.getvalue())

    def test_reports_map_and_parallax_for_a_saved_run(self):
        from fusionlab.vision.odometry.runner import run_odometry

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tracks = root / "tracks.jsonl"
            rows = translating_camera_rows(8)
            tracks.write_text("".join(json.dumps(r) + "\n" for r in rows))
            calibration = root / "calibration.json"
            calibration.write_text(json.dumps(calibration_json()))
            run_odometry(tracks, calibration, root / "vo", stride=1)

            printed = self.run_cli(str(root / "vo"), "--output-dir", str(root / "map"))

            summary = json.loads((root / "map/summary.json").read_text())
            self.assertTrue((root / "map/report.md").read_text().startswith("# Map"))
        self.assertEqual(printed["status_counts"], summary["status_counts"])
        self.assertEqual(summary["loss_context"]["losses"], 0)
        # Every track is visible from bootstrap on, so all 200 are counted once.
        self.assertEqual(summary["available_parallax"]["tracks"], 200)
        # The camera moves 0.25 units per frame past points 4-8 units away, so
        # most tracks pass 2 degrees against their first view within the run.
        self.assertGreater(
            summary["available_parallax"]["reach"]["2"]["track_share"], 0.5
        )
        self.assertEqual(summary["configuration"]["keyframe_policy"], "baseline")
        self.assertIsNotNone(summary["reference_age"])

    def test_refuses_an_existing_output_directory(self):
        from fusionlab.cli import main
        from fusionlab.vision.odometry.runner import run_odometry

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tracks = root / "tracks.jsonl"
            rows = translating_camera_rows(3)
            tracks.write_text("".join(json.dumps(r) + "\n" for r in rows))
            calibration = root / "calibration.json"
            calibration.write_text(json.dumps(calibration_json()))
            run_odometry(tracks, calibration, root / "vo", stride=1)
            (root / "map").mkdir()
            argv = [
                "fusionlab",
                "diagnose-map",
                str(root / "vo"),
                "--output-dir",
                str(root / "map"),
            ]
            with patch("sys.argv", argv):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as exit_info:
                        main()
        self.assertEqual(exit_info.exception.code, 1)
