"""Tests for scoring fixed intrinsics on a separate checkerboard capture."""

import contextlib
import io
import json
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

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


class CalibrationValidationUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cv2
            import numpy as np
        except ImportError:
            raise unittest.SkipTest("OpenCV unavailable")
        cls.cv, cls.np = cv2, np

    def calibration(self, **overrides):
        from fusionlab.vision.calibration_validation import FixedCalibration

        return FixedCalibration.from_json({**CALIBRATION_JSON, **overrides})

    def project_board(self, distortion):
        """Project the board with a known pose: exact corners for the true model."""
        from fusionlab.vision.calibration_validation import board_points_m

        cv, np = self.cv, self.np
        return cv.projectPoints(
            board_points_m(self.calibration()),
            np.array([0.4, 0.3, 0.1]),
            np.array([0.1, 0.03, 0.5]),
            np.array(CAMERA_MATRIX),
            np.array(distortion, float),
        )[0]

    def test_correct_model_predicts_corners_and_wrong_intrinsics_degrade(self):
        from fusionlab.vision.calibration_validation import score_corners

        distortion = [0.15, -0.1, 0, 0, 0.02]
        corners = self.project_board(distortion)
        correct = self.calibration(distortion_coefficients=distortion)
        self.assertLess(score_corners(corners, correct)["heldout_corner_rms_px"], 1e-3)
        wrong = self.calibration(
            distortion_coefficients=distortion,
            camera_matrix=[[650, 0, 640], [0, 1100, 360], [0, 0, 1]],
        )
        self.assertGreater(score_corners(corners, wrong)["heldout_corner_rms_px"], 1)

    def test_accepts_flat_and_nested_corner_shapes(self):
        from fusionlab.vision.calibration_validation import score_corners

        corners = self.project_board([0, 0, 0, 0, 0])
        nested = score_corners(corners, self.calibration())
        flat = score_corners(corners.reshape(-1, 2), self.calibration())
        self.assertAlmostEqual(nested["all_corner_rms_px"], flat["all_corner_rms_px"])

    def test_rejects_wrong_corner_count(self):
        from fusionlab.vision.calibration_validation import score_corners

        corners = self.project_board([0, 0, 0, 0, 0])[:-1]
        with self.assertRaisesRegex(ValueError, "corner count"):
            score_corners(corners, self.calibration())

    def test_rejects_unsupported_distortion_model(self):
        with self.assertRaisesRegex(ValueError, "Unsupported model"):
            self.calibration(distortion_order=["k1", "k2", "k3", "k4"])

    def test_board_points_use_measured_square_in_metres(self):
        from fusionlab.vision.calibration_validation import board_points_m

        points = board_points_m(self.calibration(measured_square_mm=25))
        self.assertEqual(points.shape, (54, 3))
        # Second corner is one square along +X; last is at (8, 5) squares.
        self.np.testing.assert_allclose(points[1], [0.025, 0, 0])
        self.np.testing.assert_allclose(points[-1], [0.2, 0.125, 0])


class CaptureReadingTests(unittest.TestCase):
    def test_select_images_uses_stride_then_limit(self):
        from fusionlab.vision.calibration_validation import select_images

        events = {f"camera/{i}.jpg": {} for i in range(10)}
        self.assertEqual(
            select_images(events, None, stride=3, max_views=2),
            ["camera/0.jpg", "camera/3.jpg"],
        )

    def test_select_images_prefers_explicit_detection_list(self):
        from fusionlab.vision.calibration_validation import select_images

        with tempfile.TemporaryDirectory() as tmp:
            listing = Path(tmp) / "detections.json"
            listing.write_text(json.dumps({"accepted_images": ["camera/7.jpg"]}))
            self.assertEqual(
                select_images({}, listing, stride=1, max_views=5), ["camera/7.jpg"]
            )

    def test_requires_single_camera_stream(self):
        from fusionlab.vision.calibration_validation import read_camera_events

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                camera_event("a.jpg", b"a", "rgb"),
                camera_event("b.jpg", b"b", "ir"),
            ]
            (root / "events.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rows)
            )
            with self.assertRaisesRegex(ValueError, "one camera stream"):
                read_camera_events(root)

    def test_rejects_modified_image(self):
        from fusionlab.vision.calibration_validation import (
            read_camera_events,
            read_verified_image,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_capture(root, {"camera/0.jpg": b"original"})
            events = read_camera_events(root)
            (root / "camera/0.jpg").write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "integrity"):
                read_verified_image(root, "camera/0.jpg", events)

    def test_rejects_paths_outside_capture(self):
        from fusionlab.vision.calibration_validation import read_verified_image

        with tempfile.TemporaryDirectory() as tmp:
            events = {"../outside.jpg": camera_event("../outside.jpg", b"x")}
            with self.assertRaisesRegex(ValueError, "escapes capture"):
                read_verified_image(Path(tmp), "../outside.jpg", events)


class ValidateCalibrationIntegrationTests(unittest.TestCase):
    """Render a board with known intrinsics, then run the CLI end to end."""

    @classmethod
    def setUpClass(cls):
        try:
            import cv2
            import numpy as np
        except ImportError:
            raise unittest.SkipTest("OpenCV unavailable")
        cls.cv, cls.np = cv2, np

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


if __name__ == "__main__":
    unittest.main()
