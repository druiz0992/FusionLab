import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "fixed_calibration",
    Path(__file__).resolve().parents[1] / "benchmarks/validate_calibration.py",
)
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class FixedCalibrationTests(unittest.TestCase):
    def test_correct_model_predicts_corners_and_wrong_intrinsics_degrade(self):
        try:
            import cv2 as cv
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV unavailable")
        calibration = dict(
            inner_corners=[9, 6],
            measured_square_mm=20,
            camera_matrix=[[900, 0, 640], [0, 920, 360], [0, 0, 1]],
            distortion_coefficients=[0.15, -0.1, 0, 0, 0.02],
        )
        obj = np.zeros((54, 3), np.float32)
        obj[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 0.02
        corners = cv.projectPoints(
            obj,
            np.array([0.4, 0.3, 0.1]),
            np.array([0.1, 0.03, 0.5]),
            np.array(calibration["camera_matrix"], float),
            np.array(calibration["distortion_coefficients"]),
        )[0]
        result = validator.score_corners(corners, calibration)
        self.assertLess(result["heldout_corner_rms_px"], 0.001)
        wrong = dict(
            calibration, camera_matrix=[[650, 0, 640], [0, 1100, 360], [0, 0, 1]]
        )
        self.assertGreater(
            validator.score_corners(corners, wrong)["heldout_corner_rms_px"], 1
        )
        self.assertEqual(calibration["camera_matrix"][0][0], 900)
