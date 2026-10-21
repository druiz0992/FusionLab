"""Checkerboard target and intrinsic fitting."""

import tempfile
import unittest
from pathlib import Path

from support import require


class CalibrationTests(unittest.TestCase):
    @classmethod
    @classmethod
    def setUpClass(cls):
        cls.cv, cls.np = require("cv2", "numpy")

    def test_recovers_known_intrinsics_from_varied_synthetic_poses(self):
        from fusionlab.vision.calibration import fit_points

        cv, np = self.cv, self.np
        obj = np.zeros((54, 3), np.float32)
        obj[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 0.025
        K = np.array([[800.0, 0.0, 640.0], [0.0, 820.0, 360.0], [0.0, 0.0, 1.0]])
        points = []
        for i in range(20):
            r = np.array([-0.4 + 0.2 * (i % 5), -0.3 + 0.2 * (i // 5), 0.08 * (i % 3)])
            t = np.array(
                [-0.15 + 0.04 * (i % 5), -0.1 + 0.03 * (i // 5), 0.65 + 0.08 * (i % 4)]
            )
            p, _ = cv.projectPoints(obj, r, t, K, np.zeros(5))
            points.append(p)
        result = fit_points(points, (1280, 720))
        recovered = np.array(result["camera_matrix"])
        self.assertAlmostEqual(recovered[0, 0], 800, delta=1)
        self.assertAlmostEqual(recovered[1, 1], 820, delta=1)
        self.assertLess(result["heldout_rms_px"], 0.01)
        self.assertEqual(len(result["heldout_view_indices"]), 4)
        # OpenCV versions differ in detector output shape: (N, 2) or (N, 1, 2).
        flat_result = fit_points([p.reshape(-1, 2) for p in points], (1280, 720))
        self.assertLess(flat_result["heldout_rms_px"], 0.01)
        np.testing.assert_allclose(
            flat_result["per_view_rms_px"], result["per_view_rms_px"], atol=1e-4
        )

    def test_requires_enough_views(self):
        from fusionlab.vision.calibration import fit_points

        with self.assertRaises(ValueError):
            fit_points([], (1280, 720))

    def test_printable_target_physical_dimensions(self):
        from fusionlab.vision.calibration import target

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            target(path)
            svg = (path / "checkerboard.svg").read_text()
            self.assertIn('width="297mm" height="210mm"', svg)
            self.assertEqual(svg.count('width="25" height="25"'), 35)
            self.assertTrue(
                (path / "checkerboard.pdf").read_bytes().startswith(b"%PDF")
            )
