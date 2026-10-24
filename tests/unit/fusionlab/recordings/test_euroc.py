"""EuRoC import: camera model, frames, ground truth and safety checks."""

import json
import tempfile
import unittest
from pathlib import Path

from support import require
from support.euroc import CAMERA_LEVER_ARM_X_M, FRAMES, write_euroc_sequence


class EurocImportTests(unittest.TestCase):
    def setUp(self):
        self.np, _, self.yaml, _ = require("numpy", "cv2", "yaml", "scipy")

    def sensor(self, **changes):
        sensor = dict(
            camera_model="pinhole",
            distortion_model="radial-tangential",
            resolution=[96, 64],
            intrinsics=[80, 81, 48, 32],
            distortion_coefficients=[0.1, -0.01, 0.001, 0.002],
            T_BS=dict(data=self.np.eye(4).ravel().tolist()),
        )
        sensor.update(changes)
        return sensor

    def test_import_rejects_traversal_before_creating_output(self):
        from fusionlab.recordings.euroc import import_sequence

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, epoch = write_euroc_sequence(root)
            (source / "cam0/data.csv").write_text(f"{epoch},../../sensor.yaml\n")
            with self.assertRaises(ValueError):
                import_sequence(source, root / "output")
            self.assertFalse((root / "output").exists())

    def test_import_writes_calibration_groundtruth_and_metadata(self):
        from fusionlab.recordings.euroc import import_sequence

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mav, epoch = write_euroc_sequence(root)
            # The sequence root (parent of mav0) is accepted as well as mav0.
            metadata = import_sequence(mav.parent, root / "output")
            calibration = json.loads((root / "output/calibration.json").read_text())
            truth = json.loads((root / "output/groundtruth.json").read_text())
        self.assertEqual(metadata["frames"], FRAMES)
        self.assertEqual(metadata["groundtruth_states"], FRAMES)
        self.assertEqual(calibration["distortion_coefficients"], [0.0] * 5)
        self.assertEqual(calibration["camera_matrix"][0], [80.0, 0, 48.0])
        self.assertEqual(truth["body_from_camera"][0][3], CAMERA_LEVER_ARM_X_M)
        self.assertEqual(truth["states"][3]["source_time_ns"], epoch + 30_000_000)
        self.assertAlmostEqual(truth["states"][3]["world_from_body"][1][3], 0.09)

    def test_camera_model_appends_zero_k3_and_rejects_other_models(self):
        from fusionlab.recordings.euroc import read_camera_model

        camera = read_camera_model(self.sensor())
        calibration = camera.calibration_json(Path("sensor.yaml"))
        self.assertEqual(
            calibration["distortion_coefficients"], [0.1, -0.01, 0.001, 0.002, 0.0]
        )
        self.assertEqual(calibration["camera_matrix"][1], [0, 81.0, 32.0])
        self.assertEqual(calibration["image_size"], [96, 64])
        with self.assertRaisesRegex(ValueError, "pinhole"):
            read_camera_model(self.sensor(distortion_model="equidistant"))
        with self.assertRaisesRegex(ValueError, "calibration"):
            read_camera_model(self.sensor(intrinsics=[0, 80, 48, 32]))

    def test_ground_truth_quaternion_order_and_validation(self):
        from fusionlab.recordings.euroc import read_ground_truth

        np = self.np
        half = np.sqrt(0.5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.csv"
            # 90° about +Z stored as w, x, y, z.
            path.write_text(f"1,1,2,3,{half},0,0,{half}\n2,1,2,3,1,0,0,0\n")
            states = read_ground_truth(path)
            pose = np.array(states[0]["world_from_body"])
            np.testing.assert_allclose(pose[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-12)
            np.testing.assert_allclose(pose[:3, 3], [1, 2, 3])
            path.write_text("1,0,0,0,2,0,0,0\n2,0,0,0,1,0,0,0\n")
            with self.assertRaisesRegex(ValueError, "quaternion"):
                read_ground_truth(path)
            path.write_text("2,0,0,0,1,0,0,0\n1,0,0,0,1,0,0,0\n")
            with self.assertRaisesRegex(ValueError, "timestamp"):
                read_ground_truth(path)

    def test_rigid_transform_rejects_reflections(self):
        from fusionlab.recordings.euroc import rigid_transform

        reflection = self.np.diag([1.0, 1.0, -1.0, 1.0])
        with self.assertRaisesRegex(ValueError, "rigid"):
            rigid_transform(reflection.ravel())
        self.assertEqual(rigid_transform(self.np.eye(4).ravel()).shape, (4, 4))
