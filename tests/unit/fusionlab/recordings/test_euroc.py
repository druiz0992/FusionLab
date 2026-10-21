"""EuRoC import safety checks."""

import tempfile
import unittest
from pathlib import Path


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

    def test_import_rejects_traversal_before_creating_output(self):
        from fusionlab.recordings.euroc import import_sequence

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, epoch = self.fixture(root)
            (source / "cam0/data.csv").write_text(f"{epoch},../../sensor.yaml\n")
            with self.assertRaises(ValueError):
                import_sequence(source, root / "output")
            self.assertFalse((root / "output").exists())
