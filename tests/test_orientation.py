import math
import unittest

from sensor_stream import ImuSample
from fusionlab.inertial.orientation import integrate, angle, IDENTITY, multiply


def sample(t, gyro=(0.0, 0.0, math.pi / 2), orientation=None, session="s"):
    return ImuSample(
        session + ":imu",
        session,
        int(t * 1e9),
        int(t * 1e9),
        int(t * 1e9),
        (0.0, 0.0, 0.0),
        gyro,
        orientation,
    )


class OrientationTests(unittest.TestCase):
    def test_known_rotation_and_reference_alignment(self):
        initial = (math.sin(0.3), 0.0, 0.0, math.cos(0.3))
        samples = []
        for i in range(101):
            t = i / 100
            delta = (0.0, 0.0, math.sin(t * math.pi / 4), math.cos(t * math.pi / 4))
            samples.append(sample(t, orientation=multiply(initial, delta)))
        estimates = integrate(samples)
        self.assertAlmostEqual(angle(IDENTITY, estimates[-1].orientation), 90, places=6)
        self.assertLess(max(e.error_deg for e in estimates), 1e-5)

    def test_gap_regression_and_reconnect_reset(self):
        estimates = integrate(
            [sample(0), sample(0.02), sample(1), sample(0.9), sample(0, session="new")]
        )
        self.assertEqual([e.segment for e in estimates], [1, 1, 2, 3, 4])
        self.assertEqual(estimates[-1].orientation, IDENTITY)

    def test_sign_equivalence_and_zero_gyro(self):
        self.assertEqual(angle(IDENTITY, (0.0, 0.0, 0.0, -1.0)), 0)
        estimates = integrate(
            [sample(0, gyro=(0.0, 0.0, 0.0)), sample(0.02, gyro=(0.0, 0.0, 0.0))]
        )
        self.assertEqual(estimates[-1].orientation, IDENTITY)
        self.assertIsNone(estimates[-1].error_deg)

    def test_inverse_reference(self):
        samples = [
            sample(0, orientation=IDENTITY),
            sample(
                0.02,
                orientation=(
                    0.0,
                    0.0,
                    -math.sin(math.pi * 0.005),
                    math.cos(math.pi * 0.005),
                ),
            ),
        ]
        self.assertLess(integrate(samples, inverse_reference=True)[-1].error_deg, 1e-5)


class OrientationViewerTests(unittest.TestCase):
    def test_comparison_export_alignment_and_segments(self):
        import tempfile
        from pathlib import Path
        from fusion_viewer import ViewerClient, OrientationSample, ScalarSample
        from fusion_viewer.recording import LogSink, read_log
        from fusionlab.visualization.visualization_adapter import export_orientation

        initial = (math.sin(0.3), 0.0, 0.0, math.cos(0.3))
        estimates = integrate(
            [
                sample(0, orientation=initial),
                sample(0.02, orientation=initial),
                sample(1, orientation=initial),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison"
            with ViewerClient(LogSink(path)) as viewer:
                self.assertEqual(export_orientation(estimates, viewer), 3)
            values = [v for _, v in read_log(path)]
        orientations = [v for v in values if isinstance(v, OrientationSample)]
        self.assertEqual(len(orientations), 6)
        self.assertLess(angle(orientations[0].quaternion_xyzw, IDENTITY), 1e-5)
        self.assertLess(angle(orientations[1].quaternion_xyzw, IDENTITY), 1e-5)
        self.assertEqual(orientations[-1].time.segment_id, "segment_2")
        self.assertEqual(orientations[-1].time.ticks_ns, 1_000_000_000)
        self.assertEqual(len([v for v in values if isinstance(v, ScalarSample)]), 3)
