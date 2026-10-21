import unittest
from dataclasses import replace

from fusionlab.inertial.measurements import MeasurementConvention, convert_imu
from test_orientation import sample


class MeasurementTests(unittest.TestCase):
    def test_explicit_g_conversion_preserves_raw_sample(self):
        raw = replace(sample(0), acceleration=(1.0, -1.0, 0.0))
        processed = convert_imu(raw, MeasurementConvention("g"))
        self.assertEqual(processed.acceleration_m_s2, (9.80665, -9.80665, 0.0))
        self.assertIs(processed.raw, raw)
        self.assertEqual(raw.acceleration, (1.0, -1.0, 0.0))
        self.assertEqual(processed.angular_velocity_rad_s, raw.angular_velocity)

    def test_si_input_and_invalid_values(self):
        raw = replace(sample(0), acceleration=(1.0, 2.0, 3.0))
        self.assertEqual(
            convert_imu(raw, MeasurementConvention("m/s2")).acceleration_m_s2,
            raw.acceleration,
        )
        with self.assertRaises(ValueError):
            convert_imu(
                replace(raw, acceleration=(float("nan"), 0.0, 0.0)),
                MeasurementConvention("g"),
            )
