"""Exploratory position integration."""

import math
import unittest
from dataclasses import replace

from fusionlab.inertial.dead_reckoning import integrate_position
from fusionlab.inertial.measurements import MeasurementConvention

from support.imu import sample


def measurement(t, acceleration=(1.0, 0.0, 0.0), orientation=(0.0, 0.0, 0.0, 1.0)):
    return replace(
        sample(t, gyro=(0.0, 0.0, 0.0), orientation=orientation),
        acceleration=acceleration,
    )


class PositionTests(unittest.TestCase):
    def test_stationary_clamps_velocity_and_holds_position(self):
        rows = [measurement(i / 100) for i in range(101)]
        result = integrate_position(
            rows, MeasurementConvention("m/s2"), stationary_intervals_s=((0.5, 1.01),)
        )
        stop = result.estimates[50]
        self.assertAlmostEqual(stop.pre_update_velocity_m_s[0], 0.5)
        self.assertEqual(stop.velocity_m_s, (0.0, 0.0, 0.0))
        self.assertEqual(result.estimates[-1].position_m, stop.position_m)
        self.assertAlmostEqual(result.estimates[49].velocity_m_s[0], 0.49)

    def test_rejects_invalid_stationary_intervals(self):
        for intervals in (((0.5, 0.4),), ((0, 0.5), (0.4, 0.8)), ((2, 3),)):
            with self.subTest(intervals=intervals), self.assertRaises(ValueError):
                integrate_position(
                    [measurement(i / 100) for i in range(101)],
                    MeasurementConvention("m/s2"),
                    stationary_intervals_s=intervals,
                )

    def test_constant_acceleration(self):
        result = integrate_position(
            [measurement(i / 100) for i in range(101)], MeasurementConvention("m/s2")
        )
        self.assertAlmostEqual(result.estimates[-1].velocity_m_s[0], 1)
        self.assertAlmostEqual(result.estimates[-1].position_m[0], 0.5)

    def test_rotated_axis(self):
        q = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
        result = integrate_position(
            [measurement(0, orientation=q), measurement(0.02, orientation=q)],
            MeasurementConvention("m/s2"),
        )
        self.assertAlmostEqual(result.estimates[-1].velocity_m_s[0], 0)
        self.assertAlmostEqual(result.estimates[-1].velocity_m_s[1], 0.02)

    def test_bias_and_gap_reset(self):
        rows = [measurement(0), measurement(0.02), measurement(1)]
        result = integrate_position(
            rows, MeasurementConvention("m/s2"), bias_interval_s=(0, 0.03)
        )
        self.assertEqual(result.bias_body_m_s2, (1.0, 0.0, 0.0))
        self.assertEqual(result.estimates[-1].segment, 2)
        self.assertTrue(all(e.position_m == (0.0, 0.0, 0.0) for e in result.estimates))

    def test_missing_orientation_and_invalid_interval(self):
        with self.assertRaises(ValueError):
            integrate_position(
                [measurement(0, orientation=None)], MeasurementConvention("g")
            )
        with self.assertRaises(ValueError):
            integrate_position(
                [measurement(0)], MeasurementConvention("g"), bias_interval_s=(1, 2)
            )
