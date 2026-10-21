"""sensor-stream argument validation."""

import argparse
import unittest

from sensor_stream.cli import _positive_seconds


class DurationTests(unittest.TestCase):
    def test_duration_validation(self):
        self.assertEqual(_positive_seconds("0.25"), 0.25)
        for value in ("0", "-1", "nan", "inf", "-inf", "hello"):
            with (
                self.subTest(value=value),
                self.assertRaises(argparse.ArgumentTypeError),
            ):
                _positive_seconds(value)
