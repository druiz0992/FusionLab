"""Synthetic IMU samples with explicit units: seconds in, nanoseconds stored."""

import math

from sensor_stream import ImuSample


def sample(t, gyro=(0.0, 0.0, math.pi / 2), orientation=None, session="s"):
    """An ImuSample at `t` seconds with zero acceleration and gyro in rad/s."""
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
