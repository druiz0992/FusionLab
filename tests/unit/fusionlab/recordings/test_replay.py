"""Recorded IMU replay."""

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from fusionlab.recordings.replay import replay

from support.imu import sample


class ReplayTests(unittest.IsolatedAsyncioTestCase):
    async def test_order_timestamps_and_speed_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "metadata.json").write_text("{}")
            samples = [sample(0), sample(0.02)]
            (path / "events.jsonl").write_text(
                "".join(
                    json.dumps({"type": "ImuSample", **asdict(s)}) + "\n"
                    for s in samples
                )
            )
            self.assertEqual([s async for s in replay(path)], samples)
            self.assertEqual([s async for s in replay(path, speed=100)], samples)
            with self.assertRaises(ValueError):
                _ = [s async for s in replay(path, speed=-1)]
