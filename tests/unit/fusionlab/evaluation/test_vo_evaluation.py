"""Saved-run jump inspection."""

import json
import tempfile
import unittest
from pathlib import Path

from fusionlab.evaluation.vo_evaluation import inspect_translation_jumps


class JumpInspectionTests(unittest.TestCase):
    def test_flags_are_scale_invariant_and_do_not_bridge_segments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = []
            for scale in (1.0, 100.0):
                output = root / str(scale)
                output.mkdir()
                rows = []
                for i, x in enumerate((0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 15.0)):
                    rows.append(
                        {
                            "status": "tracking",
                            "segment": 1,
                            "frame_index": i,
                            "received_monotonic_ns": i * 100_000_000,
                            "camera_center_arbitrary": [x * scale, 0.0, 0.0],
                        }
                    )
                rows.append(
                    {
                        "status": "tracking",
                        "segment": 2,
                        "frame_index": 7,
                        "received_monotonic_ns": 700_000_000,
                        "camera_center_arbitrary": [100000.0, 0.0, 0.0],
                    }
                )
                (output / "poses.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows)
                )
                reports.append(inspect_translation_jumps(output, limit=1)[0])
            self.assertEqual(reports[0]["frame_index"], 6)
            self.assertEqual(reports[1]["frame_index"], 6)
            self.assertEqual(reports[0]["step_over_segment_median"], 10.0)
            self.assertEqual(reports[1]["step_over_segment_median"], 10.0)
