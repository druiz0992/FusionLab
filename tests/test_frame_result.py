"""Tests for typed visual odometry results (`poses.jsonl` rows)."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from fusionlab.vision.odometry.frame_result import (
    FrameResult,
    TrackingStatus,
    read_frame_results,
)

# Rows in the order the estimator produces keys: core, diagnostics, then
# reason/failure or pose, then the runner's frame index.
TRACKING_ROW = {
    "source_time_ns": 100,
    "received_monotonic_ns": 200,
    "segment": 3,
    "status": "tracking",
    "landmarks": 40,
    "visible_features": 55,
    "pnp_inliers": 31,
    "triangulation": {"accepted": 4, "accepted_parallax_deg": [1.5, 2.0]},
    "camera_center_arbitrary": [0.1, 0.0, 0.9],
    "world_to_camera": [[1, 0, 0, -0.1], [0, 1, 0, 0], [0, 0, 1, -0.9]],
    "frame_index": 12,
}
LOST_ROW = {
    "source_time_ns": 300,
    "received_monotonic_ns": 400,
    "segment": 3,
    "status": "tracking_lost",
    "landmarks": 40,
    "pnp_attempted": False,
    "reason": "insufficient_pnp_support",
    "failure_category": "too_few_visible_landmarks",
    "frame_index": 13,
}


class FrameResultTests(unittest.TestCase):
    def test_rows_round_trip_with_identical_json(self):
        for row in (TRACKING_ROW, LOST_ROW):
            with self.subTest(status=row["status"]):
                result = FrameResult.from_json(row)
                self.assertEqual(json.dumps(result.to_json()), json.dumps(row))

    def test_core_fields_are_typed_and_rest_are_diagnostics(self):
        result = FrameResult.from_json(TRACKING_ROW)
        self.assertIs(result.status, TrackingStatus.TRACKING)
        self.assertTrue(result.is_tracking)
        self.assertEqual(result.frame_index, 12)
        self.assertEqual(len(result.world_to_camera), 3)
        self.assertEqual(
            list(result.diagnostics),
            ["landmarks", "visible_features", "pnp_inliers", "triangulation"],
        )

    def test_loss_carries_reason_and_failure_category(self):
        result = FrameResult.from_json(LOST_ROW)
        self.assertFalse(result.is_tracking)
        self.assertEqual(result.failure_category, "too_few_visible_landmarks")
        self.assertIsNone(result.world_to_camera)

    def test_status_compares_equal_to_saved_strings(self):
        # Existing reports and saved runs compare against plain strings.
        self.assertEqual(TrackingStatus.TRACKING_LOST, "tracking_lost")
        self.assertEqual(
            json.dumps({"s": TrackingStatus.TRACKING}), '{"s": "tracking"}'
        )

    def test_unknown_status_is_rejected(self):
        with self.assertRaises(ValueError):
            FrameResult.from_json(dict(LOST_ROW, status="lost"))

    def test_frame_index_is_added_without_mutation(self):
        result = FrameResult.from_json({**LOST_ROW, "frame_index": None})
        indexed = replace(result, frame_index=7)
        self.assertIsNone(result.frame_index)
        self.assertEqual(indexed.to_json()["frame_index"], 7)

    def test_read_frame_results_skips_blank_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "poses.jsonl"
            path.write_text(
                json.dumps(TRACKING_ROW) + "\n\n" + json.dumps(LOST_ROW) + "\n"
            )
            results = read_frame_results(path)
        self.assertEqual([r.frame_index for r in results], [12, 13])


if __name__ == "__main__":
    unittest.main()
