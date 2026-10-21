"""Map-maintenance diagnostics on hand-computable runs."""

import json
import math
import tempfile
import unittest
from pathlib import Path

from fusionlab.evaluation.map_maintenance import (
    available_parallax,
    geometric_losses,
    loss_context,
    reference_ages,
    reference_tracker_overlap,
    segment_spans,
    triangulation_totals,
)
from fusionlab.vision.odometry.frame_result import FrameResult, TrackingStatus

from support import require
from support.vision import calibration_json, track_row

IDENTITY_POSE = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]


def result(index, status="tracking", segment=1, *, pose=None, **diagnostics):
    """A FrameResult with the map counts the diagnostics read."""
    defaults = {
        "visible_features": 400,
        "visible_landmarks": 100,
        "landmarks_before": 120,
        "landmarks_added": 0,
    }
    tracking = status == "tracking"
    failure_category = diagnostics.pop("failure_category", None)
    return FrameResult(
        source_time_ns=index * 33_000_000,
        received_monotonic_ns=index * 33_000_000,
        segment=segment,
        status=TrackingStatus(status),
        diagnostics=defaults | diagnostics,
        failure_category=failure_category,
        world_to_camera=(pose or IDENTITY_POSE) if tracking else None,
        frame_index=index,
    )


class LossContextTests(unittest.TestCase):
    def test_counts_tracks_and_mapped_landmarks_before_and_at_loss(self):
        declining = [result(i, visible_landmarks=100 - 20 * i) for i in range(5)]
        loss = result(
            5,
            "tracking_lost",
            visible_landmarks=8,
            failure_category="too_few_visible_landmarks",
        )
        context = loss_context(declining + [loss])
        self.assertEqual(context["losses"], 1)
        self.assertEqual(
            context["failure_categories"], {"too_few_visible_landmarks": 1}
        )
        self.assertEqual(context["at_loss"]["median_visible_tracks"], 400)
        self.assertEqual(context["at_loss"]["median_mapped_fraction"], 8 / 400)
        # Frames 4 and 2 precede the loss by 1 and 3 frames: 20 and 60 mapped.
        before = context["visible_mapped_before_loss"]
        self.assertEqual(before["1"]["median_visible_mapped"], 20)
        self.assertEqual(before["3"]["median_visible_mapped"], 60)
        self.assertEqual(before["10"]["losses"], 0)

    def test_lookback_ignores_frames_from_an_earlier_segment(self):
        rows = [result(0, segment=1), result(1, segment=2)]
        rows.append(result(2, "tracking_lost", segment=2))
        before = loss_context(rows)["visible_mapped_before_loss"]
        self.assertEqual(before["1"]["losses"], 1)
        self.assertEqual(before["3"]["losses"], 0)

    def test_segment_spans_flag_single_frame_segments(self):
        rows = [result(0, segment=1), result(1, segment=1), result(2, segment=2)]
        spans = segment_spans(rows)
        self.assertEqual(spans["tracked_frames_per_segment"], [1, 2])
        self.assertEqual(spans["segments_with_one_tracked_frame"], 1)


class TriangulationAndReferenceTests(unittest.TestCase):
    def test_triangulation_shares_and_admitted_angles(self):
        rows = [
            result(
                0,
                landmarks_added=2,
                triangulation={
                    "candidates": 10,
                    "accepted": 2,
                    "low_parallax": 8,
                    "accepted_parallax_deg": [1.5, 3.0],
                },
            ),
            result(1, triangulation={"candidates": 10, "low_parallax": 10}),
        ]
        totals = triangulation_totals(rows)
        self.assertEqual(totals["low_parallax_share"], 0.9)
        self.assertEqual(totals["accepted_share"], 0.1)
        self.assertEqual(totals["admitted_below_2_deg_share"], 0.5)
        self.assertEqual(totals["mean_landmarks_added_per_tracking_frame"], 1.0)

    def test_reference_age_resets_when_landmarks_are_added(self):
        added = [0, 0, 5, 0, 0, 0]
        rows = [result(i, landmarks_added=a) for i, a in enumerate(added)]
        summary = {"keyframe_policy": "baseline", "landmark_policy": "refinement"}
        self.assertEqual(reference_ages(rows, summary), [0, 1, 2, 1, 2, 3])

    def test_reference_age_restarts_after_a_loss(self):
        rows = [result(0), result(1), result(2, "tracking_lost"), result(3)]
        self.assertEqual(reference_ages(rows, {}), [0, 1, None, 0])

    def test_reference_age_is_not_inferred_for_other_policies(self):
        rows = [result(0)]
        self.assertIsNone(reference_ages(rows, {"keyframe_policy": "parallax"}))
        self.assertIsNone(reference_ages(rows, {"landmark_policy": "validation"}))


class AvailableParallaxTests(unittest.TestCase):
    def setUp(self):
        require("cv2", "numpy")

    def test_known_parallax_for_a_point_seen_from_a_sideways_step(self):
        # One point 5 units ahead; the camera then moves 0.25 units along +X.
        # Its triangulation angle is atan(0.25 / 5) = 2.862 degrees.
        focal = 500.0
        rows = [
            track_row(0, [{"id": 7, "xy": [320.0, 240.0]}]),
            track_row(33_000_000, [{"id": 7, "xy": [320.0 - focal * 0.25 / 5, 240.0]}]),
        ]
        moved = [[1.0, 0.0, 0.0, -0.25], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        results = [
            result(0, visible_landmarks=1),
            result(1, pose=moved, visible_landmarks=1),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tracks = Path(tmp) / "tracks.jsonl"
            tracks.write_text("".join(json.dumps(r) + "\n" for r in rows))
            parallax = available_parallax(results, tracks, calibration_json())
        expected = math.degrees(math.atan(0.25 / 5))
        self.assertAlmostEqual(expected, 2.862, places=3)
        self.assertEqual(parallax["tracks"], 1)
        self.assertEqual(parallax["reach"]["2"]["track_share"], 1.0)
        self.assertEqual(parallax["reach"]["2"]["median_processed_frames_needed"], 1)
        self.assertEqual(parallax["reach"]["3"]["track_share"], 0.0)
        self.assertEqual(
            parallax["per_tracking_frame"]["median_visible_with_useful_parallax"], 0.5
        )

    def test_tracks_restart_their_first_observation_in_a_new_segment(self):
        rows = [
            track_row(0, [{"id": 1, "xy": [320.0, 240.0]}]),
            track_row(33_000_000, [{"id": 1, "xy": [295.0, 240.0]}]),
        ]
        results = [result(0, segment=1), result(1, segment=2)]
        with tempfile.TemporaryDirectory() as tmp:
            tracks = Path(tmp) / "tracks.jsonl"
            tracks.write_text("".join(json.dumps(r) + "\n" for r in rows))
            parallax = available_parallax(results, tracks, calibration_json())
        # Same track ID, two segments: two independent first observations.
        self.assertEqual(parallax["tracks"], 2)
        self.assertEqual(parallax["reach"]["1"]["track_share"], 0.0)


class LossDetailTests(unittest.TestCase):
    def test_geometric_losses_report_recent_and_older_errors(self):
        rows = [
            result(0),
            result(
                1,
                "tracking_lost",
                failure_category="low_pnp_inlier_ratio",
                pnp_candidates=200,
                pnp_recent_landmarks=50,
                pnp_recent_median_error_px=1.5,
                pnp_older_landmarks=150,
                pnp_older_median_error_px=6.0,
            ),
            result(2, "tracking_lost", failure_category="too_few_visible_landmarks"),
        ]
        losses = geometric_losses(rows)
        self.assertEqual([loss["frame_index"] for loss in losses], [1])
        self.assertEqual(losses[0]["older_median_error_px"], 6.0)

    def test_reference_tracker_state_at_each_loss(self):
        rows = [result(0), result(1, "tracking_lost"), result(3, "tracking_lost")]
        with tempfile.TemporaryDirectory() as tmp:
            states = Path(tmp) / "states.csv"
            # ORB codes: 2 OK, 3 RECENTLY_LOST.
            states.write_text(
                "source_time_ns,state\n0,2\n33000000,2\n66000000,3\n99000000,3\n"
            )
            overlap = reference_tracker_overlap(rows, states)
        self.assertEqual(overlap["losses_while_reference_tracking"], 1)
        self.assertEqual(
            [loss["reference_tracking"] for loss in overlap["losses"]], [True, False]
        )


if __name__ == "__main__":
    unittest.main()
