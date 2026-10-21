"""Typed tracks.jsonl records."""

import json
import tempfile
import unittest
from pathlib import Path

from fusionlab.vision.tracks import (
    Track,
    TrackFrame,
    read_track_frame_pair,
    read_track_frames,
)

# A row exactly as feature tracking writes it, in its historical key order.
FULL_ROW = {
    "segment": 2,
    "reset_reason": None,
    "previous_features": 3,
    "tracked": 2,
    "rejected_flow": 1,
    "rejected_geometry": 0,
    "geometry_status": "ok",
    "added": 1,
    "tracks": [
        {"id": 7, "xy": [298.0, 43.5], "age_frames": 4},
        {"id": 9, "xy": [10.25, 300.0], "age_frames": 1},
    ],
    "source_time_ns": 1_000_000_000,
    "received_monotonic_ns": 5_000_000_000,
    "stream_id": "rear",
    "original_size": [1920, 1080],
    "tracking_size": [640, 360],
    "pixel_scale_xy": [0.3333333333333333, 0.3333333333333333],
    "display_rotation_clockwise_deg": 90,
    "overlay_path": "overlays/000000.jpg",
}


# The fields a hand-written or minimal row must provide.
MINIMAL_ROW = {
    "source_time_ns": 0,
    "received_monotonic_ns": 0,
    "stream_id": "s",
    "segment": 1,
    "original_size": [640, 480],
    "tracking_size": [640, 480],
    "pixel_scale_xy": [1.0, 1.0],
    "tracks": [{"id": 0, "xy": [1, 2]}],
}


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class TrackFrameTests(unittest.TestCase):
    def test_full_row_round_trips_with_identical_json(self):
        frame = TrackFrame.from_json(FULL_ROW)
        self.assertEqual(json.dumps(frame.to_json()), json.dumps(FULL_ROW))

    def test_parses_types_and_geometry(self):
        frame = TrackFrame.from_json(FULL_ROW)
        self.assertEqual(frame.tracking_size, (640, 360))
        self.assertEqual(frame.geometry_key, ("rear", 2, (640, 360)))
        self.assertEqual(frame.tracks[0], Track(id=7, xy=(298.0, 43.5), age_frames=4))
        self.assertEqual(frame.positions(), {7: (298.0, 43.5), 9: (10.25, 300.0)})

    def test_minimal_row_uses_diagnostic_defaults(self):
        frame = TrackFrame.from_json(MINIMAL_ROW)
        self.assertEqual(frame.tracked, 0)
        self.assertIsNone(frame.overlay_path)
        self.assertIsNone(frame.tracks[0].age_frames)
        self.assertIsInstance(frame.tracks[0].xy[0], float)

    def test_older_rows_without_display_rotation_round_trip(self):
        older = {
            k: v for k, v in FULL_ROW.items() if k != "display_rotation_clockwise_deg"
        }
        frame = TrackFrame.from_json(older)
        self.assertIsNone(frame.display_rotation_clockwise_deg)
        self.assertEqual(json.dumps(frame.to_json()), json.dumps(older))

    def test_missing_required_field_names_it(self):
        row = {k: v for k, v in MINIMAL_ROW.items() if k != "pixel_scale_xy"}
        with self.assertRaisesRegex(ValueError, "pixel_scale_xy"):
            TrackFrame.from_json(row)


class TrackFileReadingTests(unittest.TestCase):
    def test_read_track_frames_yields_row_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tracks.jsonl"
            write_rows(path, [MINIMAL_ROW, FULL_ROW])
            frames = list(read_track_frames(path))
        self.assertEqual([index for index, _ in frames], [0, 1])
        self.assertEqual(frames[1][1].stream_id, "rear")

    def test_read_pair_selects_requested_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tracks.jsonl"
            rows = [dict(MINIMAL_ROW, source_time_ns=i) for i in range(5)]
            write_rows(path, rows)
            first, second = read_track_frame_pair(path, 1, 3)
        self.assertEqual((first.source_time_ns, second.source_time_ns), (1, 3))

    def test_read_pair_rejects_bad_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tracks.jsonl"
            write_rows(path, [MINIMAL_ROW] * 3)
            with self.assertRaisesRegex(ValueError, "0 <= frame-a < frame-b"):
                read_track_frame_pair(path, 2, 2)
            with self.assertRaisesRegex(ValueError, "exceeds"):
                read_track_frame_pair(path, 0, 10)
