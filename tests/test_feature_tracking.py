import unittest
from fusionlab.vision.feature_tracking import FeatureTracker, TrackingConfig


class FeatureTrackingTests(unittest.TestCase):
    def setUp(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("Optional tracking dependencies unavailable")
        self.cv, self.np = cv2, np

    def test_translation_ids_and_reset(self):
        np, cv = self.np, self.cv
        image = np.random.default_rng(4).integers(0, 256, (240, 320), dtype=np.uint8)
        image = cv.GaussianBlur(image, (5, 5), 0)
        shifted = cv.warpAffine(image, np.float32([[1, 0, 3], [0, 1, 2]]), (320, 240))
        tracker = FeatureTracker(TrackingConfig(max_features=100))
        first, _ = tracker.process(image, stream_id="s", source_time_ns=0)
        second, _ = tracker.process(shifted, stream_id="s", source_time_ns=33_000_000)
        old = {p["id"]: p["xy"] for p in first["tracks"]}
        shared = [p for p in second["tracks"] if p["id"] in old]
        self.assertGreater(len(shared), 30)
        motion = np.array(
            [[p["xy"][i] - old[p["id"]][i] for i in range(2)] for p in shared]
        )
        np.testing.assert_allclose(np.median(motion, axis=0), [3, 2], atol=0.3)
        reset, _ = tracker.process(image, stream_id="s", source_time_ns=1_000_000_000)
        self.assertEqual(reset["reset_reason"], "timestamp_gap")
        self.assertTrue(set(old).isdisjoint(p["id"] for p in reset["tracks"]))

    def test_blank_and_occlusion(self):
        np = self.np
        tracker = FeatureTracker()
        blank = np.zeros((120, 160), dtype=np.uint8)
        row, _ = tracker.process(blank, stream_id="s", source_time_ns=0)
        self.assertEqual(row["tracks"], [])
        image = np.random.default_rng(3).integers(0, 256, blank.shape, dtype=np.uint8)
        tracker.process(image, stream_id="s", source_time_ns=30_000_000)
        row, _ = tracker.process(blank, stream_id="s", source_time_ns=60_000_000)
        self.assertGreater(row["rejected_flow"], 0)
        row, _ = tracker.process(blank, stream_id="new", source_time_ns=90_000_000)
        self.assertEqual(row["reset_reason"], "stream_changed")

    def test_config_validation(self):
        for config in (
            TrackingConfig(max_features=0),
            TrackingConfig(fb_threshold_px=float("nan")),
        ):
            with self.assertRaises(ValueError):
                FeatureTracker(config)

    def test_capture_export_and_asset_integrity(self):
        import json
        import tempfile
        from pathlib import Path
        from hashlib import sha256
        from fusionlab.vision.feature_tracking import track_capture
        from fusion_viewer import ViewerClient, ImageSample
        from fusion_viewer.recording import LogSink, read_log

        cv, np = self.cv, self.np
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = root / "capture"
            capture.mkdir()
            events = []
            for i in range(2):
                image = np.random.default_rng(8).integers(
                    0, 256, (120, 160, 3), dtype=np.uint8
                )
                _, encoded = cv.imencode(".jpg", image)
                raw = encoded.tobytes()
                (capture / f"{i}.jpg").write_bytes(raw)
                events.append(
                    dict(
                        type="CameraSample",
                        stream_id="s",
                        source_time_ns=i * 33_000_000,
                        received_monotonic_ns=i * 33_000_000,
                        image_path=f"{i}.jpg",
                        image_size_bytes=len(raw),
                        image_sha256=sha256(raw).hexdigest(),
                    )
                )
            (capture / "events.jsonl").write_text(
                "".join(json.dumps(e) + "\n" for e in events)
            )
            with ViewerClient(LogSink(root / "viewer")) as viewer:
                report = track_capture(capture, root / "analysis", viewer=viewer)
            self.assertEqual(report["frames"], 2)
            values = [
                v for _, v in read_log(root / "viewer") if isinstance(v, ImageSample)
            ]
            self.assertEqual(len(values), 2)
            self.assertEqual((values[0].width, values[0].height), (120, 160))
            track_capture(capture, root / "unrotated", display_rotation=0)
            rotated = json.loads(
                (root / "analysis" / "tracks.jsonl").read_text().splitlines()[0]
            )
            unrotated = json.loads(
                (root / "unrotated" / "tracks.jsonl").read_text().splitlines()[0]
            )
            self.assertEqual(rotated["tracks"], unrotated["tracks"])
            self.assertEqual(rotated["display_rotation_clockwise_deg"], 90)
            self.assertEqual(unrotated["display_rotation_clockwise_deg"], 0)
            # A viewer that writes inside the analysis directory must open only
            # after the tracker reserves that directory.
            from contextlib import contextmanager

            opened = []

            @contextmanager
            def nested_export():
                opened.append(True)
                (root / "nested" / "replay.rrd").write_bytes(b"test export")
                yield None

            track_capture(capture, root / "nested", viewer_factory=nested_export)
            self.assertEqual(opened, [True])
            with self.assertRaises(FileExistsError):
                track_capture(capture, root / "nested", viewer_factory=nested_export)
            self.assertEqual(opened, [True])
            (capture / "0.jpg").write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                track_capture(capture, root / "bad")
