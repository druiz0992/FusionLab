"""Shi-Tomasi/LK tracker behavior on synthetic images."""

import unittest

from fusionlab.vision.feature_tracking import FeatureTracker, TrackingConfig

from support import require


class FeatureTrackingTests(unittest.TestCase):
    def setUp(self):
        self.cv, self.np = require("cv2", "numpy")

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
