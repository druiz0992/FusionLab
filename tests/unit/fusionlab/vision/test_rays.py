"""Undistorted track rays and tracking-pixel camera matrices."""

import unittest

from fusionlab.vision.rays import track_rays, tracking_camera_matrix
from fusionlab.vision.tracks import TrackFrame

from support import require
from support.vision import calibration_json, track_row


class TrackRayTests(unittest.TestCase):
    def setUp(self):
        (self.np,) = require("numpy")
        require("cv2")

    def test_principal_point_and_focal_offset_map_to_known_rays(self):
        # Synthetic camera: f = 500 px, principal point (320, 240), no distortion.
        tracks = [{"id": 1, "xy": [320.0, 240.0]}, {"id": 2, "xy": [370.0, 190.0]}]
        rays, camera = track_rays(
            TrackFrame.from_json(track_row(0, tracks)), calibration_json()
        )
        self.np.testing.assert_allclose(rays[1], [0.0, 0.0], atol=1e-12)
        self.np.testing.assert_allclose(rays[2], [0.1, -0.1], atol=1e-12)
        self.assertEqual(camera[0, 0], 500.0)

    def test_camera_matrix_scales_from_original_to_tracking_pixels(self):
        row = track_row(0)
        row.update(tracking_size=[320, 240], pixel_scale_xy=[0.5, 0.5])
        camera = tracking_camera_matrix(TrackFrame.from_json(row), calibration_json())
        self.np.testing.assert_allclose(camera[:2, :], [[250, 0, 160], [0, 250, 120]])

    def test_rejects_mismatched_geometry_and_models(self):
        row = TrackFrame.from_json(track_row(0))
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            tracking_camera_matrix(
                row, calibration_json() | {"image_size": [1920, 1080]}
            )
        with self.assertRaisesRegex(ValueError, "distortion model"):
            tracking_camera_matrix(row, calibration_json() | {"distortion_order": []})
        inconsistent = track_row(0)
        inconsistent.update(tracking_size=[320, 240])
        with self.assertRaisesRegex(ValueError, "pixel scale"):
            tracking_camera_matrix(
                TrackFrame.from_json(inconsistent), calibration_json()
            )

    def test_empty_frame_has_no_rays(self):
        rays, _ = track_rays(TrackFrame.from_json(track_row(0)), calibration_json())
        self.assertEqual(rays, {})


if __name__ == "__main__":
    unittest.main()
