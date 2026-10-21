"""Live preview latest-frame mailbox."""

import unittest

from sensor_stream import CameraSample
from sensor_stream.live import LatestFrame


def camera_frame(timestamp):
    return CameraSample(
        "camera", "session", timestamp, timestamp, timestamp, 16, 12, "jpeg", b"bytes"
    )


class MailboxTests(unittest.TestCase):
    def test_only_latest_display_frame_retained(self):
        mailbox = LatestFrame()
        mailbox.submit(camera_frame(1))
        mailbox.submit(camera_frame(2))
        mailbox.submit(camera_frame(3))
        self.assertEqual(mailbox.skipped, 2)
        self.assertEqual(mailbox.take(), camera_frame(3))
        self.assertIsNone(mailbox.take())
