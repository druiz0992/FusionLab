"""Offline inspection of recorded captures; no network access required.

Kept apart from the acquisition core (source, protocol, recording): these
modules read finished recordings, verify assets and summarize or plot them.
They interpret nothing about units, calibration or synchronization.

- capture: load and validate a capture; summaries and overview plots
- camera, depth, gps: per-sensor integrity checks, summaries and plots
- depth_analysis: raw depth validity and timestamp-paired RGB edges
- camera_replay: HTML replay comparing source and receive timing
"""

from .camera import image_path
from .capture import load_capture, plot_capture, summarize

__all__ = ["image_path", "load_capture", "plot_capture", "summarize"]
