"""Optional Tk camera preview; GUI work stays on the main asyncio thread."""

import asyncio
from io import BytesIO
from time import monotonic, monotonic_ns
from typing import Callable

from .models import CameraSample
from .stats import SourceStats


class LatestFrame:
    """One-frame display mailbox, independent of lossless recording delivery."""

    def __init__(self) -> None:
        self.pending: CameraSample | None = None
        self.skipped = 0

    def submit(self, frame: CameraSample) -> None:
        if self.pending is not None:
            self.skipped += 1
        # Replace an unshown frame to keep preview latency bounded while recording
        # continues to receive events through its separate delivery path.
        self.pending = frame

    def take(self) -> CameraSample | None:
        frame, self.pending = self.pending, None
        return frame


class LiveCameraPreview:
    def __init__(self, stats: Callable[[], SourceStats], rotation: int = 0) -> None:
        try:
            import tkinter as tk
            from PIL import Image, ImageTk
        except ImportError as exc:
            raise ValueError(
                'Live preview requires Tk and Pillow. Install python3-tk on Ubuntu and pip install -e ".[live]"'
            ) from exc
        self.tk, self.Image, self.ImageTk = tk, Image, ImageTk
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            raise ValueError(
                "Cannot open live window; run from a graphical desktop with a usable display"
            ) from exc
        self.root.title("SensorStream live camera — R rotates; Q/Esc stops")
        self.root.geometry("1100x760")
        self.label = tk.Label(
            self.root, bg="black", text="Waiting for camera frames…", fg="white"
        )
        self.label.pack(fill="both", expand=True)
        self.status = tk.Label(self.root, text="Waiting for camera frames…", anchor="w")
        self.status.pack(fill="x")
        self.stopping = False
        self.rotation = rotation
        self.stats = stats
        self.frames = LatestFrame()
        self.photo = None
        self.displayed = 0
        self.decode_errors = 0
        self.last_receive: int | None = None
        self.started = monotonic()
        self.root.protocol("WM_DELETE_WINDOW", self.stop)
        self.root.bind("<Escape>", lambda _: self.stop())
        self.root.bind("q", lambda _: self.stop())
        self.root.bind("r", lambda _: self.rotate())
        self._last_image = None

    def stop(self) -> None:
        self.stopping = True

    def rotate(self) -> None:
        self.rotation = (self.rotation + 90) % 360
        if self._last_image is not None:
            self._show(self._last_image)

    def submit(self, sample: CameraSample) -> None:
        self.frames.submit(sample)

    def _decode(self, data: bytes):
        with self.Image.open(BytesIO(data)) as image:
            image.load()
            return image.convert("RGB")

    def _show(self, image) -> None:
        # Display-only clockwise rotation: recorded bytes and coordinates stay intact.
        image = image.rotate(-self.rotation, expand=True)
        image.thumbnail(
            (max(64, self.label.winfo_width()), max(64, self.label.winfo_height()))
        )
        # Keep a Python reference: Tk does not keep the PhotoImage alive by itself.
        self.photo = self.ImageTk.PhotoImage(image, master=self.root)
        self.label.configure(image=self.photo, text="")

    async def run(self) -> None:
        while not self.stopping:
            self.root.update()
            if self.stopping:
                break
            frame = self.frames.take()
            if frame is not None:
                try:
                    # Decode off the event loop; update Tk widgets back on the main thread.
                    image = await asyncio.to_thread(self._decode, frame.image_data)
                    self._last_image = image
                    self._show(image)
                    self.displayed += 1
                    self.last_receive = frame.received_monotonic_ns
                except (OSError, ValueError):
                    self.decode_errors += 1
            stats = self.stats()
            elapsed = max(monotonic() - self.started, 0.001)
            age = (
                (monotonic_ns() - self.last_receive) / 1e9
                if self.last_receive
                else None
            )
            self.status.configure(
                text=(
                    f"{'Connected' if stats.connected else 'Waiting / disconnected'} | "
                    f"Receive {stats.observed_camera_rate_hz:.1f} FPS | Display avg {self.displayed/elapsed:.1f} FPS | "
                    f"Relative backlog {stats.camera_relative_delay_s:.2f} s | "
                    f"Displayed frame receive-age {age:.2f} s | "
                    if age is not None
                    else "Waiting for camera frames… | "
                )
                + f"Display skipped {self.frames.skipped} | Decode errors {self.decode_errors} | R rotates, Q stops"
            )
            await asyncio.sleep(1 / 30)

    def close(self) -> None:
        self.stopping = True
        self.root.destroy()
