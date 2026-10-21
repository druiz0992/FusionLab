"""Standalone producer: only fusion_viewer imports, no sensor acquisition."""

from math import sin, cos
from pathlib import Path
from fusion_viewer import (
    ViewerClient,
    RunSpec,
    FrameSpec,
    SampleTime,
    ScalarSample,
    OrientationSample,
)
from fusion_viewer.recording import LogSink


def main():
    with ViewerClient(LogSink(Path("demo.fvl"))) as viewer:
        run = viewer.register_run(RunSpec("demo", "Independent synthetic producer"))
        viewer.register_frame(
            run, FrameSpec("reference", "Right-handed synthetic reference")
        )
        for i in range(200):
            time = SampleTime("segment", "synthetic", i * 20_000_000)
            angle = i * 0.02
            viewer.publish(
                run,
                ScalarSample(
                    entity_id="signals/angle", time=time, value=angle, units="rad"
                ),
            )
            viewer.publish(
                run,
                OrientationSample(
                    entity_id="orientation/body",
                    time=time,
                    frame_id="reference",
                    quaternion_xyzw=(0.0, 0.0, sin(angle / 2), cos(angle / 2)),
                ),
            )
    print("Open with: fusion-viewer open demo.fvl")


if __name__ == "__main__":
    main()
