"""Synthetic localization scene, independent of sensor acquisition or a filter."""

import argparse
from math import cos, sin, exp
from pathlib import Path
from random import Random
from fusion_viewer import (
    ViewerClient,
    RunSpec,
    FrameSpec,
    SampleTime,
    Map2D,
    Pose2D,
    Trajectory2D,
    Particles2D,
)
from fusion_viewer.recording import LogSink


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("localization-demo.fvl"))
    args = parser.parse_args()
    random = Random(7)
    with ViewerClient(LogSink(args.output)) as viewer:
        run = viewer.register_run(
            RunSpec(
                "synthetic",
                "SYNTHETIC room, poses and particles; no localization algorithm",
            )
        )
        viewer.register_frame(
            run, FrameSpec("map", "XY metric map; +X right, +Y up; +Z out of plane")
        )
        walls = (
            ((0.0, 0.0), (8.0, 0.0)),
            ((8.0, 0.0), (8.0, 6.0)),
            ((8.0, 6.0), (0.0, 6.0)),
            ((0.0, 6.0), (0.0, 0.0)),
            ((5.0, 1.0), (5.0, 3.0)),
        )
        path = []
        for i in range(301):
            stamp = SampleTime("demo", "simulation", i * 100_000_000)
            if i == 0:
                viewer.publish(
                    run,
                    Map2D(entity_id="room", time=stamp, frame_id="map", walls=walls),
                )
            angle = i * 0.02
            x, y = 4 + 2 * cos(angle), 3 + 1.5 * sin(angle)
            heading = angle + 1.5707963267948966
            estimate = (x + 0.12 * sin(2 * angle), y + 0.1 * cos(3 * angle))
            viewer.publish(
                run,
                Pose2D(
                    entity_id="truth",
                    time=stamp,
                    frame_id="map",
                    xy=(x, y),
                    heading_rad=heading,
                    role="ground_truth",
                ),
            )
            viewer.publish(
                run,
                Pose2D(
                    entity_id="estimate",
                    time=stamp,
                    frame_id="map",
                    xy=estimate,
                    heading_rad=heading + 0.05,
                ),
            )
            path.append(estimate)
            viewer.publish(
                run,
                Trajectory2D(
                    entity_id="path", time=stamp, frame_id="map", points=tuple(path)
                ),
            )
            particles = tuple(
                (
                    estimate[0] + random.gauss(0, 0.3),
                    estimate[1] + random.gauss(0, 0.3),
                    heading + random.gauss(0, 0.18),
                )
                for _ in range(150)
            )
            weights = tuple(
                exp(-((p[0] - estimate[0]) ** 2 + (p[1] - estimate[1]) ** 2) / 0.18)
                for p in particles
            )
            viewer.publish(
                run,
                Particles2D(
                    entity_id="particles",
                    time=stamp,
                    frame_id="map",
                    poses=particles,
                    weights=weights,
                ),
            )
    print(f"Open with: fusion-viewer open {args.output}")


if __name__ == "__main__":
    main()
