"""Export source-timed camera images and separate map-local ORB online paths."""

import argparse
import csv
from hashlib import sha256
import json
import math
from pathlib import Path

STATE_NAMES = {
    0: "reset/no image",
    1: "initializing",
    2: "tracking",
    3: "recently lost",
    4: "lost",
    5: "tracking KLT",
}


def trajectory_updates(rows):
    """Never connect a path across a nontracking interval or active-map change."""
    active_map = None
    span = -1
    path = []
    for row in rows:
        state, map_id = int(row["state"]), int(row["map_id"])
        if state not in (2, 5):
            path = []
            active_map = None
            yield row, None, []
            continue
        if map_id != active_map:
            path = []
            span += 1
            active_map = map_id
        center = [float(row[key]) for key in ["cx", "cy", "cz"]]
        if not all(math.isfinite(v) for v in center):
            raise ValueError("Nonfinite tracked camera position")
        path.append(center)
        yield row, span, list(path)


def export_replay(run_dir, output):
    import cv2 as cv
    import numpy as np
    import rerun as rr
    import rerun.blueprint as rb

    run_dir, output = Path(run_dir), Path(output)
    if output.exists():
        raise FileExistsError(output)
    inputs = json.loads((run_dir / "input.json").read_text())
    capture = Path(inputs["capture"])
    events = {
        e["source_time_ns"]: e
        for e in (
            json.loads(line)
            for line in (capture / "events.jsonl").read_text().splitlines()
        )
        if e["type"] == "CameraSample"
    }
    with (run_dir / "states.csv").open() as f:
        states = list(csv.DictReader(f))
    expected = [
        int(line.split("\t")[0])
        for line in (run_dir / "frames.tsv").read_text().splitlines()
    ]
    if [int(row["source_time_ns"]) for row in states] != expected:
        raise ValueError(
            "Tracking log does not cover the complete selected input in order"
        )
    maps = sorted({int(row["map_id"]) for row in states})
    output.parent.mkdir(parents=True, exist_ok=True)
    stream = rr.RecordingStream("FusionLab_ORB_SLAM3_inspection")
    stream.save(output)
    stream.send_blueprint(
        rb.Blueprint(
            rb.Vertical(
                rb.Horizontal(
                    rb.Spatial2DView(
                        name="Camera: source time / tracking status",
                        origin="/orb/camera",
                        contents="$origin/**",
                    ),
                    rb.Vertical(
                        *[
                            rb.Spatial3DView(
                                name=f"Map {m}: online path, arbitrary units",
                                origin=f"/orb/maps/{m}",
                                contents="$origin/**",
                            )
                            for m in maps
                        ]
                    ),
                    column_shares=[0.4, 0.6],
                ),
                rb.TimeSeriesView(
                    name="Native state: 1 init, 2 tracking, 3 recently lost, 4 lost; map ID",
                    contents=["/orb/signals/**"],
                ),
                row_shares=[0.8, 0.2],
            ),
            rb.TimePanel(timeline="source"),
            auto_layout=False,
            auto_views=False,
        )
    )
    stream.log(
        "/orb/info",
        rr.TextDocument(
            "Online camera centers from the successful ORB run. Each map is independent, arbitrary scale; initial-camera axes, not a gravity-aligned floor plan. Paths split on tracking loss. No camera orientation/frustum was recorded. These are not retrospectively loop-corrected poses. Ground truth and verified endpoint closure are unavailable."
        ),
        static=True,
    )
    for m in maps:
        stream.log(f"/orb/maps/{m}", rr.ViewCoordinates.RDF, static=True)
    origin = int(states[0]["source_time_ns"])
    previous_map = None
    for row, span, path in trajectory_updates(states):
        stamp = int(row["source_time_ns"])
        state = int(row["state"])
        map_id = int(row["map_id"])
        event = events[stamp]
        image_path = (capture / event["image_path"]).resolve()
        if not image_path.is_relative_to(capture.resolve()):
            raise ValueError("Image path escapes capture")
        data = image_path.read_bytes()
        if (
            len(data) != event["image_size_bytes"]
            or sha256(data).hexdigest() != event["image_sha256"]
        ):
            raise ValueError("Image integrity failure")
        image = cv.imdecode(np.frombuffer(data, np.uint8), cv.IMREAD_COLOR)
        if image is None:
            raise ValueError("Unreadable image")
        width, height = inputs["image_size"]
        image = cv.rotate(
            cv.resize(image, (width, height), interpolation=cv.INTER_AREA),
            cv.ROTATE_90_CLOCKWISE,
        )
        banner = np.zeros((84, image.shape[1], 3), np.uint8)
        for index, text in enumerate(
            [
                f"ORB: {STATE_NAMES.get(state,str(state))} | map {map_id}",
                f'Source {(stamp-origin)/1e9:.2f}s | {inputs["stride"]} frame stride',
                "Map-local online path; NOT metres",
            ]
        ):
            cv.putText(
                banner,
                text,
                (6, 20 + index * 24),
                cv.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv.LINE_AA,
            )
        image = np.vstack([banner, image])
        ok, jpeg = cv.imencode(".jpg", image)
        if not ok:
            raise ValueError("Image encode failed")
        stream.set_time("source", duration=(stamp - origin) / 1e9)
        stream.log(
            "/orb/camera/image",
            rr.EncodedImage(contents=jpeg.tobytes(), media_type="image/jpeg"),
        )
        stream.log("/orb/signals/native_state", rr.Scalars(state))
        stream.log("/orb/signals/map_id", rr.Scalars(map_id))
        if previous_map is not None and (state not in (2, 5) or previous_map != map_id):
            stream.log(
                f"/orb/maps/{previous_map}/current_camera", rr.Clear(recursive=True)
            )
        if path:
            if len(path) > 1:
                stream.log(
                    f"/orb/maps/{map_id}/span_{span}",
                    rr.LineStrips3D([path], colors=[[70, 170, 255]], radii=0.003),
                )
            stream.log(
                f"/orb/maps/{map_id}/current_camera",
                rr.Points3D([path[-1]], colors=[[255, 130, 50]], radii=0.015),
            )
        previous_map = map_id
    stream.flush()
    result = dict(
        frames=len(states),
        maps=maps,
        source_span_s=(int(states[-1]["source_time_ns"]) - origin) / 1e9,
        recording=str(output.resolve()),
        source_tracking_log=str((run_dir / "states.csv").resolve()),
        limits="Separate map-local online trajectories; arbitrary scale; no estimated orientations or independent accuracy validation.",
    )
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(export_replay(args.run_dir, args.output), indent=2))
