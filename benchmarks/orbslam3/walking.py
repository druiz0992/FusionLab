"""Prepare a verified RGB manifest/configuration or summarize ORB tracking states."""

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import statistics


def prepare(capture, calibration, output, stride=1, width=640):
    capture, calibration, output = map(Path, (capture, calibration, output))
    if stride < 1 or width < 1:
        raise ValueError("Stride and width must be positive")
    c = json.loads(calibration.read_text())
    if c["distortion_order"] != ["k1", "k2", "p1", "p2", "k3"]:
        raise ValueError("Expected pinhole radial-tangential calibration")
    original_width, original_height = c["image_size"]
    height = round(original_height * width / original_width)
    sx, sy = width / original_width, height / original_height
    k = c["camera_matrix"]
    distortion = c["distortion_coefficients"]
    events = []
    for line in (capture / "events.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["type"] == "CameraSample":
            events.append(row)
    if len({e["stream_id"] for e in events}) != 1:
        raise ValueError("Expected one RGB stream")
    if any(
        b["source_time_ns"] <= a["source_time_ns"] for a, b in zip(events, events[1:])
    ):
        raise ValueError("Non-increasing timestamps")
    selected = events[::stride]
    for e in selected:
        path = (capture / e["image_path"]).resolve()
        if not path.is_relative_to(capture.resolve()):
            raise ValueError("Image path escapes capture")
        data = path.read_bytes()
        if (
            len(data) != e["image_size_bytes"]
            or sha256(data).hexdigest() != e["image_sha256"]
        ):
            raise ValueError("Image integrity failure")
    fps = 1e9 / statistics.median(
        [
            b["source_time_ns"] - a["source_time_ns"]
            for a, b in zip(selected, selected[1:])
        ]
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "frames.tsv").write_text(
        "".join(
            f"{e['source_time_ns']}\t{(capture/e['image_path']).resolve()}\n"
            for e in selected
        )
    )
    config = {
        "Camera1.fx": k[0][0] * sx,
        "Camera1.fy": k[1][1] * sy,
        "Camera1.cx": k[0][2] * sx,
        "Camera1.cy": k[1][2] * sy,
        **dict(
            zip(
                ["Camera1.k1", "Camera1.k2", "Camera1.p1", "Camera1.p2", "Camera1.k3"],
                distortion,
            )
        ),
        "Camera.fps": round(fps),
        "Camera.RGB": 0,
        "Camera.width": width,
        "Camera.height": height,
        "ORBextractor.nFeatures": 1000,
        "ORBextractor.scaleFactor": 1.2,
        "ORBextractor.nLevels": 8,
        "ORBextractor.iniThFAST": 20,
        "ORBextractor.minThFAST": 7,
        "Viewer.KeyFrameSize": 0.05,
        "Viewer.KeyFrameLineWidth": 1.0,
        "Viewer.GraphLineWidth": 0.9,
        "Viewer.PointSize": 2.0,
        "Viewer.CameraSize": 0.08,
        "Viewer.CameraLineWidth": 3.0,
        "Viewer.ViewpointX": 0.0,
        "Viewer.ViewpointY": -0.7,
        "Viewer.ViewpointZ": -1.8,
        "Viewer.ViewpointF": 500.0,
    }
    (output / "camera.yaml").write_text(
        '%YAML:1.0\nFile.version: "1.0"\nCamera.type: "PinHole"\n'
        + "".join(f"{key}: {value}\n" for key, value in config.items())
    )
    metadata = dict(
        capture=str(capture.resolve()),
        calibration=str(calibration.resolve()),
        calibration_sha256=sha256(calibration.read_bytes()).hexdigest(),
        calibration_status=c.get("status", "needs_review"),
        stride=stride,
        image_size=[width, height],
        frames=len(selected),
        fps=fps,
        sensor="MONOCULAR; no IMU/depth/GPS",
        mode="Full SLAM, loop closure/relocalization/multiple maps enabled; viewer disabled",
        pacing="source time, no frames skipped if processing falls behind",
    )
    (output / "input.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def summarize(directory):
    directory = Path(directory)
    with (directory / "states.csv").open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("No processed states")
    counts = {}
    episodes = []
    good = []
    losses = 0
    maps = set()
    previous_ok = False
    for row in rows:
        state = int(row["state"])
        stamp = int(row["source_time_ns"])
        map_id = int(row["map_id"])
        counts[str(state)] = counts.get(str(state), 0) + 1
        maps.add(map_id)
        ok = state in (2, 5)
        if previous_ok and not ok:
            losses += 1
        if ok and (not good or good[-1][1] == map_id):
            good.append((stamp, map_id))
        else:
            if good:
                episodes.append(good)
            good = [(stamp, map_id)] if ok else []
        previous_ok = ok
    if good:
        episodes.append(good)
    durations = [(e[-1][0] - e[0][0]) / 1e9 for e in episodes]
    manifest = directory / "frames.tsv"
    expected = (
        [int(line.split("\t", 1)[0]) for line in manifest.read_text().splitlines()]
        if manifest.exists()
        else None
    )
    actual = [int(row["source_time_ns"]) for row in rows]
    state_time = {}
    for a, b in zip(rows, rows[1:]):
        key = a["state"]
        state_time[key] = (
            state_time.get(key, 0.0)
            + (int(b["source_time_ns"]) - int(a["source_time_ns"])) / 1e9
        )
    result = dict(
        state_sample_hold_duration_s=state_time,
        source_span_s=(actual[-1] - actual[0]) / 1e9,
        input_frames=len(expected) if expected is not None else None,
        complete_input=actual == expected if expected is not None else None,
        processed_frames=len(rows),
        tracked_frames=sum(counts.get(str(s), 0) for s in [2, 5]),
        tracked_fraction=sum(counts.get(str(s), 0) for s in [2, 5]) / len(rows),
        state_counts=counts,
        state_labels={
            "-1": "SYSTEM_NOT_READY",
            "0": "NO_IMAGES_YET",
            "1": "NOT_INITIALIZED",
            "2": "OK",
            "3": "RECENTLY_LOST",
            "4": "LOST",
            "5": "OK_KLT",
        },
        tracking_to_nontracking_transitions=losses,
        continuous_tracked_spans=len(episodes),
        longest_tracked_span_s=max(durations, default=0),
        observed_map_ids=sorted(maps),
        map_id_transitions=sum(
            a["map_id"] != b["map_id"] for a, b in zip(rows, rows[1:])
        ),
        median_processing_ms=statistics.median(float(r["processing_ms"]) for r in rows),
        limits="Tracking states and map continuity, not independent trajectory accuracy. Map IDs may change through reset/merge. Source pacing is best effort; all selected frames processed.",
    )
    (directory / "tracking_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("capture", type=Path)
    prep.add_argument("calibration", type=Path)
    prep.add_argument("output", type=Path)
    prep.add_argument("--stride", type=int, default=1)
    prep.add_argument("--width", type=int, default=640)
    report = sub.add_parser("summarize")
    report.add_argument("directory", type=Path)
    a = parser.parse_args()
    print(
        json.dumps(
            (
                prepare(a.capture, a.calibration, a.output, a.stride, a.width)
                if a.command == "prepare"
                else summarize(a.directory)
            ),
            indent=2,
        )
    )
