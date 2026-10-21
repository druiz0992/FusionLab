"""Depth validity and uncalibrated RGB correspondence diagnostics."""

from bisect import bisect_left
from collections import Counter
from math import isfinite
from pathlib import Path
import json

from .capture import load_capture
from .depth import depth_values
from .camera import image_path

# Method caveats saved with every report (what the numbers do not show).
DEPTH_ANALYSIS_INTERPRETATION = (
    "Zero, negative, nonfinite and uint16 maximum are counted separately; "
    "invalid/saturation semantics are not proven. Timestamp pairing is not exposure "
    "synchronization. RGB resizing and edge overlap are diagnostics, not calibrated "
    "registration."
)


def pair_frames(
    depth: list[dict], cameras: list[dict], tolerance_ms: float
) -> tuple[list[dict], int]:
    if not isfinite(tolerance_ms) or tolerance_ms < 0:
        raise ValueError("Pairing tolerance must be finite and nonnegative")
    # Pair only within a connection session; source clocks may reset on reconnect.
    groups = {}
    for camera in cameras:
        groups.setdefault(camera["session_id"], []).append(camera)
    for rows in groups.values():
        if len({e["stream_id"] for e in rows}) > 1:
            raise ValueError(
                "Multiple RGB streams in a session; select one with --camera-stream"
            )
        rows.sort(key=lambda e: e["source_time_ns"])
    indexes = {
        session: [e["source_time_ns"] for e in rows] for session, rows in groups.items()
    }
    pairs = []
    unmatched = 0
    for frame in depth:
        rows = groups.get(frame["session_id"], [])
        pos = bisect_left(indexes.get(frame["session_id"], []), frame["source_time_ns"])
        # In sorted timestamps the nearest frame must be just before or after
        # the insertion point. RGB reuse is allowed; this is not one-to-one matching.
        candidates = rows[max(0, pos - 1) : pos + 1]
        if not candidates:
            unmatched += 1
            continue
        camera = min(
            candidates, key=lambda e: abs(e["source_time_ns"] - frame["source_time_ns"])
        )
        delta = (camera["source_time_ns"] - frame["source_time_ns"]) / 1e6
        if abs(delta) > tolerance_ms:
            unmatched += 1
            continue
        pairs.append({"depth": frame, "camera": camera, "camera_minus_depth_ms": delta})
    return pairs, unmatched


def analyze(
    capture: Path,
    output: Path,
    *,
    tolerance_ms: float = 20.0,
    camera_stream: str | None = None,
    edge_threshold: float = 100.0,
    plots: bool = False,
) -> dict:
    try:
        import numpy as np
    except ImportError as exc:
        raise ValueError('Depth analysis requires pip install -e ".[plot]"') from exc
    if not isfinite(edge_threshold) or edge_threshold <= 0:
        raise ValueError("Edge threshold must be finite and positive")
    _, events = load_capture(capture)
    depth = [e for e in events if e["type"] == "DepthSample"]
    cameras = [
        e
        for e in events
        if e["type"] == "CameraSample"
        and (camera_stream is None or e["stream_id"] == camera_stream)
    ]
    if not depth:
        raise ValueError("No depth frames in capture")
    pairs, unmatched = pair_frames(depth, cameras, tolerance_ms)
    maps = {}
    totals = Counter()
    failures = []
    for e in depth:
        if e.get("file_check") != "ok" or e.get("depth_check") != "ok":
            failures.append(
                {
                    "path": e["image_path"],
                    "file_check": e.get("file_check"),
                    "depth_check": e.get("depth_check"),
                }
            )
            continue
        values = np.asarray(depth_values(capture, e), dtype=float).reshape(
            e["height"], e["width"]
        )
        finite = np.isfinite(values)
        zero = finite & (values == 0)
        negative = finite & (values < 0)
        maximum = (
            (values == 65535)
            if e["encoding"].lower() == "16uc1"
            else np.zeros(values.shape, dtype=bool)
        )
        totals.update(
            pixels=values.size,
            zero=int(zero.sum()),
            negative=int(negative.sum()),
            nonfinite=int((~finite).sum()),
            uint16_max=int(maximum.sum()),
            frames=1,
        )
        # Accumulate pixel-frequency maps only for compatible stream/image layouts.
        key = (e["session_id"], e["stream_id"], e["width"], e["height"], e["encoding"])
        if key not in maps:
            maps[key] = [np.zeros(values.shape), np.zeros(values.shape), 0]
        maps[key][0] += zero
        maps[key][1] += maximum
        maps[key][2] += 1
    deltas = [p["camera_minus_depth_ms"] for p in pairs]
    reused = len(pairs) - len(
        {(p["camera"]["stream_id"], p["camera"]["image_path"]) for p in pairs}
    )
    report = {
        "depth_frames": len(depth),
        "rgb_frames": len(cameras),
        "analyzed_counts": dict(totals),
        "zero_fraction": (
            totals["zero"] / totals["pixels"] if totals["pixels"] else None
        ),
        "frame_failures": failures,
        "pairing_tolerance_ms": tolerance_ms,
        "paired_depth_frames": len(pairs),
        "unmatched_depth_frames": unmatched,
        "rgb_reuses": reused,
        "signed_offset_ms": (
            {"min": min(deltas), "median": float(np.median(deltas)), "max": max(deltas)}
            if deltas
            else None
        ),
        "absolute_offset_ms": (
            {
                "median": float(np.median(np.abs(deltas))),
                "p95": float(np.percentile(np.abs(deltas), 95)),
                "max": max(map(abs, deltas)),
            }
            if deltas
            else None
        ),
        "depth_edge_threshold_raw": edge_threshold,
        "interpretation": DEPTH_ANALYSIS_INTERPRETATION,
    }
    if plots:
        report["sampled_edge_checks"] = _plots(
            capture, output, maps, pairs, edge_threshold
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "pairs.jsonl").open("w") as file:
        for p in pairs:
            file.write(
                json.dumps(
                    {
                        "session_id": p["depth"]["session_id"],
                        "depth_path": p["depth"]["image_path"],
                        "rgb_path": p["camera"]["image_path"],
                        "depth_time_ns": p["depth"]["source_time_ns"],
                        "rgb_time_ns": p["camera"]["source_time_ns"],
                        "camera_minus_depth_ms": p["camera_minus_depth_ms"],
                    }
                )
                + "\n"
            )
    return report


def _plots(capture, output, maps, pairs, threshold):
    import numpy as np

    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from PIL import Image, ImageFilter
    except ImportError as exc:
        raise ValueError('Plots require pip install -e ".[plot]"') from exc
    output.mkdir(parents=True, exist_ok=True)
    for number, (key, (zeros, maximum, count)) in enumerate(maps.items()):
        fig = Figure(figsize=(10, 4))
        FigureCanvasAgg(fig)
        axes = fig.subplots(1, 2)
        for axis, data, title in zip(
            axes,
            (zeros / count, maximum / count),
            ("Zero frequency", "65535 frequency (candidate saturation/sentinel)"),
        ):
            image = axis.imshow(data, vmin=0, vmax=1, cmap="magma")
            axis.set_title(title)
            fig.colorbar(image, ax=axis)
        fig.suptitle(f"{key[1]}: {count} frames; raw pixel locations")
        fig.tight_layout()
        fig.savefig(output / f"depth_value_frequency_{number}.png", dpi=140)
    if not pairs:
        return []
    selected = sorted({round(i * (len(pairs) - 1) / 3) for i in range(4)})
    fig = Figure(figsize=(13, 4 * len(selected)))
    FigureCanvasAgg(fig)
    axes = fig.subplots(len(selected), 3, squeeze=False)
    metrics = []
    for row, index in enumerate(selected):
        pair = pairs[index]
        d, c = pair["depth"], pair["camera"]
        if d.get("file_check") != "ok" or c.get("file_check") != "ok":
            continue
        values = np.asarray(depth_values(capture, d), dtype=float).reshape(
            d["height"], d["width"]
        )
        valid = np.isfinite(values) & (values > 0)
        if d["encoding"].lower() == "16uc1":
            valid &= values != 65535
        # Compare adjacent valid pixels in raw depth units; candidate invalid values
        # are excluded so sentinel boundaries do not become apparent scene edges.
        edge = np.zeros(values.shape, dtype=bool)
        edge[:, :-1] |= (
            valid[:, :-1]
            & valid[:, 1:]
            & (np.abs(values[:, :-1] - values[:, 1:]) > threshold)
        )
        edge[:-1, :] |= (
            valid[:-1, :]
            & valid[1:, :]
            & (np.abs(values[:-1, :] - values[1:, :]) > threshold)
        )
        with Image.open(image_path(capture, c["image_path"])) as image:
            resized = image.convert("RGB").resize((d["width"], d["height"]))
            rgb = np.asarray(resized)
            gray = np.asarray(resized.convert("L"), dtype=float)
        rgb_edge = np.zeros(values.shape, dtype=bool)
        rgb_edge[:, :-1] |= np.abs(gray[:, :-1] - gray[:, 1:]) > 20
        rgb_edge[:-1, :] |= np.abs(gray[:-1, :] - gray[1:, :]) > 20
        dilated = (
            np.asarray(
                Image.fromarray(rgb_edge.astype("uint8") * 255).filter(
                    ImageFilter.MaxFilter(5)
                )
            )
            > 0
        )
        overlap = float(np.mean(dilated[edge])) if edge.any() else None
        metrics.append(
            {
                "depth_path": d["image_path"],
                "depth_edge_pixels": int(edge.sum()),
                "fraction_near_rgb_edge_within_2_depth_pixels": overlap,
                "rgb_gray_gradient_threshold": 20,
            }
        )
        overlay = rgb.copy()
        overlay[edge] = [255, 0, 255]
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title("RGB resized to depth; no registration")
        axes[row, 1].imshow(np.ma.masked_where(~valid, values), cmap="viridis")
        axes[row, 1].set_title("Positive raw depth (candidate sentinels masked)")
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title(
            f"Depth edges (magenta); Δt={pair['camera_minus_depth_ms']:.1f} ms"
        )
        for axis in axes[row]:
            axis.axis("off")
    fig.suptitle(
        "Uncalibrated edge diagnostic — coincidence is not pixel accuracy or a fitted transform"
    )
    fig.tight_layout()
    fig.savefig(output / "rgb_depth_edges.png", dpi=140)
    return metrics
