"""Offline pinhole RGB calibration in original stored pixel coordinates."""

import json
from math import isfinite
from pathlib import Path


def dependencies():
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise ValueError(
            'Calibration requires: pip install -e ".[calibration]"'
        ) from exc
    return cv2, np


def target(output: Path) -> None:
    """A4 landscape, 10x7 squares, 25 mm each (9x6 inner corners)."""
    output.mkdir(parents=True, exist_ok=True)
    rects = "".join(
        f'<rect x="{23.5+x*25}" y="{17.5+y*25}" width="25" height="25"/>'
        for y in range(7)
        for x in range(10)
        if (x + y) % 2 == 0
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="297mm" height="210mm" viewBox="0 0 297 210">
<rect width="297" height="210" fill="white"/><g fill="black">{rects}</g>
<text x="23.5" y="9" font-size="4">Fusion Lab: 10 x 7 squares; 9 x 6 inner corners; 25 mm squares</text>
<text x="23.5" y="201" font-size="3.5">Print A4 landscape at 100% / Actual size. Disable fit-to-page. Measure squares after printing.</text></svg>"""
    (output / "checkerboard.svg").write_text(svg)
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_pdf import FigureCanvasPdf
        from matplotlib.patches import Rectangle
    except ImportError as exc:
        raise ValueError(
            'SVG saved; PDF generation requires pip install -e ".[plot]"'
        ) from exc
    figure = Figure(figsize=(297 / 25.4, 210 / 25.4))
    FigureCanvasPdf(figure)
    axis = figure.add_axes((0, 0, 1, 1))
    axis.set(xlim=(0, 297), ylim=(210, 0))
    axis.axis("off")
    for y in range(7):
        for x in range(10):
            if (x + y) % 2 == 0:
                axis.add_patch(
                    Rectangle(
                        (23.5 + x * 25, 17.5 + y * 25),
                        25,
                        25,
                        color="black",
                        linewidth=0,
                    )
                )
    axis.text(
        23.5,
        9,
        "Fusion Lab: 10 x 7 squares; 9 x 6 inner corners; 25 mm squares",
        fontsize=9,
    )
    axis.text(
        23.5,
        201,
        "A4 landscape, Actual size / 100%. Disable fit-to-page. Measure printed squares.",
        fontsize=8,
    )
    figure.savefig(output / "checkerboard.pdf")
    (output / "target.json").write_text(
        json.dumps(
            {
                "squares": [10, 7],
                "inner_corners": [9, 6],
                "nominal_square_mm": 25,
                "page_mm": [297, 210],
                "board_mm": [250, 175],
            },
            indent=2,
        )
        + "\n"
    )


def fit_points(points, size, *, columns=9, rows=6, square_mm=25.0):
    cv, np = dependencies()
    if not isfinite(square_mm) or square_mm <= 0:
        raise ValueError("Measured square size must be positive and finite")
    if len(points) < 15:
        raise ValueError("Need at least 15 distinct detected views; aim for 20–30")
    # The checkerboard is a flat grid: Z is zero and XY spacing is in metres.
    # Each detected image supplies a different view of these same physical points.
    obj = np.zeros((columns * rows, 3), np.float32)
    obj[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * (square_mm / 1000)
    # Reserve every fifth view to assess reprojection with fixed fitted intrinsics.
    heldout = [i for i in range(len(points)) if i % 5 == 4]
    training = [i for i in range(len(points)) if i not in heldout]
    rms, K, dist, rvecs, tvecs = cv.calibrateCamera(
        [obj.copy() for _ in training], [points[i] for i in training], size, None, None
    )
    if not np.isfinite(K).all() or not np.isfinite(dist).all():
        raise ValueError("Nonfinite calibration result")
    errors = []
    tilts = []
    for i in range(len(points)):
        # Fit only the board pose for this view, then compare projected corners
        # with detected pixels; this also works for views excluded from calibration.
        ok, r, t = cv.solvePnP(obj, points[i], K, dist)
        if not ok:
            raise ValueError(f"Pose estimation failed for view {i}")
        projected, _ = cv.projectPoints(obj, r, t, K, dist)
        errors.append(
            float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            (projected.reshape(-1, 2) - points[i].reshape(-1, 2)) ** 2,
                            axis=1,
                        )
                    )
                )
            )
        )
        normal = cv.Rodrigues(r)[0][:, 2]
        tilts.append(float(np.degrees(np.arccos(np.clip(abs(normal[2]), 0, 1)))))
    # Corner coverage and board tilt reveal weak capture geometry that a low
    # reprojection error alone might hide.
    cloud = np.concatenate(points).reshape(-1, 2)
    coverage = float(
        cv.contourArea(cv.convexHull(cloud.astype(np.float32))) / (size[0] * size[1])
    )
    warnings = []
    if coverage < 0.35:
        warnings.append(
            "Limited image coverage (<35% convex hull); include edges and corners"
        )
    if max(tilts) < 20:
        warnings.append("Limited target tilt (<20 degrees); capture oblique views")
    if max(tilts) - min(tilts) < 15:
        warnings.append("Limited tilt diversity (<15 degrees)")
    if max(errors) > 1.0:
        warnings.append(
            "Some reprojection errors exceed 1 pixel; inspect corner detections"
        )
    if not (0 < K[0, 2] < size[0] and 0 < K[1, 2] < size[1]):
        warnings.append("Principal point outside image")
    return {
        "camera_matrix": K.tolist(),
        "distortion_coefficients": dist.ravel().tolist(),
        "distortion_order": ["k1", "k2", "p1", "p2", "k3"],
        "image_size": list(size),
        "inner_corners": [columns, rows],
        "measured_square_mm": square_mm,
        "training_view_indices": training,
        "heldout_view_indices": heldout,
        "training_rms_px": float(rms),
        "per_view_rms_px": errors,
        "heldout_rms_px": float(np.sqrt(np.mean([errors[i] ** 2 for i in heldout]))),
        "corner_coverage_fraction": coverage,
        "tilt_range_deg": [min(tilts), max(tilts)],
        "warnings": warnings,
        "status": "needs_review",
        "model": "OpenCV pinhole, five distortion coefficients, original stored pixels",
        "limits": "Held-out poses are fitted with fixed intrinsics. Low reprojection error is not proof of metric accuracy. No depth/IMU extrinsics or timing calibration.",
    }


def calibrate(
    capture: Path,
    output: Path,
    *,
    square_mm: float,
    columns=9,
    rows=6,
    stride=15,
    max_views=40,
    camera_stream=None,
) -> dict:
    from sensor_stream.inspection import load_capture
    from sensor_stream.camera_inspection import image_path

    cv, np = dependencies()
    if columns < 3 or rows < 3 or stride < 1 or max_views < 15:
        raise ValueError(
            "Use >=3 inner corners per axis, positive stride, >=15 max views"
        )
    if not isfinite(square_mm) or square_mm <= 0:
        raise ValueError("Measured square size must be positive and finite")
    _, events = load_capture(capture)
    frames = [
        e
        for e in events
        if e["type"] == "CameraSample"
        and (camera_stream is None or e["stream_id"] == camera_stream)
    ]
    if len({e["stream_id"] for e in frames}) > 1:
        raise ValueError("Select one RGB stream with --camera-stream")
    if not frames:
        raise ValueError("No RGB frames")
    if len({(e["width"], e["height"]) for e in frames}) != 1:
        raise ValueError(
            "Mixed image dimensions; use a consistent camera configuration"
        )
    size = (frames[0]["width"], frames[0]["height"])
    points = []
    accepted = []
    rejected = []
    output.mkdir(parents=True, exist_ok=True)
    for e in frames[::stride]:
        if len(points) >= max_views:
            break
        if e["file_check"] != "ok":
            raise ValueError(f"Image integrity failure: {e['image_path']}")
        image = cv.imread(str(image_path(capture, e["image_path"])))
        if image is None or image.shape[1::-1] != size:
            raise ValueError("Unreadable image or dimension mismatch")
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        ok, corners = cv.findChessboardCornersSB(
            gray, (columns, rows), flags=cv.CALIB_CB_NORMALIZE_IMAGE
        )
        if not ok:
            rejected.append(
                {"image": e["image_path"], "reason": "no complete checkerboard"}
            )
            continue
        # Reject nearly identical detected corner layouts (including reversed order).
        scale = np.array(size, dtype=float)
        if any(
            min(
                float(np.sqrt(np.mean(((corners - p) / scale) ** 2))),
                float(np.sqrt(np.mean(((corners[::-1] - p) / scale) ** 2))),
            )
            < 0.015
            for p in points
        ):
            rejected.append({"image": e["image_path"], "reason": "near-duplicate pose"})
            continue
        points.append(corners)
        accepted.append(e["image_path"])
        cv.drawChessboardCorners(image, (columns, rows), corners, True)
        if not cv.imwrite(str(output / f"corners_{len(points)-1:03d}.jpg"), image):
            raise OSError("Failed to write corner diagnostic")
    detection = {
        "accepted_images": accepted,
        "rejected": rejected,
        "stride": stride,
        "max_views": max_views,
    }
    (output / "detections.json").write_text(json.dumps(detection, indent=2) + "\n")
    report = fit_points(points, size, columns=columns, rows=rows, square_mm=square_mm)
    report.update(
        capture=str(capture.resolve()),
        stream_id=frames[0]["stream_id"],
        opencv_version=cv.__version__,
        **detection,
    )
    (output / "calibration.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
