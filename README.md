# FusionLab

A Python 3.12 laboratory for learning and comparing localization algorithms from
recorded iPhone SensorStream data (IMU, GPS, RGB and rear depth). No ROS2 needed.

**Current milestone:** experimental camera-only monocular visual odometry. It still
loses tracking and does not reconstruct a walk continuously; see
[status](docs/2_status_and_roadmap.md).

| Read next | Contents |
| --- | --- |
| [AGENTS.md](AGENTS.md) | Working rules for every development session |
| [docs/1_project_overview.md](docs/1_project_overview.md) | Purpose, packages, source map, data layout |
| [docs/2_status_and_roadmap.md](docs/2_status_and_roadmap.md) | What works, what fails, what comes next |
| [docs/3_experiment_results.md](docs/3_experiment_results.md) | Measured outcomes of each attempted variant |
| [docs/4_sensors_and_acquisition.md](docs/4_sensors_and_acquisition.md) | Sensor conventions, acquisition API, wire protocol |
| [docs/5_camera_and_visual_odometry.md](docs/5_camera_and_visual_odometry.md) | Calibration, feature tracking, VO pipeline |
| [docs/6_viewer.md](docs/6_viewer.md) | Standalone viewer API |
| [docs/7_benchmarking.md](docs/7_benchmarking.md) | EuRoC and ORB-SLAM3 comparisons, evaluation conventions |
| [data/README.md](data/README.md) | Active captures, selected results, archive |

## Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,odometry,viewer,plot,live,benchmark]"
```

Install only what you need: the base `-e .` is enough for capture.

| Extra | Adds | Needed for |
| --- | --- | --- |
| `tracking`, `calibration` | OpenCV, NumPy | Feature tracking, camera calibration |
| `odometry` | OpenCV, NumPy, SciPy | VO, landmark refinement, bundle adjustment |
| `benchmark` | PyYAML, NumPy, SciPy, OpenCV, Matplotlib | EuRoC import and evaluation |
| `plot` | Matplotlib | `--plot` outputs |
| `viewer` | Rerun SDK 0.39.0 | `--viewer`, `--rrd`, `fusion-viewer open` |
| `live` | Pillow | Live camera preview (Linux also needs `python3-tk`) |
| `dev` | black | Formatting |

USB capture needs usbmuxd, a data-capable cable, an unlocked phone and trust/pairing
when requested.

## Quick start: view the current result

```bash
WALK=data/sessions/walking/20261017T183215.982977Z-45b7f9c2
python -m rerun "$WALK/analysis/current/local_ba/replay.rrd"
```

The replay shows camera overlays, a segment-local XZ path and tracking diagnostics.
Units are **arbitrary**, not metres; paths clear on each tracking loss. The ranked
[motion diagnostic report](data/sessions/walking/20261017T183215.982977Z-45b7f9c2/analysis/current/diagnostics/index.html)
opens in a browser.

## Running the programs

Run from the project root with `.venv` active. Every command supports `--help`.
**Output directories for new experiments must be new**; put them under the
capture's `analysis/runs/<name>/`, not `/tmp`. Shared variables used below:

```bash
WALK=data/sessions/walking/20261017T183215.982977Z-45b7f9c2
CURRENT="$WALK/analysis/current"
TRACKS="$CURRENT/tracks/tracks.jsonl"
CALIBRATION=data/sessions/rgb_calibration/20261018T071611.049465Z-3cb5f83c/calibration/calibration.json
```

These are selected development inputs, not generic defaults. A calibration must
match the camera, zoom and processing, not just the resolution.

Three entry points are installed: `sensor-stream` (acquisition), `fusionlab`
(processing and algorithms) and `fusion-viewer` (viewer logs).

### Capture: `sensor-stream`

```bash
# Wi-Fi: phone and laptop on the same network; app sends to laptop LAN IP, port 5678.
sensor-stream listen --record --output-dir data/sessions/new_experiment
# USB, with optional live camera preview:
sensor-stream listen --transport usb --live --record --output-dir data/sessions/new_experiment
# Timed stop (timer starts at listener startup, including connection wait):
sensor-stream listen --record --duration 60 --output-dir data/sessions/timed
```

Enable the desired sensors in the app. Stop with Ctrl+C to drain and finalize
decoded samples. USB was the reliable choice for combined high-rate recording;
Wi-Fi camera streams accumulated substantial backlog in tests.

| Command | Purpose and options |
| --- | --- |
| `listen` | Receive and optionally record. `--transport {wifi,usb}`, `--host`, `--port` (Wi-Fi), `--usb-port` (phone port, default 2345), `--usb-udid`, `--record`, `--output-dir` (default `data/sessions`), `--duration SECONDS`, `--live`, `--display-rotation {0,90,180,270}` (preview only), `--queue-capacity`, `--overflow-policy {raise,drop_oldest}`, `--print-samples` |
| `inspect CAPTURE` | Summarize a capture. `--plot`, `--plot-output PATH` |
| `replay-camera CAPTURE` | Browser replay comparing source and receive timing. `--output HTML` |
| `analyze-depth CAPTURE` | Depth validity and timestamp-paired RGB edges. `--output-dir`, `--tolerance-ms`, `--camera-stream`, `--edge-threshold` (raw units, default 100), `--plot` |

### Inspect and visualize a capture

```bash
sensor-stream inspect "$WALK" --plot
fusionlab visualize "$WALK"
```

`fusionlab visualize CAPTURE` opens the Rerun viewer. Options: `--rrd PATH` (export
without GUI), `--export-log DIR` (portable log instead of Rerun),
`--depth-metres-per-unit` (explicit depth scale, e.g. `0.001` for tested rear depth).

### IMU baselines

```bash
fusionlab orientation "$WALK" --viewer
fusionlab dead-reckoning "$WALK" --acceleration-unit g --viewer --route "$WALK/route.json"
fusionlab replay "$WALK" --speed 1 --acceleration-unit g
```

| Command | Options |
| --- | --- |
| `orientation` | Gyro integration vs phone orientation. `--max-gap SECONDS`, `--inverse-reference`, `--plot`, `--plot-output`, `--output JSONL`, `--viewer`, `--rrd`, `--export-log` |
| `dead-reckoning` | Exploratory position integration. `--acceleration-unit {g,m/s2}`, `--bias-interval START END`, `--stationary START END` (repeatable), `--max-gap`, `--route route.json`, `--plot`, `--plot-output`, `--output`, `--summary-output`, `--viewer`, `--rrd`, `--export-log` |
| `replay` | IMU samples as JSON lines. `--speed` (0 immediate, 1 recorded timing), `--acceleration-unit` |

These are separate baselines, not fused inputs to VO.

### Camera calibration

```bash
fusionlab calibration-target                      # writes docs/calibration_target/
fusionlab calibrate-camera CALIBRATION_CAPTURE --square-mm 25
fusionlab validate-calibration VERIFICATION_CAPTURE EXISTING_CALIBRATION_JSON \
  --output-dir NEW_VALIDATION_DIRECTORY
```

`calibrate-camera` options: `--square-mm` (measured printed square), `--columns`/
`--rows` (inner corners, default 9×6), `--stride` (default every 15th frame),
`--max-views` (default 40), `--camera-stream`, `--output-dir`. Use the measured
square size of your own print.

`validate-calibration` holds an existing fit fixed on a **separate** capture and
scores held-out corners, writing `summary.json` to a new `--output-dir` (required).
Options: `--stride` (default every 60th frame), `--max-views` (default 40),
`--image-list` (a prior `detections.json` whose `accepted_images` replace stride
sampling). The full procedure is in
[camera calibration](docs/5_camera_and_visual_odometry.md#camera-calibration).

### Feature tracking and visual odometry

```bash
OUTPUT="$WALK/analysis/runs/tracking_new"
fusionlab track-features "$WALK" --output-dir "$OUTPUT" \
  --display-rotation 90 --rrd "$OUTPUT/replay.rrd"

fusionlab two-view "$WALK" --tracks "$TRACKS" \
  --frame-a 1000 --frame-b 1010 --output-dir "$WALK/analysis/runs/two_view_new"
fusionlab camera-motion --tracks "$TRACKS" --calibration "$CALIBRATION" \
  --frame-a 1000 --frame-b 1010 --output-dir "$WALK/analysis/runs/motion_new"

OUTPUT="$WALK/analysis/runs/vo_new"
fusionlab visual-odometry --tracks "$TRACKS" --calibration "$CALIBRATION" \
  --landmark-policy refinement --bundle-adjustment --seed 0 \
  --output-dir "$OUTPUT" --rrd "$OUTPUT/replay.rrd"
```

`track-features` options: `--output-dir`, `--max-width` (default 640),
`--max-features` (default 400), `--max-frames` (smoke tests), `--camera-stream`,
`--display-rotation {0,90,180,270}` (overlays only), `--viewer`, `--rrd`.

The VO command above reproduces the inspected experimental configuration, not the
CLI defaults. Omit `--bundle-adjustment` for landmark-only refinement.

| `visual-odometry` option | Values; default first |
| --- | --- |
| `--pose-acceptance` | `baseline`, `refined` |
| `--keyframe-policy` | `baseline`, `parallax` |
| `--landmark-policy` | `baseline`, `validation`, `refinement`, `multiview` |
| `--bundle-adjustment` | Off unless given (explicit opt-in) |
| `--stride` / `--seed` | 5 / 0 |
| `--max-frames`, `--viewer`, `--rrd` | Smoke tests and visualization |

Check [experiment results](docs/3_experiment_results.md) before choosing a variant.

### Diagnose a saved VO run

```bash
fusionlab diagnose-vo "$CURRENT/local_ba" --capture "$WALK" \
  --output-dir "$WALK/analysis/runs/motion_checks_new"
```

Replays the saved configuration, checks reproduction, then compares PnP, constant
velocity prediction and gyro. It can take minutes with BA. `--imu-offset-ms` shifts
camera timestamps when indexing IMU; `--camera-from-imu FILE` supplies a JSON 3×3
rotation for full-axis comparison (otherwise only rotation magnitudes are compared).

```bash
fusionlab diagnose-map "$CURRENT/local_ba" --tracks "$TRACKS" --calibration "$CALIBRATION" \
  --reference-states "$WALK/analysis/runs/orbslam3_fullrate_640/states.csv" \
  --output-dir "$WALK/analysis/runs/map_diagnostics_new"
```

`diagnose-map` reads a saved run (no replay) and reports, per loss, visible tracks
versus mapped landmarks, triangulation outcomes, the inferred reference-keyframe age,
the parallax each track offers against its first observation, and recent-versus-older
landmark errors at geometric losses. `--tracks`/`--calibration` override the paths
saved in the run (older runs record historical paths); `--reference-states` adds the
ORB-SLAM3 state at each loss.

### Benchmark against ground truth (EuRoC)

```bash
SOURCE=data/benchmarks/euroc/source/MH_01_easy      # extracted ASL sequence
CAPTURE=data/benchmarks/euroc/MH_01_easy
fusionlab import-euroc "$SOURCE" --output-dir "$CAPTURE"
fusionlab track-features "$CAPTURE" --display-rotation 0 \
  --output-dir "$CAPTURE/analysis/current/tracks"
fusionlab visual-odometry --tracks "$CAPTURE/analysis/current/tracks/tracks.jsonl" \
  --calibration "$CAPTURE/calibration.json" --stride 1 --seed 0 \
  --output-dir "$CAPTURE/analysis/runs/baseline"
fusionlab evaluate-trajectory "$CAPTURE/analysis/runs/baseline" \
  --groundtruth "$CAPTURE/groundtruth.json" \
  --output-dir "$CAPTURE/analysis/comparisons/baseline"
```

`evaluate-trajectory` option: `--max-gap-ms` (largest reference gap to interpolate,
default 50). Use stride 1 on EuRoC. For the comparison configuration, conventions
and how to read the report, see [benchmarking](docs/7_benchmarking.md). The
external ORB-SLAM3 driver is documented in [benchmarks/orbslam3](benchmarks/orbslam3/README.md).

### Viewer: `fusion-viewer` and examples

```bash
python examples/viewer_producer.py               # writes demo.fvl
fusion-viewer open demo.fvl
python examples/localization_demo.py --output localization-demo.fvl
fusion-viewer open localization-demo.fvl --rrd localization-demo.rrd
```

`fusion-viewer open LOG` opens a portable log in Rerun; `--rrd PATH` exports instead
of opening a GUI. The examples are synthetic rendering demos, not algorithms.

## Tests and formatting

```bash
python -m pip install -e ".[dev,odometry,viewer,plot,live,benchmark]"
PYTHONPATH=src python -m unittest discover -s tests -v                    # full suite
PYTHONPATH=src python -m unittest discover -s tests/unit -t tests         # unit only (fast)
PYTHONPATH=src python -m unittest discover -s tests/integration -t tests  # integration only
PYTHONPATH=src python -m unittest discover -s tests -p test_policies.py   # one file
python -m black --workers 1 src tests examples benchmarks
python3 scripts/restore_data_layout.py --check               # data layout audit
```

Unit tests in `tests/unit/` mirror the source tree (`sensor_stream`, `fusionlab`,
`fusion_viewer`, `benchmarks`); `tests/integration/` covers end-to-end capture,
pipelines and viewer exports; shared fixtures are in `tests/support/`. Tests whose
optional extras are missing are skipped, so install the extras before validating
algorithms. Generated Protobuf is excluded
from formatting. Passing tests verify controlled behavior on synthetic fixtures;
they do not establish real-world localization accuracy.

`scripts/restore_data_layout.py --check` verifies the recorded data consolidation;
`--restore` reverses those moves and must not be used just to validate (read the
[data index](data/README.md) first).
