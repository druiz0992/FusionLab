# 7. Benchmarking and evaluation

Updated 2026-10-21. Benchmarks distinguish estimator problems from phone
calibration/capture uncertainty. Commands are in the
[README](../README.md#benchmark-against-ground-truth-euroc); results belong in
[3. Experiment results](3_experiment_results.md).

## EuRoC monocular benchmark

The EuRoC workflow imports cam0 only for estimation; IMU, cam1 and ground truth
are not estimator inputs. Ground truth is read only during evaluation.

Dataset and downloads: [official EuRoC page](https://projects.asl.ethz.ch/datasets/euroc-mav/).
Start with `MH_01_easy` in extracted ASL format (a `mav0/` directory containing
`cam0/` and `state_groundtruth_estimate0/`), placed at
`data/benchmarks/euroc/source/MH_01_easy` (or point directly at `mav0/`). The ETH
archive can package multiple sequences; extract only the required one. Respect the
provider's license. This project does not redistribute the dataset or
automatically unpack untrusted archives.

### Import

`fusionlab import-euroc` creates a new directory and copies original PNGs without
decoding or re-encoding them. Source paths/timestamps and hashes are recorded;
synthetic receive times are relative source times, not measured laptop arrival.
`calibration.json` converts the supplied pinhole radial-tangential cam0 model to
FusionLab's five-coefficient order, appending zero k3; other camera models are
rejected. `groundtruth.json` retains body poses and the camera-to-body transform.
The imported layout supports tracking; it is not a complete SensorStream capture
with live acquisition statistics or sensor declarations.

### Run

Track with `--display-rotation 0`, then run two configurations on the same tracks,
frame selection and calibration:

- baseline: `--stride 1 --seed 0`
- local BA: `--stride 1 --seed 0 --landmark-policy refinement --bundle-adjustment`

All output directories must be new. Start with all frames; do not inherit the
phone's stride five. For a smoke test, add `--max-frames 100` to tracking with a
separate output name, and do not present that shortened run as a full benchmark.

### Evaluate

`fusionlab evaluate-trajectory RUN --groundtruth groundtruth.json --output-dir DIR`
writes `report.md`, `summary.json`, `aligned.jsonl` and `trajectory.svg` (when an
alignment is possible).

- Evaluation interpolates reference body translation and SLERPs orientation, then
  applies `T_world_camera = T_world_body T_body_camera`.
- Integer timestamps are preserved until subtracting the epoch; there is no
  extrapolation, and interpolation is refused over gaps larger than `--max-gap-ms`
  (default 50).
- EuRoC quaternion CSV order is w,x,y,z; estimator poses are world-to-camera.
- Each segment receives its own positive-scale similarity alignment fitted on the
  scored positions. Reports include position RMSE/median/max, rotation RMSE,
  fitted scale, matched counts, segment spans, losses and frame coverage.
- Fewer than three matches or collinear/stationary motion cannot constrain the
  alignment and are reported as unavailable. Frames without reference remain in
  continuity counts.

**These are per-segment consistency scores, not a whole-route drift score.**
Alignment on short fragments can conceal failures. Always compare coverage,
segment lengths and resets alongside error. Frame coverage is measured over
processed frames, not continuous time. Recovered alignment scale is supplied by
ground truth; it is not metric scale estimated by monocular VO.

## ORB-SLAM3 reference

The [external ORB-SLAM3 driver](../benchmarks/orbslam3/README.md) documents its
C++ dependencies, pinned source revisions, build commands, frame preparation and
per-frame tracking-state report. It runs monocular full-SLAM on the phone captures
using RGB only and the selected phone calibration; it does not require EuRoC.
A EuRoC reference run and a ground-truth scoring adapter for its output remain
pending. Record frame selection, initialization, loop closure and relocalization
settings; do not compare monocular VO against stereo or inertial estimation.

## Other benchmark scripts

- `benchmarks/camera_receiver.py`: local TCP camera receive throughput test using a saved JPEG (no phone needed);
  not an accuracy benchmark.

## Validation status

Importer/evaluator tests use generated fixtures with known transforms, timestamps,
scale and missing-reference cases. They do not establish real EuRoC performance.
The first real MH_01_easy run is pending dataset access; see
[next work](2_status_and_roadmap.md#next-work).
