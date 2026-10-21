# ORB-SLAM3 walking comparison

This external C++ runner checks whether the phone recording also loses tracking
in an established monocular implementation. ORB-SLAM3 is not imported into the
Python packages, and its source/vocabulary are not vendored here.

Upstream: <https://github.com/UZ-SLAMLab/ORB_SLAM3>

## Configuration

- Official ORB-SLAM3 commit `4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4`.
- Pangolin commit `3667bb76edcc3e4111d2bee7935ce204f04026d9`.
- System compiler, OpenCV 4.6, Eigen 3.4, Boost serialization and OpenSSL.
- Monocular mode: no IMU, depth, GPS or route constraints.
- Full SLAM: mapping, loop closure, relocalization and multiple maps enabled.
- Headless execution; Pangolin remains a required build dependency.
- 640×360 grayscale images from unchanged source JPEGs; intrinsics scaled by
  the actual image resize and all five distortion coefficients retained.
- Existing provisional phone calibration. No independent accuracy claim.
- Standard example ORB extractor settings: 1000 features, eight levels,
  scale factor 1.2, FAST thresholds 20/7.
- Frames paced by their source timestamps, with no deliberate frame dropping.
  Slow processing extends wall-clock duration; it does not skip input frames.

`prepare_build.py` adds a read-only current-map ID accessor and this driver to an
external checkout, and enables C++14 for modern Pangolin headers. It does not
change tracking, initialization, optimization or map policies. These source
changes can be reviewed with `git diff` in the external checkout.

## Build locally

Install compiler/development prerequisites using upstream documentation first.
The current session uses `/tmp/` for builds; retain checkouts elsewhere if needed
for future reruns. These are native C++ dependencies, not pip requirements.

```bash
git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git /tmp/fusionlab-orbslam3
git -C /tmp/fusionlab-orbslam3 checkout 4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4
git clone https://github.com/stevenlovegrove/Pangolin.git /tmp/fusionlab-pangolin
git -C /tmp/fusionlab-pangolin checkout 3667bb76edcc3e4111d2bee7935ce204f04026d9
cmake -S /tmp/fusionlab-pangolin -B /tmp/fusionlab-pangolin/build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/tmp/fusionlab-pangolin/install \
  -DBUILD_EXAMPLES=OFF -DBUILD_TOOLS=OFF -DBUILD_PANGOLIN_PYTHON=OFF -DBUILD_TESTS=OFF
cmake --build /tmp/fusionlab-pangolin/build -j4
cmake --install /tmp/fusionlab-pangolin/build
cmake -S /tmp/fusionlab-orbslam3/Thirdparty/DBoW2 \
  -B /tmp/fusionlab-orbslam3/Thirdparty/DBoW2/build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/fusionlab-orbslam3/Thirdparty/DBoW2/build -j4
cmake -S /tmp/fusionlab-orbslam3/Thirdparty/g2o \
  -B /tmp/fusionlab-orbslam3/Thirdparty/g2o/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=14
cmake --build /tmp/fusionlab-orbslam3/Thirdparty/g2o/build -j4
python3 benchmarks/orbslam3/prepare_build.py /tmp/fusionlab-orbslam3
tar -xf /tmp/fusionlab-orbslam3/Vocabulary/ORBvoc.txt.tar.gz -C /tmp/fusionlab-orbslam3/Vocabulary
cmake -S /tmp/fusionlab-orbslam3 -B /tmp/fusionlab-orbslam3/build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/tmp/fusionlab-pangolin/install
cmake --build /tmp/fusionlab-orbslam3/build --target fusionlab_walking -j3
```

## Prepare and execute

From the project root, use a new output directory:

```bash
WALK=data/sessions/walking/20261017T183215.982977Z-45b7f9c2
CALIBRATION=data/sessions/rgb_calibration/20261018T071611.049465Z-3cb5f83c/calibration/calibration.json
OUTPUT="$WALK/analysis/runs/orbslam3_new"
python3 benchmarks/orbslam3/walking.py prepare "$WALK" "$CALIBRATION" "$OUTPUT" --stride 1
LD_LIBRARY_PATH=/tmp/fusionlab-pangolin/install/lib \
/tmp/fusionlab-orbslam3/Examples/Monocular/fusionlab_walking \
  /tmp/fusionlab-orbslam3/Vocabulary/ORBvoc.txt "$OUTPUT/camera.yaml" \
  "$OUTPUT/frames.tsv" "$OUTPUT/states.csv" "$OUTPUT/keyframes.txt" \
  > "$OUTPUT/console.log" 2>&1
python3 benchmarks/orbslam3/walking.py summarize "$OUTPUT"
```

Repeat with `--stride 5` and a different directory to match the current FusionLab
run's selected source frames. A full-rate run is also important because reducing
image rate can itself make tracking harder. Runs are threaded and not guaranteed
to reproduce exactly; use repeats before making strong comparative claims.

To regenerate the tracking timeline with the project's Matplotlib dependency:

```bash
.venv/bin/python benchmarks/orbslam3/plot_tracking.py \
  "$WALK/analysis/runs/orbslam3_fullrate_640" \
  "$WALK/analysis/runs/orbslam3_stride5_640" \
  "$WALK/analysis/current/local_ba" \
  "$WALK/analysis/comparisons/orbslam3/tracking.png"
```

## Camera and trajectory replay

With the project's `viewer` and `odometry` extras installed:

```bash
.venv/bin/python benchmarks/orbslam3/export_replay.py "$OUTPUT" "$OUTPUT/replay.rrd"
.venv/bin/python -m rerun "$OUTPUT/replay.rrd"
```

The output must be new. The exporter checks that tracking timestamps cover the
selected input, verifies source image hashes, and logs camera video and online
camera centers on the source timeline. Each map gets a separate 3D panel with
arbitrary scale. Paths split on nontracking intervals or map changes, and the
current-camera marker clears on loss. Old map paths remain historical context.
Display rotation affects video only. There are no saved camera orientations, so
the replay shows points/paths rather than camera frustums. This external diagnostic
uses Rerun's 3D API directly; the reusable `fusion_viewer` trajectory API remains 2D.

These are the original online poses, not a reconstruction from the optimized
keyframe export. Later loop/map corrections do not retroactively update them.
The source map axes are not gravity-aligned, and returning to a physical starting
location does not imply displayed paths should close across independent maps.

## Interpretation

`states.csv` logs every input frame: original nanosecond source timestamp,
tracking state, active map ID, TrackMonocular processing time, and camera center
when tracked. Pose coordinates are arbitrary and map-local. They are online poses,
not retrospectively loop-corrected output. `keyframes.txt` is a supplementary
optimized keyframe export, not evidence of uninterrupted tracking.

`tracking_summary.json` counts OK/OK_KLT frames, transitions from tracking into
nontracking, map-ID changes and uninterrupted tracked spans. NOT_INITIALIZED,
RECENTLY_LOST and LOST remain distinct. Map-ID changes can arise from resets or
map merging; inspect `console.log` to interpret them. Compare time spans as well
as frame fractions, because the full-rate and stride-five denominators differ.

A successful SLAM run would show that these images support stronger tracking under
this configuration; it would not establish phone calibration accuracy. Failures
in both methods would not prove the images or calibration are solely responsible.

Build g2o with the same C++14 standard as ORB-SLAM3. On this host, mixing a
C++17 g2o build with C++14 ORB caused an Eigen allocation/deallocation failure
during initial bundle adjustment; that crash is not a tracking-loss result.

For a four-panel full-rate ablation timeline, add
`--fusionlab-fullrate PATH_TO_FULLRATE_FUSIONLAB_RUN` to `plot_tracking.py`.
FusionLab panel labels read the actual stride from each run summary.
