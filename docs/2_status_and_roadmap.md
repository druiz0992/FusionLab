# 2. Status and roadmap

Updated 2026-10-18. Current milestone: **camera-only monocular VO research baseline**.
It is not reliable continuous localization. This document covers capabilities,
limitations and planned work; [3. Experiment results](3_experiment_results.md)
holds the evidence tables.

## What works

| Area | Current capability |
| --- | --- |
| Acquisition | IMU, GPS, RGB and rear depth recording over tested Wi-Fi/USB paths; inspection and shutdown draining |
| IMU baselines | Gyro integration and exploratory position integration; separate from VO |
| RGB calibration | Provisional 1920×1080 checkerboard fit, 34 views, 61% coverage, tilt up to 41°, held-out RMS 0.47 px |
| Visual frontend | Persistent LK feature tracks and two-view geometric diagnostics |
| Monocular VO | Triangulated landmarks, PnP, explicit losses/reinitialization and experimental refinements |
| Visualization | Source overlays, active segment-local path, diagnostics and portable/Rerun exports |
| Diagnostics | Failure reports, common-map CV comparison, spatial support and gyro angle checks |
| Benchmarking | EuRoC cam0 import and per-segment ground-truth evaluation (tested on generated fixtures only) |

Selected capture: [walking data index](../data/sessions/walking/20261017T183215.982977Z-45b7f9c2/README.md).
Selected calibration remains `needs_review`; it has not been independently validated
or silently applied to every camera recording.

## What does not work yet

On the original walking capture, the selected local-BA run has **29 tracking losses and
30 independent segments** over about 85 seconds. Its longest tracked span is
10.83 s. All scales are arbitrary and independent per segment. There is no valid
whole-walk trajectory, metric accuracy result or ground-truth-aligned evaluation.

VO uses RGB only. IMU, depth and GPS do not contribute motion constraints.
Independent IMU experiments and gyro diagnostic plots do not constitute fusion.

## Limitation register

| ID | Limitation | Work already attempted | Remaining action |
| --- | --- | --- | --- |
| L1 | Frequent geometric disagreement and losses | Failure diagnostics; pose acceptance A/B; keyframe-policy A/B | Validate against calibrated ground-truth data and an established baseline; inspect high-risk frames |
| L2 | Fragile two-view depth and map quality | Third-view validation, landmark-only refinement and ablation; map-maintenance diagnosis (2026-10-21): narrow reference baselines starve the map | Per-track triangulation against each track's first observation; parallax keyframes validated on ground truth; improvement is not depth ground truth |
| L3 | Local BA improves selected subset but often harms full-map support | Robust sparse local BA; explicit anchors; reject degrading updates | Reconcile local objective and full-map constraints; do not simply remove guard |
| L4 | Translation jumps and ambiguous CV innovations | Common-map pose comparisons, spatial coverage and ranked image reports | Review weak-support/rotation-disagreement events; test on separate recordings |
| L5 | Sensor calibration/timing uncertainty | RGB checkerboard fit; empirical depth units; gyro magnitude checks | Independent RGB validation; camera–IMU extrinsics/timing; RGB–depth registration |
| L6 | No absolute scale or continuous segment linking | Explicit labels and reset isolation | Depth assistance or properly initialized VIO; relocalization/loop closure later |
| L7 | Unknown generalization | Synthetic geometry tests and one main development walk | Benchmark and unseen real sequences; no route-specific fitting |
| L8 | Forster input mismatch | Gravity-compensated acceleration identified as a limitation | Specific-force-compatible input before faithful phone-data preintegration |

## Defaults and recommended inspected run

CLI defaults remain baseline pose acceptance, baseline keyframe policy, baseline
landmark policy, BA disabled, stride five and seed zero. The inspected run uses
`landmark_policy=refinement` with BA enabled. It is an experiment to inspect,
not a new default or a production recommendation.

Source-space residual summaries can precede BA; selected-window BA costs and
common-map post-BA diagnostics are different quantities. They must not be compared
as if they were a single independent accuracy metric.

## Reference check on the phone capture

ORB-SLAM3 provides an external monocular comparison on the existing walk.
At 640×360 it tracked 55.6% of full-rate frames (longest span 10.93 s), and 40.2%
of stride-five frames (8.49 s). It also loses tracking and creates multiple maps.
Different recovery policies prevent ranking methods by loss counts alone. See
[experiment results](3_experiment_results.md#established-monocular-reference-on-the-walking-capture).
This reinforces inspecting shared failure windows and calibration/image assumptions;
it does not establish the cause or independently validate either trajectory.

Image review on 2026-10-18 found plain-wall views and blurred turns around several
losses. The fixed calibration also gave low residuals on two earlier captures
(median view held-out-corner RMS 0.132/0.240 px), weakening the gross-miscalibration
hypothesis for that setup. Shared screen-target errors and walking camera-mode
changes remain unresolved. Next physical step: a separate printed-target
verification capture, evaluated without refitting; see the
[procedure](5_camera_and_visual_odometry.md#validate-an-existing-fit-without-refitting).

## New textured walks

The earlier textured recording `20261018T125813.647700Z-7b9ace2a`
accumulated 45.1 s receive backlog in its prior audit; its recorded directory is
no longer present in this workspace. The [latest recapture](../data/sessions/walking_textured/20261018T131057.054271Z-8e72c47e/README.md)
contains 69.34 s RGB, 12.29 s backlog growth, no protocol errors, and all 4,035
camera/depth assets verified. It includes IMU and GPS; this was not an RGB-only
throughput isolation experiment. Source timestamps are regular enough for offline VO.

On the recapture, ORB-SLAM3 full rate tracks 94.4% with a longest span of 51.73 s
(one loss, two maps). At stride five it sustains 18.53 s (three losses, four maps).
FusionLab refinement + BA at stride five sustains only 7.00 s with 21 losses and
22 independent segments. See the [comparison](../data/sessions/walking_textured/20261018T131057.054271Z-8e72c47e/analysis/comparisons/monocular/README.md).
The comparison report also links a source-timed ORB camera/3D-path Rerun replay,
with independent map panels and no accuracy claim. This exposes an estimator/frontend
continuity gap even with a common frame selection; the full-rate result also
highlights the effect of temporal subsampling.

User reports starting/ending at the same place. GPS endpoints differ by 9.56 m,
with reported covariance-axis scales about 14.18/4.73 m; this does not verify route
accuracy. Multiple maps/segments and untimed endpoints preclude a valid simple
closure score.

The follow-up FusionLab stride-one run processes all 2,080 frames: 94.0% tracked,
22 losses/23 segments, longest span 17.50 s versus 7.00 s at stride five. Median
segment span falls from 2.17 to 0.67 s; fragmentation remains. Of 22 losses, 18
lack enough visible mapped landmarks; 89.4% of admitted landmark observations
have <2-degree triangulation angle, versus 44.6% at stride five. These diagnostics
motivate inspecting map refresh, landmark baseline and observation management,
not simply loosening gates. The same frame-based BA settings cover less time at
full rate, so this is not a time-normalized optimization ablation. See the
[full-rate experiment](../data/sessions/walking_textured/20261018T131057.054271Z-8e72c47e/analysis/comparisons/monocular/README.md#fusionlab-full-rate-experiment).
No estimator defaults were changed. Independent trajectory accuracy remains unvalidated.

## Next work

**Unfinished (2026-10-24): code cleanup before per-track triangulation.** The user
asked to finish cleaning before more algorithm work. Done and pushed (outputs
byte-identical each step): typed `TrackFrame`/`FrameResult`, VO split into steps,
policy registries, `cli/` package, `sensor_stream/inspection/`, tests in
`unit/`/`integration/`/`support/`, `diagnose-map`, and splitting local BA,
two-view motion, the feature tracker, VO diagnostics and the VO viewer replay.
Group-1 typing pass done 2026-10-24: the six group-1 files plus new
`vision/odometry/pnp.py` and `triangulation.py` pass mypy with
`--disallow-untyped-defs` (project total 84 → 62 errors). Estimator state uses
typed objects (anchor, keyframe, BA window frame, `PnpSnapshot`, triangulation
counts), BA returns a `WindowAdjustment` and raises `DegenerateAnchorBaseline`.
Group 2 done 2026-10-24: `evaluate_run` and `import_sequence` split into typed
steps (`ReferenceTrajectory`, `Similarity`, `CameraModel`, camera-frame and
ground-truth readers); the parity script gained a generated EuRoC sequence
(`scripts/make_synthetic_euroc.py`) covering import → VO → evaluation.
Validation: full suite (181 tests) and `scripts/output_parity.sh` (257 files,
0 differ). Next actions, in order:
1. Ask the user about group 3 (`sensor_stream/source._receive` 249 lines, viewer
   `api.publish`/`rerun_backend.publish`, inspection plots, IMU baselines,
   calibration fitting); it was deferred because acquisition needs a live phone.
2. Then item 2 below: per-track triangulation as an opt-in landmark variant.
3. Decide whether to change the trajectory rotation metric: it inherits a
   position-fitted alignment rotation that is weakly constrained on straight
   segments ([benchmarking](7_benchmarking.md#evaluate)). Candidates: relative
   rotation errors, or fitting the alignment rotation to orientations as well.

**Unfinished (2026-10-18): real EuRoC benchmark run.** EuRoC cam0 import and
timestamp-associated ground-truth evaluation are implemented and tested on
generated fixtures ([workflow](7_benchmarking.md)). The real MH_01_easy run is
blocked on dataset access: the legacy download host timed out and the current ETH
archive returned a rate-limit page. On 2026-10-21 the legacy host still timed out
and the ETH Research Collection (via the dataset DOI) returned HTTP 429 to automated
requests, so `data/benchmarks/euroc/` is still absent. The dataset must be downloaded
in a browser. Next action: obtain the extracted sequence, run the two documented
configurations and record coverage, losses and per-segment aligned errors in the
experiment results.

1. Establish a calibrated ground-truth benchmark and a credible reference method
   before accumulating more phone-specific tweaks.
2. Map maintenance (diagnosed 2026-10-21, [results](3_experiment_results.md#map-maintenance-at-full-rate)):
   losses are mostly map starvation from 1–2-frame triangulation baselines while
   usable parallax goes unused. Next action: add per-track triangulation (each
   track triangulated against its own first observation once its angle passes
   the gate) as an opt-in landmark variant; compare it with baseline and
   `parallax` keyframes at full rate on both walks with `diagnose-map` and the
   ORB timeline. Candidates after that: bootstrap hysteresis for single-frame
   segments in hard windows, and culling/re-refining drifting older landmarks.
   `parallax` keyframes stay a candidate until EuRoC shows accuracy.
3. Validate on a separate recording. Gate any new default on continuity, geometric
   consistency, failures and independent accuracy, not only fewer resets.
4. Proceed to depth assistance or IMU fusion after their geometry/time assumptions
   are validated. They should be separate comparable methods, not hidden patches.

## Roadmap

The sequence below distinguishes implemented experiments from planned methods.
See [1. Project overview](1_project_overview.md) for the current implementation.

| Step | Algorithm | Status / prerequisite |
| --- | --- | --- |
| 1 | Gyroscope orientation integration | Implemented experimental baseline |
| 2 | IMU position dead reckoning | Implemented; substantial drift |
| 3 | Visual feature tracking | Implemented initial Shi–Tomasi/LK method |
| 4 | Monocular visual odometry | **In progress:** fragmented experimental estimator; refinements and diagnostics implemented |
| 5 | Depth-assisted visual odometry | Planned; validate RGB–depth geometry, units and timing first |
| 6a | Error-state EKF | Planned; explicit motion/observation models and independently characterized inputs |
| 6b | SO(3) foundations and IMU preintegration | Planned; can start with synthetic specific-force measurements independently of camera calibration |
| 6c | Factor-graph visual–inertial estimation | Planned; Forster implementation track alongside EKF |
| 7 | Loop closure and pose-graph optimization | Planned |
| 8 | Mapping | Deferred until localization machinery is usable; real captures or real floor-plan imagery |
| 9 | Particle-filter localization | Planned; requires an existing map and observation model |

### Shared experimental interfaces

The eventual algorithm boundary should specify replayable inputs, explicit units,
frames and timestamps; outputs should carry poses, status and uncertainty only
when estimated. Methods must use common evaluation data and explicit alignment.
The standalone viewer API already provides the rendering boundary; a general
interchangeable algorithm/pipeline framework is not implemented yet.

### Forster implementation track

Target: [Forster et al., On-Manifold Preintegration for Real-Time Visual-Inertial
Odometry](https://rpg.ifi.uzh.ch/docs/TRO16_forster.pdf).

1. SO(3) exponential/logarithm maps, perturbations and Jacobians; numerical tests.
2. Preintegrated rotation, velocity and position between keyframes; known-motion tests.
3. Covariance propagation, bias Jacobians and first-order bias correction;
   Monte Carlo and reintegration comparisons.
4. Pose/velocity/bias states, inertial residuals, priors and batch factor-graph tests.
5. Calibrated visual constraints, scale/gravity/velocity/bias initialization and
   structureless visual factors; joint visual–inertial estimation.
6. Incremental smoothing and relinearization; compare against batch estimates,
   benchmark ground truth and later phone captures.

This is one estimator track, not a replacement for EKF or other experiments.
Preintegration does not require completing EKF first. The phone currently supplies
acceleration experimentally supported as gravity compensated; the paper assumes
specific force with gravity handled by the estimator. Resolve that mismatch for
phone data. Synthetic tests can use the correct model immediately.

Camera–IMU geometry/timing, noise conventions and independent validation remain
prerequisites. The existing local visual BA is not the paper's inertial graph.
