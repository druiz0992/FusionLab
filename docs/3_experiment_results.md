# 3. Experiment results

Updated 2026-10-18. Main development sequence: one 85-second walking capture.
The original-walk table uses the same tracks/calibration, stride five and seed zero.
The separate recapture experiments below state their own frame selection.
This is development evidence, not independent validation or trajectory ground truth.

## What was tried

| Variant | Losses | Segments | Longest tracked span | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Baseline monocular VO | 50 | 51 | 7.17 s | Does not reconstruct the walk |
| Refine pose before acceptance | 51 | 52 | 6.67 s | No continuity gain; recount also checks positive depth |
| Parallax/track-survival keyframes | 50 | 51 | 8.00 s | Stronger admitted angles, but no reduction in losses |
| Third-view validation only | 51 | 52 | 8.00 s | No continuity gain alone |
| Landmark-only multiview refinement | 30 | 31 | 8.50 s | Main observed continuity gain; jumps remain |
| Validation plus landmark refinement | 30 | 31 | 8.82 s | Similar loss count; larger step-ratio flags |
| Landmark refinement plus local BA | 29 | 30 | 10.83 s | Modest additional gain; still fragmented |

**Conclusion:** point-coordinate refinement provided most of the gain here.
Neither keyframe scheduling nor pose-acceptance order solved fragmentation.
No experimental variant has been promoted to CLI default.

## What the diagnostics say

- Baseline losses: 33 low PnP-inlier-ratio failures, seven insufficient-inlier-count
  failures and ten insufficient-visible-landmark failures. These are failed gates,
  not established root causes.
- 73% of baseline admitted landmarks had under 2° triangulation angle. The keyframe
  experiment lowered this to 56%, but survival-triggered refreshes still dominated.
- Local BA accepted 35/130 attempts; 94 were rejected for worsening current-map
  support despite improving selected-window cost. Keep that guard.
- Latest BA diagnostics reproduced all 513 saved states. CV was available at 415
  frames and gyro integration at 435 intervals under clock-alignment assumptions.
- CV did not satisfy the geometric gates at any of 23 tracking losses where it
  was available. Visual fit beating CV is partly expected: the map was visual.
- Median visual/gyro rotation-magnitude disagreement was 0.47°. This does not
  establish axis alignment or prove the visual trajectory correct.
- Large normalized translation innovations near stops can coexist with broad,
  strong visual support. Do not turn a ratio into an automatic rejection gate.

## Established monocular reference on the walking capture

ORB-SLAM3 was run on 2026-10-18 in monocular full-SLAM mode, using the same
provisional calibration and 640×360 unrotated images. No IMU/depth/GPS was used.

| Input | Tracked frames | Longest uninterrupted span | Tracking-to-nontracking events | Observed maps |
| --- | ---: | ---: | ---: | ---: |
| Every RGB frame (~30 FPS) | 1426/2565 (55.6%) | 10.93 s | 7 | 8 |
| Every fifth frame (~6 FPS) | 206/513 (40.2%) | 8.49 s | 6 | 5 |

Both runs completed all selected frames. ORB also loses tracking; fewer loss
events do not establish better continuity because its recovery/initialization
policy differs. FusionLab's inspected BA run tracks 446/513 frames (86.9%) but
restarts 29 times. Neither result establishes trajectory accuracy. Common failure
periods warrant image/calibration inspection; the cause is not established.

See the [comparison and tracking timeline](../data/sessions/walking/20261017T183215.982977Z-45b7f9c2/analysis/comparisons/orbslam3/README.md)
and [reproducible external driver](../benchmarks/orbslam3/README.md). These are
single runs with relocalization, loop closure and multiple maps enabled in ORB;
they do not replace calibrated ground-truth benchmark validation.

## Textured recapture with reported return to start

Capture `20261018T131057.054271Z-8e72c47e`, evaluated 2026-10-18 with the same
calibration and 640×360 estimation pixels:

| Method | Tracked frames | Losses | Longest span |
| --- | ---: | ---: | ---: |
| ORB-SLAM3, all RGB frames | 1963/2080 (94.4%) | 1 | 51.73 s |
| ORB-SLAM3, stride 5 | 331/416 (79.6%) | 3 | 18.53 s |
| FusionLab refinement + BA, stride 5/seed 0 | 364/416 (87.5%) | 21 | 7.00 s |
| FusionLab refinement + BA, stride 1/seed 0 | 1956/2080 (94.0%) | 22 | 17.50 s |

Native loss/recovery semantics differ. The matched-stride ORB run has longer
continuous spans despite lower frame coverage; rapid reinitialization is not
continuous localization. The stronger full-rate result indicates that frame
subsampling also matters. Single runs do not establish repeatability or accuracy.
GPS is not precise enough here to validate the user-reported closed route, and
all runs remain split across independent maps/segments. [Report and timeline](../data/sessions/walking_textured/20261018T131057.054271Z-8e72c47e/analysis/comparisons/monocular/README.md).

The full-rate follow-up improves frame coverage and longest span but leaves
22 losses versus 21. Median tracked-segment span decreases from 2.17 to 0.67 s.
18/22 losses are insufficient visible mapped landmarks; the full-rate audit has
89.4% of admitted landmark observations below 2 degrees parallax, versus 44.6%
at stride five. BA accepts 450/638 attempts versus 12/108; its five-frame window
and every-third-frame schedule have different physical time horizons at full rate.
This motivates a focused landmark/map-maintenance investigation without claiming
an isolated cause or better independent accuracy. [Full-rate results and timeline](../data/sessions/walking_textured/20261018T131057.054271Z-8e72c47e/analysis/comparisons/monocular/README.md#fusionlab-full-rate-experiment).

## Where to inspect the evidence

Subsequent inspection found nearly featureless walls at several ORB loss windows
and visible blur during others. Fixed selected intrinsics were also checked on
35/28 accepted views from two earlier captures: median view held-out-corner RMS
was 0.132/0.240 px in original pixels. This is evidence against a gross mismatch
on that calibration setup, not independent validation of the walking camera mode.
The captures share a screen target; a fresh printed-target check remains pending.
See the [calibration and image review](../data/sessions/walking/20261017T183215.982977Z-45b7f9c2/analysis/comparisons/calibration_review/README.md).

Use the walking capture's [analysis index](../data/sessions/walking/20261017T183215.982977Z-45b7f9c2/README.md).
Canonical current outputs are in `analysis/current/`:

| Output | Role |
| --- | --- |
| `tracks/` | Rotated feature overlays and unrotated image-coordinate tracks |
| `baseline/` | Original estimator comparison |
| `landmark_refinement/` | Main improvement without BA |
| `local_ba/` | Latest joint-refinement run and replay |
| `diagnostics/` | Ranked events, plots, images and gyro/CV comparisons |
| `imu_baselines/` | Orientation and position experiments |

`analysis/comparisons/` retains machine-readable A/B tables. Superseded exports and
other variants are in `analysis/archive/`. The capture's `analysis/README.md`
explains retained folders; the consolidation manifest maps old output paths to
their canonical locations. Do not read every intermediate `poses.jsonl` to
understand status.

## Measurement limits

The user supplied a 25.15 m route intended to return to its start. It has no
per-leg timestamps or camera-frame alignment and is not a pointwise reference.
Reported distances have an 18 cm closure discrepancy and were preserved unchanged.
Independent segment scales prohibit a full-walk metric drift score.

IMU orientation/phone agreement is about 0.63° RMS. Uncorrected IMU integration
produced about 194.3 m path length and 190.4 m horizontal endpoint displacement.
These are separate baselines, not fused inputs or independent ground truth.
