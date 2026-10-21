# FusionLab working rules

These rules apply to every session. They are standards, not project facts:
current results, designs and priorities live in the numbered documents in
[docs/](docs/). Change this file only when a new working agreement is
explicitly made with the user.

## Start a session

1. Read this file, then [project overview](docs/1_project_overview.md) and
   [status and roadmap](docs/2_status_and_roadmap.md).
2. For algorithm work, also read [experiment results](docs/3_experiment_results.md)
   so rejected approaches are not repeated.
3. Read the topic document (4–7) for the component being changed.
4. Follow the user's current task. The roadmap is context, not authorization
   to start unrelated work.

Inspect only relevant source and data. Do not recursively read raw recordings
or historical analysis outputs to get oriented; use [data/README.md](data/README.md).

## Documentation structure

| File | Contents |
| --- | --- |
| `AGENTS.md` | Rules to follow in every session (this file) |
| `README.md` | How to install, run every command and its options, and run tests |
| `docs/1_…` to `docs/7_…` | Everything else, numbered in reading order |
| `data/**/README.md`, `benchmarks/*/README.md` | Per-capture, per-run and per-tool reports kept beside their data/code |

- Keep each fact in one place and link to it; do not duplicate numbers or
  commands across documents.
- Update an existing section instead of appending chat transcripts, parallel
  plans or new summary documents. Add a new numbered document only for a new
  major topic, and keep the numbering in reading order.
- Documentation must be current: update it in the same change as the code,
  data or decision it describes. When docs disagree with code or artifacts,
  verify which is right and correct the wrong one; report unresolved doubt.
- Where things go: new commands/options → `README.md`; architecture → `1_`;
  capabilities, limitations, priorities and unfinished work → `2_`; measured
  outcomes and retained/rejected variants → `3_`; component behavior → `4_`–`7_`;
  selected data/results → `data/README.md` and the capture guides.
- For unfinished work, leave a short dated note in `2_status_and_roadmap.md`
  with what changed, validation, the remaining blocker and the next concrete
  action; replace it when resolved.

## Code standards

- **Typed Python 3.12.** Annotate all function signatures and public attributes.
  Prefer dataclasses (frozen where practical) and enums over loose dicts and strings.
- **Clear, well-commented code.** This is an educational project: comments must
  explain the algorithm, its assumptions, units, frames and why a step exists,
  not restate the syntax. Give public functions and classes docstrings.
- **Caveats in outputs, conclusions in docs.** Saved reports may carry method
  caveats (what a number does and does not measure) as named module constants.
  Never hard-code findings, recommendations or next steps in generated output;
  they go stale and repeat for runs they do not fit. Put them in
  `docs/2_status_and_roadmap.md` or `docs/3_experiment_results.md`.
- **Short, focused functions.** One responsibility per function; split long
  functions into named steps. One statement per line. Descriptive names that
  carry units or frames where ambiguous (`dt_seconds`, `R_camera_world`).
- **Extensible design.** New algorithms or variants should plug in alongside
  existing ones through small explicit interfaces or policy options, without
  editing unrelated code. Favor composition and readable modules over a new
  framework; do not build speculative abstractions.
- **Package boundaries.** Keep acquisition (`sensor_stream`), processing and
  estimation (`fusionlab`) and visualization (`fusion_viewer`) separate. The
  viewer must not import the other two. Within `fusionlab`, use the grouped
  modules (`recordings`, `inertial`, `vision`, `vision/odometry`, `evaluation`,
  `visualization`); keep CLI entry points at the package root and thin.
  Diagnostic evaluation must not feed estimator decisions.
- Keep optional heavy dependencies (OpenCV, SciPy, Rerun, Matplotlib) lazily
  imported where they currently are.
- Do not edit generated Protobuf by hand; regenerate it (see
  [acquisition](docs/4_sensors_and_acquisition.md#wire-protocol)).
- Format with black before finishing.

## Testing

- Every code change needs tests. Write **unit tests** for individual functions
  and classes (known inputs, edge cases, error paths) and **integration tests**
  for end-to-end paths (CLI commands, capture → tracks → VO → evaluation on small
  generated fixtures).
- Algorithm changes need tests with known geometry or known ground truth, and a
  regression test for each fixed bug.
- Layout: `tests/unit/<package>/...` mirrors `src/` (one component, in process,
  generated inputs; temp files are fine). `tests/integration/` holds tests that
  cross components or packages: network/USB sources, record → inspect, CLI
  commands doing real work, multi-stage pipelines and fusionlab → fusion_viewer
  exports. Name files `test_*.py`.
- Reuse fixtures from `tests/support/` (`wire`, `imu`, `vision`, `require`)
  instead of copying builders between test files; never import one test module
  from another. Optional dependencies go through `require(...)`, which skips
  cleanly when they are missing.
- Run the relevant tests for every change, and the full suite for shared
  behavior. Docs-only changes need link/command checks, not estimator tests.
  Run the data audit after any data layout change.
- Report what ran, what failed and which optional checks were skipped. Passing
  synthetic tests does not establish real-world localization accuracy.

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m black --workers 1 src tests examples benchmarks
python3 scripts/restore_data_layout.py --check
```

## Scientific conventions

- Make units, coordinate frames, timestamp assumptions and calibration explicit.
  Preserve wire measurements; apply interpretations in processing.
- World-to-camera poses use `X_camera = R X_world + t`; camera center `-R.T t`.
  Camera axes: +X right, +Y down, +Z forward in unrotated image coordinates.
- Display rotation must never change tracking or estimator coordinates.
- Monocular segments have independent arbitrary scale. Do not join them or label
  them as metres without an established scale/alignment method.
- Do not silently assume identity transforms or synchronized exposure. Camera–IMU
  timing/extrinsics and RGB–depth registration are not established.
- IMU baselines and gyro diagnostics are not sensor fusion; do not describe them so.
- Compare configurations on common inputs. Record failures and regressions too.
  Fewer resets or lower reprojection error alone do not establish better accuracy;
  distinguish development data from validation data.
- Keep experimental policies explicit. Do not silently promote a selected run to
  a CLI default or remove acceptance guards to improve headline counts.

## Data and experiment outputs

- Preserve raw `camera/`, `depth/`, `events.jsonl`, `metadata.json`, human route
  annotations and calibration evidence. Never delete recordings or results as
  incidental cleanup.
- Within a capture: `analysis/current/` holds selected outputs, `analysis/comparisons/`
  cross-run comparisons, `analysis/archive/` historical runs, and new experiments
  go in a new `analysis/runs/<unique-name>/`. Never overwrite results.
- Keep configuration, source provenance and replay outputs with each run.
- Read the capture's README and `analysis/README.md` before changing its layout.
  When moving data, update references and `data/consolidation_manifest.json` so
  the audit/restore tool keeps working. Never run `--restore` just to validate.

## Finish a session

End with a brief account of what changed, what was validated (with commands and
results), and what remains unfinished.
