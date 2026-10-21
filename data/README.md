# Data index

Consolidated 2026-10-18. There are three active input datasets; other recordings are
historical characterization evidence. No raw recordings or results were deleted.

## Active inputs

| Dataset | Canonical location | Role |
| --- | --- | --- |
| Textured walking | [recaptures and backlog audits](sessions/walking_textured/README.md) | New sequences, latest includes GPS and a reported return to start |
| Walking | [capture index](sessions/walking/20261017T183215.982977Z-45b7f9c2/README.md) | Main development sequence; not an independent benchmark |
| RGB calibration | [selected fit](sessions/rgb_calibration/20261018T071611.049465Z-3cb5f83c/calibration/calibration.json) | Provisional 1920×1080 fit with 20 mm screen squares |

The selected calibration's raw images and review notes remain beside its fit.
Its camera–IMU/depth/timing calibration is not established. All 39 capture records,
including empty or superseded attempts, are listed in [catalog.json](catalog.json).

## Walking analysis structure

```text
walking/<capture>/
  metadata.json, events.jsonl, camera/, depth/   raw recording
  notes.md, route.json                         human route evidence
  README.md                                   capture entry point
  analysis/
    current/
      tracks/                                 rotated overlays + pixel tracks
      baseline/                               original monocular baseline
      landmark_refinement/                    strongest simple comparison
      local_ba/                               latest inspected VO + replay
      diagnostics/                            ranked CV/gyro/spatial report
      imu_baselines/                          separate IMU experiments
    comparisons/                              full A/B tables, JSON and Markdown
    archive/                                  intermediate runs, audits, exports
    runs/                                     suggested location for new runs
```

Use `analysis/current/diagnostics/index.html` in a browser and
`analysis/current/local_ba/replay.rrd` in Rerun. Current means “selected for
inspection,” not validated localization or a changed CLI default.

## Archive and compatibility policy

Historical sensor captures were moved to `data/archive/sensor_validation/`.
Superseded calibration captures are under `data/archive/calibration/`.
Archived captures appear only in `archive/`; their 21 former shortcuts under
`sessions/` were removed on 2026-10-18. `sessions/` initially contained walking and
the selected RGB calibration; the new `walking_textured/` experiment is now also present. These shortcuts were not duplicate recordings,
so removing them does not reclaim recording storage.

For historical commands, use `catalog.json` to map `original_path` to
`canonical_path`. The 35 old walking output shortcuts were also removed; use
the walking capture's README and `analysis/README.md` for canonical result paths.
The consolidation manifest maps old output names to their retained destinations.
Saved reports may still contain historical paths.

The raw files themselves were not edited or re-encoded. Archiving reorganizes
storage; removing shortcuts does **not** free recording or result storage.
Raw evidence and generated data remain excluded from Git.

New captures stay under `data/sessions/<experiment>/`. New derived runs should use
an explicit `analysis/runs/<new-name>` directory. Do not overwrite selected
current results silently; update this index and the results document when selecting
a replacement. Keep calibration, raw recording and analysis provenance together.

## Audit and undo

External benchmark inputs and outputs belong in `data/benchmarks/`, separately
from phone sessions and archives. They are excluded from Git. The
[EuRoC guide](../docs/7_benchmarking.md) explains import and evaluation;
no real benchmark sequence has been imported yet.

`consolidation_manifest.json` records every move and all raw-file verification
metadata. It is a local generated audit file, excluded from Git. If it is missing,
do not attempt automatic rollback.

```bash
# Read-only check of data moves and remaining compatibility aliases:
python3 scripts/restore_data_layout.py --check
# Optional: undo the directory moves (does not restore old documentation):
python3 scripts/restore_data_layout.py --restore
```

The restore command checks all paths first and refuses conflicting changes.
The earlier `docs/archive/` documentation is not present in this workspace; current documentation is in [docs/](../docs/).
