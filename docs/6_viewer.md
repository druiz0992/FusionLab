# 6. Standalone viewer

Updated 2026-10-21 from the 2026-10-18 implementation reference. `fusion_viewer`
is independent of the acquisition and algorithm packages. Commands are in the
[README](../README.md#viewer-fusion-viewer-and-examples).

## Implemented boundary

`ViewerClient(sink)` is synchronous. Producers register a `RunSpec` and spatial
`FrameSpec`, publish typed updates, and flush/close; context management is supported.
Publish validates and submits to the sink, not necessarily to the screen.
There is no remote connection protocol or background-queue guarantee.

| Contract | Meaning |
| --- | --- |
| `RunSpec` | Stable run ID and label |
| `FrameSpec` | Named axes and `m` or `arbitrary` units |
| `SampleTime` | Segment ID, clock ID, signed nanosecond ticks |
| `ScalarSample` | Finite scalar and unit label |
| `OrientationSample` | Unit Hamilton XYZW quaternion, frame and role |
| `ImageSample` | Encoded image and dimensions |
| `DepthSample` | Explicit encoding, byte order and optional metric scale |
| `Map2D` | Wall-segment snapshot |
| `Pose2D` | XY, heading and role |
| `Trajectory2D` | Full path snapshot |
| `Particles2D` | Pose population and nonnegative weights |
| `SegmentBoundary` | Explicit discontinuity |

An entity cannot change type or spatial frame. No transform/alignment is inferred.
Heading is radians counterclockwise from +X about +Z. The immutable contracts and
validation live in `models.py` and `api.py`.

## Persistence and rendering

- `LogSink`: new-only portable directory, manifest, typed JSONL and SHA-256 binary
  assets. `read_log` reconstructs types and checks asset containment and hashes.
- `RerunSink`: optional pinned Rerun SDK 0.39.0 backend. Logs encoded RGB, depth,
  orientation, 2D geometry and scalars. The `odometry` layout has camera/status,
  one active-segment path and diagnostic plots.
- `fusion-viewer open`: opens portable logs or exports `.rrd` without a GUI.

New segments clear prior display geometry. Gaps over 0.5 s clear prior entity
state before the next update; trailing values can persist. Static map semantics,
exact staleness guarantees and robust crash recovery are not promised. Flush does
not guarantee fsync durability. Errors can leave partial output folders.

The VO active frame is a display container for the currently selected playback
segment. Clearing and restarting it does not align independent physical segments.
The XZ projection is not a gravity-aligned floor plan. Seek the timeline to inspect
another segment; current replay shows only the active segment.

## Examples

`examples/viewer_producer.py` writes scalar and orientation samples to `demo.fvl`.
`examples/localization_demo.py` writes a room, poses, trajectory and particles.
These are generated illustrations; the example does not run a localization
algorithm.

## Not implemented

Map editing, live remote clients, generic clock registration/alignment, batch
transport protocols, calibrated camera frusta and depth point-cloud rendering,
3D maps, and automatic estimator uncertainty are not implemented. Rendering
synthetic particles does not mean a particle filter exists.
