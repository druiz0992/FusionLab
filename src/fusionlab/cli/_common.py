"""Helpers shared by command handlers: JSON reports and viewer sinks."""

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any


def print_json(report: dict[str, Any], *, omit: Iterable[str] = ()) -> None:
    """Print a report as indented JSON, leaving out bulky per-item keys."""
    excluded = set(omit)
    print(json.dumps({k: v for k, v in report.items() if k not in excluded}, indent=2))


def rerun_viewer_factory(
    rrd: Path | None, spawn: bool, *, layout_profile: str | None = None
) -> Callable[[], Any] | None:
    """Return a factory for a Rerun-backed viewer, or None if neither is wanted.

    Used by long-running commands that open the viewer only once their output
    directory is reserved. `spawn` opens the GUI; `rrd` saves a recording.
    """
    if not (spawn or rrd):
        return None
    from fusion_viewer import ViewerClient
    from fusion_viewer.rerun_backend import RerunSink

    options = {} if layout_profile is None else {"layout_profile": layout_profile}

    def factory():
        return ViewerClient(RerunSink(output=rrd, spawn=spawn, **options))

    return factory


def open_viewer_sink(export_log: Path | None, rrd: Path | None, spawn: bool):
    """Choose a portable log directory or a Rerun sink; never both."""
    if export_log and (rrd or spawn):
        raise ValueError("Choose portable export or Rerun output")
    if export_log:
        from fusion_viewer.recording import LogSink

        return LogSink(export_log)
    from fusion_viewer.rerun_backend import RerunSink

    return RerunSink(output=rrd, spawn=spawn)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write one JSON object per line, creating the parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")
