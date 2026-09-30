"""The five views: projections of one store built from an exported trace.

Read them in order. View 1 is the time spine and answers "waiting or busy but
inefficient" before any of the others are worth opening.

Finish all five before concluding. They rank by different things on purpose:
view 2 by recoverable time, view 3 by kernel time, view 4 by exposure. A 3 ms
stall on every step and an 8 ms hiccup every twentieth step are not comparable
until both have been measured.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from . import kernels, lost_time, memory, overlap, runtime

# Numbered so a directory listing keeps reading order.
FILES = (
    ("1_runtime_decomposition.txt", "the time spine, wall = busy + idle per step"),
    ("2_lost_time.txt", "where time is lost, ranked, per rank"),
    ("3_kernel_table.txt", "cumulative GPU time by kernel and by class"),
    ("4_overlap_exposure.txt", "per collective, hidden against exposed"),
    ("5_memory.txt", "allocated, reserved, peak and headroom per GPU"),
)


def render_each(con: sqlite3.Connection, capacity_gb: float | None = None) -> dict[str, str]:
    bodies = (runtime.render(con), lost_time.render(con), kernels.render(con),
              overlap.render(con), memory.render(con, capacity_gb))
    return {name: body for (name, _), body in zip(FILES, bodies)}


def write(con: sqlite3.Connection, out_dir: Path,
          capacity_gb: float | None = None) -> Path:
    """One file per view, so a reader opens only the one it needs and a single
    view can be diffed across candidates."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, body in render_each(con, capacity_gb).items():
        (out_dir / name).write_text(body + "\n")
    return out_dir
