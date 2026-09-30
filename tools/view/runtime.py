"""View 1: runtime decomposition, the time spine.

`wall = busy + idle` is the first fork of the whole diagnosis. It separates a job
that is waiting from a job that is busy but inefficient, and nothing on the busy
side is worth deriving until this is on the page.
"""

from __future__ import annotations

import sqlite3
import statistics as st

from .common import GPU_CATS, MS, busy_intervals, fmt_ms, table

SPREAD_FLAG = 0.10      # step-time spread worth calling an outlier
STRAGGLER_FLAG = 0.05   # rank-to-rank gap worth naming a straggler


def render(con: sqlite3.Connection) -> str:
    steps = con.execute("SELECT rank, idx, ts_ns, dur_ns FROM steps ORDER BY rank, idx").fetchall()
    out = ["== 1. Runtime decomposition =="]
    if not steps:
        return "\n".join(out + _no_steps(con))

    rows, per_rank_ms = [], {}
    for rank, idx, ts, dur in steps:
        busy = sum(e - s for s, e in busy_intervals(con, rank, ts, ts + dur))
        rows.append([f"r{rank}", idx, fmt_ms(dur), fmt_ms(busy), f"{busy / max(dur, 1):.0%}",
                     fmt_ms(dur - busy), f"{1 - busy / max(dur, 1):.0%}"])
        per_rank_ms.setdefault(rank, []).append(dur / MS)
    out.append(table(rows, ["rank", "step", "wall_ms", "busy_ms", "busy%", "idle_ms", "idle%"]))

    for rank, ds in sorted(per_rank_ms.items()):
        med = st.median(ds)
        spread = (max(ds) - min(ds)) / med if med else 0
        flag = "  <- OUTLIER SPREAD" if spread > SPREAD_FLAG else ""
        out.append(f"  r{rank}: median {med:.1f}ms  spread {spread:.1%}{flag}")

    if len(per_rank_ms) > 1:
        meds = {r: st.median(d) for r, d in per_rank_ms.items()}
        slow, fast = max(meds, key=meds.get), min(meds, key=meds.get)
        if meds[fast] and (meds[slow] - meds[fast]) / meds[fast] > STRAGGLER_FLAG:
            out.append(f"  straggler: r{slow} {meds[slow]:.1f}ms vs r{fast} {meds[fast]:.1f}ms "
                       f"(+{meds[slow] / meds[fast] - 1:.1%})")
    return "\n".join(out)


def _no_steps(con: sqlite3.Connection) -> list[str]:
    """Fall back to each rank's GPU event window when no steps are annotated."""
    spans = con.execute(f"SELECT rank, MIN(ts_ns), MAX(ts_ns+dur_ns) FROM events "
                        f"WHERE cat IN {GPU_CATS} GROUP BY rank ORDER BY rank").fetchall()
    if not spans:
        return ["  (no GPU events)"]
    out = []
    for rank, t0, t1 in spans:
        busy = sum(e - s for s, e in busy_intervals(con, rank, t0, t1))
        wall = t1 - t0
        busy_fraction = busy / wall if wall else 0
        out.append(f"  r{rank}: no step markers; window {fmt_ms(wall)}ms  busy {fmt_ms(busy)}ms "
                   f"({busy_fraction:.1%})  idle {fmt_ms(wall - busy)}ms ({1 - busy_fraction:.1%})")
    return out + ["  (capture with a step annotation, such as torch.profiler's ProfilerStep, "
                  "to get the per-step spine)"]
