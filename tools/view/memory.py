"""View 5: allocated and reserved over time, with the peak and the headroom.

Nothing else says whether a fix that spends memory is available at all.
"""

from __future__ import annotations

import sqlite3

from .common import table


def render(con: sqlite3.Connection, capacity_gb: float | None = None) -> str:
    rows = con.execute(
        "SELECT rank, device, MAX(allocated), MAX(reserved), COUNT(*) FROM mem "
        "GROUP BY rank, device ORDER BY rank, device").fetchall()
    out = ["== 5. Memory =="]
    if not rows:
        # torch.profiler only emits [memory] instants with profile_memory=True,
        # and most captures do not set it. The unprofiled run of ① recorded peak
        # memory anyway, so the numbers exist even when this table is empty.
        return "\n".join(out + [
            "  (the trace carries no [memory] instants: capture with "
            "profile_memory=True, or read the peaks the unprofiled run of ① saved "
            "under commits/<sha>/)"])

    def gb(b):
        return f"{b / 2**30:.1f}" if b else "-"

    body = [[f"r{r}", "cpu" if d < 0 else f"cuda:{d}", gb(a), gb(res), n,
             f"{capacity_gb - res / 2**30:.1f}" if capacity_gb and d >= 0 else "?"]
            for r, d, a, res, n in rows]
    out.append(table(body, ["rank", "dev", "peak_alloc_GB", "peak_reserved_GB", "samples",
                            "headroom_GB"]))
    if capacity_gb is None:
        out.append("  headroom unknown: pass --capacity-gb "
                   "(torch.cuda.get_device_properties(d).total_memory / 2**30) to judge "
                   "whether the batch can grow")
    return "\n".join(out)
