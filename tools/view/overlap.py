"""View 4: per collective, how much hid behind compute and how much was exposed.

Only the exposed part lengthens the step, so a large communication share that is
fully hidden is not a finding. Two traps: hidden is a timeline property and not a
cost model, and a large collective share on ONE rank is usually straggler wait,
so read view 1's per-rank split before diagnosing communication here.
"""

from __future__ import annotations

import sqlite3

from .common import clip, fmt_bw, fmt_ms, fmt_size, issuer_col, table, union

ISSUER_WIDTH = 72
NAME_WIDTH = 44


def render(con: sqlite3.Connection, top: int = 10) -> str:
    out = ["== 4. Overlap and exposure =="]
    ranks = [r for (r,) in con.execute("SELECT DISTINCT rank FROM events WHERE cat='comm'")]
    if not ranks:
        return "\n".join(out + ["  (no communication kernels)"])

    issuer = issuer_col(con)
    agg: dict[str, list[int]] = {}
    issued_by: dict[str, dict[str, int]] = {}
    for rank in ranks:
        comms = con.execute(
            f"SELECT e.name, e.ts_ns, e.dur_ns, COALESCE(e.nbytes,0), {issuer} FROM events e "
            "LEFT JOIN kernel_op k ON k.rank = e.rank AND k.corr = e.corr "
            "WHERE e.rank=? AND e.cat='comm' ORDER BY e.ts_ns", (rank,)).fetchall()
        for name, ts, dur, nbytes, who in comms:
            busy = con.execute(
                "SELECT ts_ns, dur_ns FROM events WHERE rank=? AND cat='kernel' "
                "AND ts_ns < ? AND ts_ns + dur_ns > ?", (rank, ts + dur, ts)).fetchall()
            hidden = union([(max(s, ts), min(s + d, ts + dur)) for s, d in busy
                            if min(s + d, ts + dur) > max(s, ts)])
            key = name.split("(")[0][:NAME_WIDTH]
            a = agg.setdefault(key, [0, 0, 0, 0])
            a[0] += dur
            a[1] += hidden
            a[2] += 1
            a[3] += nbytes
            if who:
                by = issued_by.setdefault(key, {})
                by[who] = by.get(who, 0) + dur

    rows = []
    for n, (tot, h, c, b) in sorted(agg.items(), key=lambda x: -(x[1][0] - x[1][1]))[:top]:
        who = max(issued_by.get(n, {}).items(), key=lambda kv: kv[1], default=("-", 0))[0]
        rows.append([n, c, fmt_ms(tot), fmt_ms(h), fmt_ms(tot - h),
                     f"{1 - h / max(tot, 1):.0%}", fmt_size(b), fmt_bw(b, tot),
                     clip(who, ISSUER_WIDTH)])
    # algbw is size over time, the figure NCCL's own benchmarks report. Turning it
    # into busbw needs the collective's algorithm factor, 2(n-1)/n for all-reduce
    # and (n-1)/n for all-gather, which depends on the group size. It is printed
    # here so the reader can do that against the interconnect they actually have.
    # Without a size column this table said "152 ms" and left no way to tell a
    # healthy collective from a sick one.
    out.append(table(rows, ["collective", "n", "total_ms", "hidden_ms", "exposed_ms",
                            "exposed%", "size", "algbw GB/s", "issued by"]))
    tot = sum(v[0] for v in agg.values())
    hid = sum(v[1] for v in agg.values())
    out.append(f"  comm total {fmt_ms(tot)}ms, exposed {fmt_ms(tot - hid)}ms "
               f"({1 - hid / max(tot, 1):.0%}). Only the exposed part lengthens the step.")
    return "\n".join(out)
