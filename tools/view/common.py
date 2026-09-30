"""Formatting and interval helpers the five views share.

Every view is a projection of one SQLite store built from the exported trace:

    events(rank, device, stream, tid, cat, name, ts_ns, dur_ns, corr, nbytes)
        cat is one of kernel, comm, memcpy, memset, cpu_op, runtime, annotation
    steps(rank, idx, name, ts_ns, dur_ns)
    mem(rank, device, ts_ns, allocated, reserved)
    kernel_op(rank, corr, op, stack)   a kernel resolved back to the op that
                                       launched it, which is what makes an
                                       anchor possible
"""

from __future__ import annotations

import bisect
import re
import sqlite3

US = 1_000
MS = 1_000_000

_CATS = (
    ("communication", r"nccl|all_?reduce|all_?gather|reduce_?scatter|broadcast|c10d"),
    ("attention", r"flash|fmha|attention|sdpa|scaled_dot_product|mem_efficient"),
    ("GEMM", r"cutlass|nvjet|gemm|cublas|xmma|wgrad|dgrad|scaled_mm|ampere_|sm\d0_|conv"),
    ("element-wise", r"elementwise|vectorized|pointwise|cudafunctor|gpu_kernel|triton_poi|"
                     r"norm|softmax|reduce|copy|cast|fill|silu|gelu|add|mul"),
)

GPU_CATS = "('kernel','comm','memcpy','memset')"


def classify(name: str) -> str:
    for cat, pat in _CATS:
        if re.search(pat, name, re.I):
            return cat
    return "other"


def fmt_ms(ns: float) -> str:
    ms = ns / MS
    return f"{ms:.1f}" if ms >= 10 else f"{ms:.2f}" if ms >= 1 else f"{ms:.3f}"


def fmt_dur(ns: float) -> str:
    """A duration at a unit that survives rounding.

    `{ns/MS:.0f}ms` printed `0ms` for everything under half a millisecond, which
    is most causes in most lesions, so the column meant to say how much of a hole
    an op accounts for said "none" for all of them.
    """
    ms = ns / MS
    return f"{ms:.0f}ms" if ms >= 10 else f"{ms:.1f}ms" if ms >= 1 else f"{ns / US:.0f}us"


def fmt_size(nbytes: float) -> str:
    """Bytes at a readable scale, or `-` when the event carries no size.

    A fixed GB column printed `0.00` for everything under 10 MB, which is most
    transfers in a short capture, and reads as "no data" rather than "small".
    """
    if not nbytes:
        return "-"
    for unit, div in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if nbytes >= div:
            return f"{nbytes / div:.2f} {unit}"
    return f"{int(nbytes)} B"


def fmt_bw(nbytes: float, ns: float) -> str:
    """GB/s, the number that says whether a transfer or a collective was slow.

    A duration alone cannot: 152 ms is fast for 3 GB and terrible for 30 MB.
    """
    return "-" if not nbytes or not ns else f"{nbytes / (ns / 1e9) / 1e9:.1f}"


def clip(s: str, limit: int) -> str:
    """Clip an attribution string at a separator, never mid-name.

    A fixed slice cut through whatever token straddled it, so a row could end in
    `aten::_effici`, which reads as an op name and is not one.
    """
    if len(s) <= limit:
        return s
    cut = max(s.rfind(" · ", 0, limit), s.rfind(" > ", 0, limit))
    return (s[:cut] if cut > 0 else s[:limit].rstrip()) + " …"


def table(rows: list[list], header: list[str]) -> str:
    if not rows:
        return "  (empty)"
    w = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    def line(r):
        return "  " + "  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)).rstrip()
    return "\n".join([line(header), "  " + "  ".join("-" * x for x in w)] + [line(r) for r in rows])


def busy_intervals(con: sqlite3.Connection, rank: int, t0: int, t1: int) -> list[tuple[int, int]]:
    """Merged GPU-busy intervals in [t0, t1): kernels, comm, memcpy, memset.

    Uses the crossing-safe window predicate, `start < t1 AND end > t0`. The
    intuitive `start >= t0 AND end <= t1` silently drops events that straddle the
    boundary, which are usually the interesting ones.
    """
    rows = con.execute(
        f"SELECT ts_ns, ts_ns + dur_ns FROM events WHERE rank=? AND cat IN {GPU_CATS} "
        "AND ts_ns < ? AND ts_ns + dur_ns > ? ORDER BY ts_ns", (rank, t1, t0)).fetchall()
    merged: list[list[int]] = []
    for s, e in rows:
        s, e = max(s, t0), min(e, t1)
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(a, b) for a, b in merged]


def union(spans: list[tuple[int, int]]) -> int:
    """Total length covered by a set of possibly overlapping intervals."""
    merged: list[list[int]] = []
    for a, b in sorted(spans):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return sum(b - a for a, b in merged)


class Intervals:
    """Overlap queries over one rank's events, without a table scan per query.

    SQL cannot use an index for `ts_ns < t1 AND ts_ns + dur_ns > t0`, so asking
    per lesion was 3k lesions against 3.3M rows. `max_end[i]` is the largest end
    among intervals 0..i, which lets a backward scan stop as soon as no earlier
    interval can still reach into the window.
    """

    def __init__(self, rows: list[tuple]) -> None:
        rows.sort(key=lambda r: r[0])
        self.rows = rows
        self.start = [r[0] for r in rows]
        self.max_end: list[int] = []
        best = -1
        for r in rows:
            best = max(best, r[1])
            self.max_end.append(best)

    def overlapping(self, t0: int, t1: int) -> list[tuple]:
        out = []
        i = bisect.bisect_left(self.start, t1) - 1
        while i >= 0 and self.max_end[i] > t0:
            if self.rows[i][1] > t0:
                out.append(self.rows[i])
            i -= 1
        return out


def has_stack(con: sqlite3.Connection) -> bool:
    return any(r[1] == "stack" for r in con.execute("PRAGMA table_info(kernel_op)"))


def issuer_col(con: sqlite3.Connection) -> str:
    """The column naming who launched a kernel, falling back when no stack was captured."""
    return "COALESCE(NULLIF(k.stack,''), k.op)" if has_stack(con) else "k.op"
