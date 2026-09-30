"""View 2: where time is lost, ranked, per rank.

One machine, two detectors: find intervals with a property, cost them, attribute
them, rank them. A single local finding is a lesion.

    gap         nothing running, the complement of the busy-interval union
    fragmented  a run of kernels all shorter than a self-calibrated floor, so the
                GPU is busy doing HBM round-trips. This is the only view that
                explains a 99.8% busy step that still has headroom.

The fragmented thresholds come from the trace:

    launch_floor = p10 of the gaps between merged busy intervals
    tau_small    = max(3 * launch_floor, 20us)
    region       = >= K kernels under tau_small, at most m misses, no gap >= MIN_GAP
    cost         = the idle inside the region, the only part the timeline can price
"""

from __future__ import annotations

import bisect
import re
import sqlite3
import statistics as st
from dataclasses import dataclass, field

from .common import (MS, US, Intervals, busy_intervals, clip, fmt_dur, fmt_ms, issuer_col,
                     table, union)

CAUSE_WIDTH = 200
MAX_CAUSES = 3
# A runtime call is named as blocking when it covers this much of the lesion.
# Below it, it is one call among many rather than the reason.
BLOCKING_SHARE = 0.10
# Under this much explained, the lesion is reported as UNKNOWN rather than
# labelled from the little that was found. On an 8-rank capture, 90.9% of gap
# time was covered by an aten op and 0.4% by a runtime call, leaving 9.1% that no
# default trace can explain: the CPU was in python, numpy, a lock or a queue.
MIN_COVERED = 0.50
NEIGHBOUR_FRAMES = 2
# Shortest idle the gap detector reports, and the boundary of a fragmented run.
MIN_GAP = 50 * US
_STEP_ONLY = re.compile(r"^ProfilerStep#\d+$")


@dataclass
class Lesion:
    kind: str
    rank: int
    ts_ns: int
    dur_ns: int
    cost_ns: int
    step_idx: int | None = None
    offset_ns: int | None = None
    attribution: list[str] = field(default_factory=list)
    cause: list[tuple[str, int]] = field(default_factory=list)
    blocking: list[tuple[str, int]] = field(default_factory=list)
    covered: float = 0.0
    neighbours: tuple[str, str] | None = None
    detail: dict = field(default_factory=dict)

    @property
    def cause_text(self) -> str:
        """What the host was doing, and what it was blocked in.

        When nothing inside explains the lesion, fall back to its two ends. Never
        close on the waiter: what a gap waits for is never inside it. A 725 ms
        hole reading `UNKNOWN` and a 725 ms hole reading "between a DtoH for
        aten::item and an HtoD for the next batch" are the same measurement and
        not the same finding.
        """
        if self.covered < MIN_COVERED and not self.blocking:
            head = f"UNKNOWN ({self.covered:.0%} covered)"
            return f"{head} — waits: {self.neighbours[0]}  ->  {self.neighbours[1]}" \
                if self.neighbours else head
        out = " · ".join(f"{n} {fmt_dur(t)}" for n, t in self.cause) or "-"
        if self.blocking:
            out += "  [blocked in " + ", ".join(f"{n} {fmt_dur(t)}" for n, t in self.blocking) + "]"
        return out


def launch_floor_ns(con: sqlite3.Connection, rank: int) -> int:
    """p10 of the gaps between merged busy intervals, one launch as this trace ran it.

    Merged, because kernels on different streams overlap and pairing each with
    its predecessor invents gaps while the GPU is busy.
    """
    row = con.execute("SELECT MIN(ts_ns), MAX(ts_ns + dur_ns) FROM events WHERE rank=? "
                      "AND cat IN ('kernel','comm')", (rank,)).fetchone()
    if not row or row[0] is None:
        return 2 * US
    iv = busy_intervals(con, rank, row[0], row[1])
    gaps = [b[0] - a[1] for a, b in zip(iv, iv[1:]) if b[0] > a[1]]
    if not gaps:
        return 2 * US
    floor = int(st.quantiles(gaps, n=10)[0]) if len(gaps) >= 10 else min(gaps)
    return min(floor, MIN_GAP)   # anything this long is a gap, not a launch


def detect_gaps(con: sqlite3.Connection, rank: int, t0: int, t1: int,
                min_ns: int = MIN_GAP) -> list[Lesion]:
    out, prev = [], t0
    for s, e in busy_intervals(con, rank, t0, t1):
        if s - prev >= min_ns:
            out.append(Lesion("gap", rank, prev, s - prev, s - prev))
        prev = max(prev, e)
    if t1 - prev >= min_ns:
        out.append(Lesion("gap", rank, prev, t1 - prev, t1 - prev))
    return out


def detect_fragmented(con: sqlite3.Connection, rank: int, t0: int, t1: int,
                      k: int = 8, m: int = 1) -> list[Lesion]:
    """Runs of tiny kernels: busy time that is mostly launch plus HBM round-trip."""
    floor = launch_floor_ns(con, rank)
    tau = max(3 * floor, 20 * US)
    rows = con.execute(
        "SELECT ts_ns, dur_ns, name FROM events WHERE rank=? AND cat IN ('kernel','comm') "
        "AND ts_ns < ? AND ts_ns + dur_ns > ? ORDER BY ts_ns", (rank, t1, t0)).fetchall()

    out: list[Lesion] = []
    run: list[tuple[int, int, str]] = []
    misses = 0

    def flush() -> None:
        nonlocal run, misses
        if len([r for r in run if r[1] < tau]) >= k:
            t_end = max(ts + d for ts, d, _ in run)
            span = t_end - run[0][0]
            # Every kernel in the window, not just the run's own: one long kernel
            # excluded from the run can still be filling the span.
            busy = sum(e - s for s, e in busy_intervals(con, rank, run[0][0], t_end))
            names: dict[str, int] = {}
            for _, d, nm in run:
                key = nm.split("(")[0][:48]
                names[key] = names.get(key, 0) + d
            out.append(Lesion(
                "fragmented", rank, run[0][0], span,
                # Idle only: a fusion estimate prices the same gaps twice, and
                # what fusing saves on the CPU is not idle, so it stays in detail.
                cost_ns=max(0, span - busy),
                detail={"n_kernels": len(run),
                        "fusible_launches": max(0, sum(1 for r in run if r[1] < tau) - 1),
                        "density_per_ms": round(len(run) / max(span / MS, 1e-9), 1),
                        "top": [f"{nm} {t / MS:.2f}ms"
                                for nm, t in sorted(names.items(), key=lambda x: -x[1])[:3]]}))
        run, misses = [], 0

    # The same idle the gap detector sees, merged over every GPU category, or the
    # two disagree about what a gap is and both charge for one.
    iv = busy_intervals(con, rank, t0, t1)
    holes = [a[1] for a, b in zip(iv, iv[1:]) if b[0] - a[1] >= MIN_GAP]

    end = 0
    for ts, dur, name in rows:
        i = bisect.bisect_left(holes, end)
        if run and i < len(holes) and holes[i] < ts:
            flush()
        if dur >= tau and not (run and misses < m):
            flush()          # then fall through: this event may start the next run
        if dur < tau:
            run.append((ts, dur, name))
        elif run:
            run.append((ts, dur, name))
            misses += 1      # total misses in the run, not consecutive ones
        end = max(end, ts + dur)
    flush()
    return out


class _Neighbours:
    """The GPU event ending before an instant and the one starting after it.

    A gap is the complement of the busy union, so these two are exactly the event
    it interrupted and the event it ended at: the ends of the wait.
    """

    def __init__(self, rows: list[tuple]) -> None:
        by_end = sorted(rows, key=lambda r: r[1])
        self.ends = [r[1] for r in by_end]
        self.end_label = [_event_label(r) for r in by_end]
        by_start = sorted(rows, key=lambda r: r[0])
        self.starts = [r[0] for r in by_start]
        self.start_label = [_event_label(r) for r in by_start]

    def around(self, t0: int, t1: int) -> tuple[str, str] | None:
        i = bisect.bisect_right(self.ends, t0) - 1
        j = bisect.bisect_left(self.starts, t1)
        if i < 0 and j >= len(self.starts):
            return None
        return (self.end_label[i] if i >= 0 else "(start of capture)",
                self.start_label[j] if j < len(self.starts) else "(end of capture)")


def _event_label(row: tuple) -> str:
    """`<what the GPU did> (<who issued it>)`, short enough for a table cell.

    For a memcpy or a collective the event name is the finding: "Memcpy DtoH
    (Device -> Pinned)" says a synchronising read, where the op stack alone would
    not. For a compute kernel it is the reverse.
    """
    _, _, cat, name, stack = row
    issuer = " > ".join((stack or "").split(" > ")[-NEIGHBOUR_FRAMES:])
    if cat in ("memcpy", "memset", "comm"):
        head = name.split("(")[0].strip() if cat == "memcpy" else name.split("(")[0][:40]
        return f"{head} ({issuer})" if issuer else head
    return issuer or name.split("(")[0][:48]


def _top_level(rows: list[tuple]) -> list[tuple[int, int, str]]:
    """The outermost ops per thread: the callers, not the leaves.

    Ops on one thread nest by containment, so sorting by (start asc, end desc)
    and keeping any op starting at or after the current outermost one ends yields
    exactly the top level. `aten::copy_` names no call site; whatever called it
    does.
    """
    by_tid: dict[int, list[tuple[int, int, str]]] = {}
    for s, e, tid, name in rows:
        by_tid.setdefault(tid, []).append((s, e, name))
    out: list[tuple[int, int, str]] = []
    for spans in by_tid.values():
        spans.sort(key=lambda r: (r[0], -r[1]))
        cur_end = -1
        for s, e, name in spans:
            if s >= cur_end:
                out.append((s, e, name))
                cur_end = e
    return out


def attribute(con: sqlite3.Connection, lesions: list[Lesion]) -> None:
    """Fill step offsets, the enclosing scope, and the cause, in place.

    Two questions needing different evidence. What scope is this in comes from the
    enclosing `record_function`, when the workload emits one. What was the machine
    doing instead comes from the ops that launched the kernels inside a busy
    lesion; a gap has no kernels inside it by construction, so for a gap the
    answer is on the CPU timeline instead.

    The second question used to go unasked for gaps, and gaps are half of all
    lesion cost. They came back labelled `ProfilerStep#N`, which reads as
    attribution and is not.
    """
    steps = con.execute("SELECT rank, idx, ts_ns, dur_ns FROM steps ORDER BY ts_ns").fetchall()
    ann = con.execute(
        "SELECT rank, name, ts_ns, dur_ns FROM events WHERE cat='annotation' AND dur_ns > 0 "
        "ORDER BY ts_ns").fetchall()
    stack = issuer_col(con)
    ranks = {les.rank for les in lesions}

    def index(sql: str) -> dict[int, Intervals]:
        return {r: Intervals(con.execute(sql, (r,)).fetchall()) for r in ranks}

    # Indexed in memory for the same reason on both sides: asking SQL once per
    # lesion was 7905 queries and 47 s of a 54 s scan, and the window predicate
    # cannot use an index, so every one of them was a full scan.
    cpu_ops = index("SELECT ts_ns, ts_ns + dur_ns, tid, name FROM events "
                    "WHERE rank=? AND cat='cpu_op'")
    runtime = index("SELECT ts_ns, ts_ns + dur_ns, tid, name FROM events "
                    "WHERE rank=? AND cat='runtime'")
    launched = index(f"SELECT e.ts_ns, e.ts_ns + e.dur_ns, e.dur_ns, {stack} FROM events e "
                     "LEFT JOIN kernel_op k ON k.rank = e.rank AND k.corr = e.corr "
                     "WHERE e.rank=? AND e.cat IN ('kernel','comm')")
    around = {r: _Neighbours(con.execute(
        f"SELECT e.ts_ns, e.ts_ns + e.dur_ns, e.cat, e.name, {stack} FROM events e "
        "LEFT JOIN kernel_op k ON k.rank = e.rank AND k.corr = e.corr "
        "WHERE e.rank=? AND e.cat IN ('kernel','comm','memcpy','memset')",
        (r,)).fetchall()) for r in ranks}

    for les in lesions:
        t0, t1 = les.ts_ns, les.ts_ns + les.dur_ns
        span = max(les.dur_ns, 1)
        mid = t0 + les.dur_ns // 2
        for r, idx, s, d in steps:
            if r == les.rank and s <= mid < s + d:
                les.step_idx, les.offset_ns = idx, t0 - s
                break
        enclosing = sorted((d, n) for r, n, s, d in ann if r == les.rank and s <= mid < s + d)
        les.attribution = [n for _, n in enclosing[:MAX_CAUSES]]

        ops = _top_level(cpu_ops[les.rank].overlapping(t0, t1))
        by_name: dict[str, list[tuple[int, int]]] = {}
        for s, e, name in ops:
            by_name.setdefault(name, []).append((max(s, t0), min(e, t1)))
        cause = [(n, union(v)) for n, v in by_name.items()]

        # Ranked by kernel time, not by how often an op appears: counting picked
        # `aten::div x6` out of a 52-kernel region as if div were the story.
        by_launcher: dict[str, int] = {}
        for _, _, dur, label in launched[les.rank].overlapping(t0, t1):
            if label:
                by_launcher[label] = by_launcher.get(label, 0) + dur
        cause += list(by_launcher.items())

        merged: dict[str, int] = {}
        for name, ns in cause:
            merged[name] = max(merged.get(name, 0), ns)
        les.cause = sorted(merged.items(), key=lambda kv: -kv[1])[:MAX_CAUSES]

        rt = [(n, max(s, t0), min(e, t1)) for s, e, _, n in runtime[les.rank].overlapping(t0, t1)]
        les.blocking = sorted(((n, e - s) for n, s, e in rt if (e - s) / span >= BLOCKING_SHARE),
                              key=lambda kv: -kv[1])[:2]
        les.covered = union([(max(s, t0), min(e, t1)) for s, e, _ in ops]
                            + [(s, e) for _, s, e in rt]) / span
        if les.covered < MIN_COVERED and not les.blocking:
            les.neighbours = around[les.rank].around(t0, t1)

        # A step number is not a place in the code, so when the only annotation on
        # offer is ProfilerStep#N, the cause replaces it.
        if all(_STEP_ONLY.match(a) for a in les.attribution) and les.cause:
            les.attribution = [n for n, _ in les.cause] + les.attribution[:1]


def scan(con: sqlite3.Connection) -> list[Lesion]:
    """Run both detectors on every rank and return lesions ranked by cost."""
    ranks = [r for (r,) in con.execute("SELECT DISTINCT rank FROM events ORDER BY 1")]
    out: list[Lesion] = []
    for rank in ranks:
        row = con.execute(
            "SELECT MIN(ts_ns), MAX(ts_ns + dur_ns) FROM events WHERE rank=? "
            "AND cat IN ('kernel','comm','memcpy','memset')", (rank,)).fetchone()
        if not row or row[0] is None:
            continue
        t0, t1 = row
        out += detect_gaps(con, rank, t0, t1)
        out += detect_fragmented(con, rank, t0, t1)
    attribute(con, out)
    out.sort(key=lambda x: -x.cost_ns)
    return out


def rollup(lesions: list[Lesion]) -> list[tuple[str, int, int]]:
    """Aggregate lesion cost by enclosing scope."""
    agg: dict[str, list[int]] = {}
    for les in lesions:
        key = les.attribution[0] if les.attribution else "(unattributed)"
        a = agg.setdefault(key, [0, 0])
        a[0] += les.cost_ns
        a[1] += 1
    return sorted(((k, v[0], v[1]) for k, v in agg.items()), key=lambda x: -x[1])


def render(con: sqlite3.Connection, top: int = 12, cutoff: float = 0.01) -> str:
    lesions = scan(con)
    out = ["== 2. Lost time, ranked =="]
    if not lesions:
        return "\n".join(out + ["  (none)"])
    total = sum(x.cost_ns for x in lesions)
    if not total:
        return "\n".join(out + [f"  {len(lesions)} sites, none costing measurable wall clock."])

    rows = []
    for les in lesions[:top]:
        if les.cost_ns < cutoff * total:
            break
        note = (f"n={les.detail['n_kernels']} dens={les.detail['density_per_ms']}/ms "
                f"fuse={les.detail['fusible_launches']}" if les.kind == "fragmented" else "")
        rows.append([les.kind, f"r{les.rank}",
                     les.step_idx if les.step_idx is not None else "-",
                     fmt_ms(les.dur_ns), fmt_ms(les.cost_ns), f"{les.covered:.0%}",
                     f"{clip(les.cause_text, CAUSE_WIDTH)} {note}".strip()])

    if rows:
        out.append(table(rows, ["kind", "rank", "step", "dur_ms", "cost_ms", "cov%",
                                "cause — what ran instead"]))
        hidden = len(lesions) - len(rows)
        if hidden > 0:
            rest = sum(x.cost_ns for x in lesions[len(rows):])
            out.append(f"  +{hidden} more below the {cutoff:.0%} cutoff ({fmt_ms(rest)}ms total)")
    else:
        # Not a bug and not "nothing to find". An empty table above a "+15177
        # more" line has been read as both. It is a finding: the recoverable time
        # is real but spread so thin that no single site clears the cutoff, which
        # rules out a local fix and points at a structural one.
        out.append(f"  no single site reaches the {cutoff:.0%} cutoff, so the loss is DIFFUSE "
                   f"rather than localised.")
        out.append(f"  {len(lesions)} sites share {fmt_ms(total)}ms; the biggest is "
                   f"{fmt_ms(lesions[0].cost_ns)}ms ({lesions[0].cost_ns / total:.1%}).")
        out.append("  Read it as: there is no one place to fix. Look for a change that removes "
                   "a CLASS\n  of sites, such as fusion, fewer launches, or a different schedule.")

    gap = sum(x.cost_ns for x in lesions if x.kind == "gap")
    frag = total - gap
    out.append(f"  recoverable upper bound: {fmt_ms(total)}ms "
               f"(gap={fmt_ms(gap)}ms  fragmented={fmt_ms(frag)}ms)")

    dark = sum(x.cost_ns for x in lesions if x.covered < MIN_COVERED and not x.blocking)
    if dark:
        out.append(f"  {fmt_ms(dark)}ms ({dark / total:.0%}) is UNKNOWN: nothing inside it "
                   f"explains it, no aten op and no runtime call.\n  That is the CPU in python, "
                   f"numpy, a lock or a queue, and with_stack=True is the only way to name it "
                   f"directly.\n  The waits pair on each row is what is left: the GPU event that "
                   f"ran into the hole and the one that ended it.")

    roll = rollup(lesions)[:6]
    if roll:
        out.append("  roll-up by enclosing scope (record_function if there is one, else cause):")
        out.append(table([[s[:56], fmt_ms(c), n, f"{c / total:.0%}"] for s, c, n in roll],
                         ["scope", "cost_ms", "n", "share"]))
    return "\n".join(out)
