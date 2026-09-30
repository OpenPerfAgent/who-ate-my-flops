"""Load exported torch.profiler traces into the store the views read.

One chrome trace per rank, gzipped or not. The schema is in `common.py`.

The rank comes from `rank<N>` in the filename. torch.profiler's own export name
carries no rank, so name the files with it at capture time: without it the rank
is sort order, and every per-rank row downstream is then labelled with a rank
that may not be the one that produced it.
"""

from __future__ import annotations

import gzip
import json
import re
import sqlite3
import sys
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY, rank INTEGER, device INTEGER, stream INTEGER,
    tid INTEGER, cat TEXT, name TEXT, ts_ns INTEGER, dur_ns INTEGER, corr INTEGER,
    nbytes INTEGER);
CREATE TABLE IF NOT EXISTS steps (
    rank INTEGER, idx INTEGER, name TEXT, ts_ns INTEGER, dur_ns INTEGER);
CREATE TABLE IF NOT EXISTS mem (
    rank INTEGER, device INTEGER, ts_ns INTEGER, allocated INTEGER, reserved INTEGER);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS kernel_op (
    rank INTEGER, corr INTEGER, op TEXT, stack TEXT, PRIMARY KEY (rank, corr));
CREATE INDEX IF NOT EXISTS ix_ev_cat ON events(cat, rank, ts_ns);
CREATE INDEX IF NOT EXISTS ix_ev_corr ON events(rank, corr);
"""

# Anything unlisted is dropped. cuda_runtime is kept, not dropped: it is the only
# mechanical path from a GPU kernel back to the aten op that launched it.
_CAT = {
    "kernel": "kernel", "gpu_memcpy": "memcpy", "gpu_memset": "memset",
    "cpu_op": "cpu_op", "user_annotation": "annotation",
    "gpu_user_annotation": "annotation",
    "cuda_runtime": "runtime", "cuda_driver": "runtime",
}
_COMM = re.compile(r"nccl|all_?reduce|all_?gather|reduce_?scatter|broadcast|c10d|send_recv", re.I)
_STEP = re.compile(r"ProfilerStep#?(\d+)")
# Kineto spells a collective's dtype, not its width.
_WIDTH = {"float": 4, "float32": 4, "f32": 4, "double": 8, "float64": 8,
          "half": 2, "float16": 2, "f16": 2, "bfloat16": 2, "bf16": 2,
          "char": 1, "int8": 1, "uint8": 1, "byte": 1, "bool": 1, "fp8": 1,
          "int": 4, "int32": 4, "long": 8, "int64": 8, "short": 2, "int16": 2}
# Both ends of the op nesting, not the deepest few. The outermost frames say
# which phase and call site, the innermost say which op, and the middle restates
# one or the other. Keeping only the deepest dropped the frame that located
# anything and bought nothing for it.
_HEAD, _TAIL = 3, 3


def fold_stack(frames: list[str]) -> str:
    if len(frames) <= _HEAD + _TAIL:
        return " > ".join(frames)
    return " > ".join(frames[:_HEAD]) + " … " + " > ".join(frames[-_TAIL:])


def _nbytes(cat: str, args: dict) -> int | None:
    """Bytes moved, or None when the question does not apply.

    A collective carries element counts and a dtype rather than bytes, and
    max(In, Out) is the size both the all-reduce and all-gather conventions
    divide by, so it is the one that makes GB/s comparable across collectives.
    Without it a 4.7 GB device-to-host copy is a duration and nothing else.
    """
    if cat in ("memcpy", "memset"):
        return int(args.get("bytes") or 0) or None
    if cat == "comm":
        n = max(int(args.get("In msg nelems") or 0), int(args.get("Out msg nelems") or 0))
        w = _WIDTH.get(str(args.get("dtype", "")).lower())
        return n * w if n and w else None
    return None


def load_trace(con: sqlite3.Connection, path: Path, rank: int) -> None:
    with (gzip.open(path, "rt") if path.suffix == ".gz" else open(path)) as f:
        trace = json.load(f)
    evs, steps, mems = [], [], []
    for e in trace.get("traceEvents", []):
        args = e.get("args") or {}
        if e.get("ph") == "i" and e.get("name") == "[memory]":
            mems.append((rank, int(args.get("Device Id", -1)),
                         int(float(e.get("ts", 0)) * 1000),
                         int(args.get("Total Allocated", 0) or 0),
                         int(args.get("Total Reserved", 0) or 0)))
            continue
        if e.get("ph") != "X":
            continue
        cat = _CAT.get(e.get("cat", ""))
        if cat is None:
            continue
        name = e.get("name", "")
        if cat == "kernel" and _COMM.search(name):
            cat = "comm"
        ts = int(float(e.get("ts", 0)) * 1000)
        dur = int(float(e.get("dur", 0)) * 1000)
        tid = e.get("tid", 0)
        evs.append((rank, int(args.get("device", -1)), int(args.get("stream", -1)),
                    int(tid) if str(tid).lstrip("-").isdigit() else 0,
                    cat, name, ts, dur, int(args.get("correlation", -1)), _nbytes(cat, args)))
        m = _STEP.search(name)
        # The CPU-side annotation only. Kineto also emits a gpu_user_annotation
        # twin covering just the GPU span, and counting both duplicates the step
        # spine and fakes a large duration spread.
        if m and e.get("cat") == "user_annotation":
            steps.append((rank, int(m.group(1)), name, ts, dur))

    con.executemany("INSERT INTO events(rank,device,stream,tid,cat,name,ts_ns,dur_ns,corr,nbytes)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?)", evs)
    con.executemany("INSERT INTO steps VALUES (?,?,?,?,?)", steps)
    con.executemany("INSERT INTO mem VALUES (?,?,?,?,?)", mems)
    con.commit()
    print(f"  rank{rank}: {len(evs)} events, {len(steps)} steps, {len(mems)} memory samples")


def attribute_kernels(con: sqlite3.Connection) -> int:
    """Link every GPU kernel to the op that launched it, through `corr`.

    Kineto gives a `cuda_runtime` row, the launch call, sharing a `correlation`
    with the kernel it launched, and that row sits inside the `cpu_op` that
    issued it on the same thread. So the chain is

        kernel.corr -> cuda_runtime(same corr) -> the cpu_op stack containing it

    Both ends are recorded. `op` is the innermost caller and `stack` the nesting
    outermost first, because the leaf alone is rarely the diagnosis: `aten::copy_`
    names no call site and whatever called it does.

    One sweep per thread, not a scan per launch. Ops on one thread nest by time
    containment, so pushing each as it opens and popping it as it closes leaves
    the stack holding exactly what encloses the current instant. The pairwise
    version was launches times ops, and on an 8-rank capture, 355k launches
    against 3.27M ops, it never finished. The table came out empty, nothing
    reported it, and every lesion silently fell back to `ProfilerStep#N`.
    """
    runtime: dict[tuple[int, int], list] = {}
    for rank, corr, tid, ts, dur in con.execute(
            "SELECT rank, corr, tid, ts_ns, dur_ns FROM events WHERE cat='runtime' AND corr >= 0"):
        runtime.setdefault((rank, tid), []).append((ts, ts + dur, corr))
    if not runtime:
        return 0

    ops: dict[tuple[int, int], list] = {}
    for rank, tid, ts, dur, name in con.execute(
            "SELECT rank, tid, ts_ns, dur_ns, name FROM events WHERE cat='cpu_op'"):
        ops.setdefault((rank, tid), []).append((ts, ts + dur, name))

    out = []
    for key, launches in runtime.items():
        # (start asc, end desc) so an enclosing op is pushed before the one it
        # contains, which is what makes top of stack mean innermost.
        spans = sorted(ops.get(key, ()), key=lambda r: (r[0], -r[1]))
        launches.sort()
        stack: list[tuple[int, int, str]] = []
        i = 0
        for ts, end, corr in launches:
            while i < len(spans) and spans[i][0] <= ts:
                stack.append(spans[i])
                i += 1
            # Prune the whole stack, not its top: ops on one thread can cross
            # rather than nest, and a closed one buried under an open one stays.
            stack = [s for s in stack if s[1] > ts]
            chain = [s for s in stack if s[1] >= end]   # emit only from real ancestors
            if chain:
                out.append((key[0], corr, chain[-1][2], fold_stack([s[2] for s in chain])))
    con.executemany("INSERT OR REPLACE INTO kernel_op VALUES (?,?,?,?)", out)
    con.commit()
    return len(out)


def build(traces: list[Path], out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    con = sqlite3.connect(out)
    con.executescript(SCHEMA)
    for i, t in enumerate(sorted(traces)):
        m = re.search(r"rank(\d+)", t.name)
        if not m:
            print(f"  warning: {t.name} has no rank<N> in its name, calling it rank {i} by "
                  "position. Rename the traces if the rank matters, which it does for the "
                  "straggler row and every per-rank number.", file=sys.stderr)
        load_trace(con, t, int(m.group(1)) if m else i)
    linked = attribute_kernels(con)
    ranks = con.execute("SELECT COUNT(DISTINCT rank) FROM events").fetchone()[0]
    con.executemany("INSERT OR REPLACE INTO meta VALUES (?,?)",
                    [("n_ranks", str(ranks)), ("traces", ";".join(t.name for t in traces))])
    con.commit()
    con.close()
    print(f"  {linked} kernels linked to the op that launched them")
    return out
