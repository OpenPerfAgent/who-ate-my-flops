"""The workspace layout, and the two record formats every skill writes into.

One shape per campaign, defined here and nowhere else. Three campaigns run from
a prose description produced three different shapes, and one of them left 126
loose files at its root.

    <ws>/
      contract.md              what was agreed in init, copied here as the record
      benchmark.csv            the candidates, numbers only
      latest-report.md         the report, and later the handover
      records/
        computation_graph.json what the job computes
        execution_schedule.json when and where it computes it, and what moves
        candidates.jsonl       one line per candidate, in the order they were tried
        mistakes.md            what you got wrong, for fixing the plugin
      tools/                   helper scripts
      runs/                    what the benchmark script itself writes
      commits/<sha>/           one directory per candidate
        trace/                 the profiler output for that candidate
        report.md              the diagnosis torch-profile wrote for it

`runs/` and `commits/` both hold run artifacts, and the line between them is who
wrote them. The benchmark script drops its own output in `runs/`, redirected
there so it never lands in a candidate commit. Everything produced about one
candidate lives under that candidate's sha instead.

records/candidates.jsonl carries the reasoning and benchmark.csv carries the numbers, so
`add` writes both from one call. Written separately they drift, and then nobody
can tell which row belongs to which decision.

records/candidates.jsonl says what each change did. records/mistakes.md says what the agent
running the campaign got wrong, and it is read afterwards to fix this plugin, so
it is the one file here written for somebody other than the campaign.

Commands:
    python workspace.py init      <ws> [contract.md]   create the layout
    python workspace.py candidate <ws> <sha>           make commits/<sha>/, print the path
    python workspace.py add       <ws> --sha ... --verdict ... [numbers]   record one candidate
    python workspace.py arm       <ws>                 record which session owns the campaign
    python workspace.py finish    <ws>                 is the campaign actually done?

`finish` is the loop condition. In Claude Code, `optimize` sets a session timer
that runs it whenever the session goes idle. In Codex, the optimize skill
instructs the agent to run it before ending its work. Either way the campaign
is over when it passes, and `finish` takes it off the registry `arm` wrote.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import re
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

DIRS = ("records", "tools", "runs", "commits")

# Every candidate ends up as one row here. The columns are fixed so the file
# stays readable in a spreadsheet; anything that does not fit goes in the jsonl.
COLUMNS = ("sha", "verdict", "warmup_s", "step_s", "peak_gpu_gb", "peak_cpu_gb")


def measurement(value: str | float) -> float | str:
    """Accept a finite nonnegative number or N/A with a reason."""
    if isinstance(value, str) and re.fullmatch(r"N/A\s*[:—-]\s*\S.*", value.strip()):
        return value.strip()
    try:
        number = float(value)
        if not isinstance(value, bool) and math.isfinite(number) and number >= 0:
            return number
    except (TypeError, ValueError):
        pass
    raise argparse.ArgumentTypeError("use a nonnegative number or 'N/A: reason'")

MISTAKES = """\
# Mistakes

Write one the moment you get something wrong, in the same turn. This file is
read after the campaign to fix the plugin, so an entry earns its place by saying
what would have stopped you.

Three lines each: what you did and what it cost, what caught it, and what in the
plugin should have caught it sooner. A wrong turn you found yourself in five
minutes still counts.

* took the first step time for steady state, so three candidates were judged
  against a baseline that was two thirds compile warmup
  caught by the 90 s gap between step 1 and step 2 in a later trace
  setup-baseline should say to check the first step against the median before
  fixing the window

The `*` entry is an example of the shape. Write yours as `-` bullets.
"""

SCAFFOLD = {
    "records/computation_graph.json": '{"_comment": "written by extract-computation-graph"}\n',
    "records/execution_schedule.json": '{"_comment": "written by extract-computation-graph"}\n',
    "records/candidates.jsonl": "",
    "latest-report.md": "# Report\n\nNot written yet.\n",
    "records/mistakes.md": MISTAKES,
}


def init(ws: Path, contract: Path | None = None) -> None:
    """Create the layout. Existing files are left alone, so this is safe to rerun."""
    ws.mkdir(parents=True, exist_ok=True)
    for d in DIRS:
        (ws / d).mkdir(exist_ok=True)

    for name, body in SCAFFOLD.items():
        path = ws / name
        if not path.exists():
            path.write_text(body)

    bench = ws / "benchmark.csv"
    if not bench.exists():
        with bench.open("w", newline="") as f:
            csv.writer(f).writerow(COLUMNS)

    target = ws / "contract.md"
    if contract and not target.exists():
        shutil.copyfile(contract, target)
    elif not target.exists():
        target.write_text("# Campaign contract\n\nCopy the settled contract here.\n")
        print(f"note: no contract given, so {target} is a stub", file=sys.stderr)

    print(ws)


def candidate(ws: Path, sha: str) -> Path:
    """Make the directory this candidate's trace and report go in, and print it."""
    d = ws / "commits" / sha
    (d / "trace").mkdir(parents=True, exist_ok=True)
    print(d)
    return d


def add(ws: Path, sha: str, verdict: str, why: str = "", title: str = "",
        **numbers: float | str | None) -> None:
    """Append one candidate to both records.

    verdict is `base` for the starting measurement, then `kept` or `rejected`.
    A rejected candidate still gets a row: the reason it lost is half of what a
    later reader is looking for. Fill each measurement with a number or
    `N/A: reason`; `finish` flags missing values.
    """
    (ws / "commits" / sha / "trace").mkdir(parents=True, exist_ok=True)
    record = {"sha": sha, "verdict": verdict, "title": title, "why": why,
              **{k: v for k, v in numbers.items() if v is not None}}
    with (ws / "records/candidates.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")

    row = [sha, verdict] + [numbers.get(c) for c in COLUMNS[2:]]
    with (ws / "benchmark.csv").open("a", newline="") as f:
        csv.writer(f).writerow(["" if v is None else v for v in row])

    print(f"{verdict} {sha[:12]}")


# ------------------------------------------------------------------ arm

# One campaign per session, in a registry outside every workspace and every
# checkout. Keyed by session so a hook in one session never holds another.
STATE_ENV = "WAMF_STATE_DIR"
SESSION_VARS = ("CODEX_THREAD_ID", "CLAUDE_CODE_SESSION_ID")


def state_dir() -> Path:
    """Absolute, so `arm` and the hook agree on it whatever directory each runs in."""
    return Path(os.environ.get(STATE_ENV) or Path.home() / ".who-ate-my-flops") \
        .expanduser().resolve()


@contextmanager
def locked_registry():
    """One writer at a time. Two sessions arming in the same second must not
    lose each other's line."""
    state_dir().mkdir(parents=True, exist_ok=True)
    with open(state_dir() / "armed.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def registry() -> Path:
    return state_dir() / "armed.tsv"


def armed() -> dict[str, str]:
    """session id -> workspace, one line each."""
    out: dict[str, str] = {}
    try:
        lines = registry().read_text().splitlines()
    except OSError:
        return out
    for ln in lines:
        if "\t" in ln:
            sid, ws = ln.split("\t", 1)
            out[sid.strip()] = ws.strip()
    return out


def rewrite_registry(entries: dict[str, str]) -> None:
    """Call under `locked_registry`."""
    reg = registry()
    reg.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=reg.parent, prefix=reg.name + ".")
    with os.fdopen(fd, "w") as f:
        f.write("".join(f"{sid}\t{ws}\n" for sid, ws in entries.items()))
    os.replace(tmp, reg)


def session_id() -> str | None:
    """The session this shell belongs to. Codex and Claude Code each set one."""
    for var in SESSION_VARS:
        if os.environ.get(var):
            return os.environ[var]
    return None


def campaign_for(session: str) -> Path | None:
    ws = armed().get(session)
    return Path(ws) if ws else None


def bump(session: str) -> int | None:
    """Per session continuation counter. Returns the new count, or None when it
    could not be written: a count that does not persist never reaches the
    limit, and the hook must fail open on it rather than hold the turn forever."""
    path = state_dir() / f"{session}.count"
    try:
        n = int(path.read_text().strip() or 0)
    except (OSError, ValueError):
        n = 0
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(n + 1))
    except OSError:
        return None
    return n + 1


def arm(ws: Path, session: str | None = None) -> None:
    """Record which session owns the campaign.

    Arm after setup passes. Setup can still hand back, and a campaign armed
    ahead of it is trapped in a loop it cannot start.
    """
    ws = ws.resolve()
    if not ws.is_dir():
        raise SystemExit(f"no workspace at {ws}. Run `workspace.py init` first")
    session = session or session_id()
    if not session:
        raise SystemExit("no session id: none of " + ", ".join(SESSION_VARS) + " is set. "
                         "Pass --session <id> from the client's own hook payload or status line.")
    with locked_registry():
        entries = armed()
        if entries.get(session) != str(ws):
            (state_dir() / f"{session}.count").unlink(missing_ok=True)   # a new campaign starts at zero
        entries[session] = str(ws)
        rewrite_registry(entries)
    print(f"armed {ws}")
    print(f"  session  {session}")
    print(f"  registry {registry()}")


def deregister(ws: str | Path) -> None:
    """Take a campaign off the registry, and its counters with it."""
    ws = str(Path(ws).resolve())
    with locked_registry():
        entries = armed()
        gone = [sid for sid, w in entries.items() if w == ws]
        for sid in gone:
            del entries[sid]
            (state_dir() / f"{sid}.count").unlink(missing_ok=True)
        if gone:
            rewrite_registry(entries)


def release_session(session: str) -> None:
    """Stop holding one session, leaving the campaign registered to any other."""
    with locked_registry():
        entries = armed()
        if session in entries:
            del entries[session]
            rewrite_registry(entries)
        (state_dir() / f"{session}.count").unlink(missing_ok=True)


# ------------------------------------------------------------------ finish

# `<the command that runs the job>`. Lowercase words only, so `->` and `<=` in a
# real answer are left alone.
PLACEHOLDER = re.compile(r"<[a-z][^<>]*[a-z][^<>]*>")

# On its own line, so the word inside a sentence denying it does not count.
STATUS_CLOSED = re.compile(r"^\s*(?:[-*]\s*)?\**\s*status\s*:\s*closed\b", re.I | re.M)


def _candidate_problems(ws: Path) -> list[str]:
    out: list[str] = []
    led = ws / "records/candidates.jsonl"
    if not led.exists():
        return ["records/candidates.jsonl is missing, so the record was removed"]

    records, unparsed = [], []
    for n, line in enumerate(led.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except ValueError:
            unparsed.append(n)
    if unparsed:
        out.append(f"records/candidates.jsonl line(s) {', '.join(map(str, unparsed))} do not parse. "
                   "`workspace.py add` writes both records in step")
    if not records:
        return out + ["no candidate recorded, not even the baseline"]

    if not any(r.get("verdict") == "base" for r in records):
        out.append("no `base` row, and every later number is read against it")
    silent = [str(r.get("sha", "?"))[:12] for r in records
              if r.get("verdict") == "rejected" and not str(r.get("why", "")).strip()]
    if silent:
        out.append(f"rejected with no reason: {', '.join(silent)}")
    for r in records:
        if r.get("verdict") != "base":
            continue
        sha = str(r.get("sha", "?"))
        d = ws / "commits" / sha
        if not any((d / "trace").glob("*")):
            out.append(f"no trace under commits/{sha[:12]}/trace/. The baseline is the one "
                       "diagnosis of the user's own code, so profile it in full")
        report = d / "report.md"
        if not report.exists() or not report.read_text().strip():
            out.append(f"no diagnosis at commits/{sha[:12]}/report.md. torch-profile writes "
                       "it, and the handover points back to it")
    for r in records:
        for key in COLUMNS[2:]:
            try:
                measurement(r.get(key))
            except argparse.ArgumentTypeError:
                out.append(f"{str(r.get('sha', '?'))[:12]}: {key} needs a measurement "
                           "or `N/A: reason`")

    try:
        with (ws / "benchmark.csv").open(newline="") as f:
            rows = list(csv.reader(f))
    except OSError:
        out.append("benchmark.csv is missing")
    else:
        if max(len(rows) - 1, 0) != len(records):
            out.append(f"benchmark.csv has {max(len(rows) - 1, 0)} row(s) for {len(records)} "
                       "candidate(s). `workspace.py add` writes both from one call")
    return out


def finish_problems(ws: Path) -> list[str]:
    """What is outstanding. An empty list means the loop can end.

    This checks the record: the contract settled, the graph and schedule built,
    every candidate judged, the report closed. Whether
    anything is left worth trying is a judgement, and it goes in the report.
    """
    problems: list[str] = []

    contract = ws / "contract.md"
    if not contract.exists():
        problems.append("contract.md is missing")
    else:
        text = contract.read_text()
        if "Copy the settled contract here." in text:
            problems.append("contract.md is still the stub. Copy in the one init settled")
        holes = PLACEHOLDER.findall(text)
        if holes:
            problems.append(f"contract.md has {len(holes)} unfilled placeholder(s), "
                            f"from `{holes[0]}`")
        if re.search(r"^\s*TBD\s*$", text, re.M):
            problems.append("contract.md still has a TBD section")

    for name in ("records/computation_graph.json", "records/execution_schedule.json"):
        path = ws / name
        if not path.exists():
            problems.append(f"{name} is missing, and the profile is read against it")
        elif path.read_text().strip() == SCAFFOLD[name].strip():
            problems.append(f"{name} is still the scaffold, so extract-computation-graph "
                            "never ran")

    report = ws / "latest-report.md"
    if not report.exists():
        problems.append("latest-report.md is missing")
    elif report.read_text().strip() == SCAFFOLD["latest-report.md"].strip():
        problems.append("latest-report.md is still the scaffold")
    elif not STATUS_CLOSED.search(report.read_text()):
        problems.append("latest-report.md has no `status: closed` line of its own. Write that "
                        "marker deliberately, or say what the campaign is blocked on")

    return problems + _candidate_problems(ws)


def cmd_finish(ws: Path) -> None:
    problems = finish_problems(ws)
    if problems:
        print(f"== {ws} : NOT finished, {len(problems)} outstanding ==")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)
    print(f"== {ws} : finished ==")
    deregister(ws)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="create the layout")
    p.add_argument("ws", type=Path)
    p.add_argument("contract", type=Path, nargs="?", help="the settled contract to copy in")

    p = sub.add_parser("candidate", help="make commits/<sha>/ for a candidate's artifacts")
    p.add_argument("ws", type=Path)
    p.add_argument("sha")

    p = sub.add_parser("add", help="record one candidate in both files")
    p.add_argument("ws", type=Path)
    p.add_argument("--sha", required=True)
    p.add_argument("--verdict", required=True, choices=("base", "kept", "rejected"))
    p.add_argument("--title", default="", help="one line on what changed")
    p.add_argument("--why", default="", help="why it was kept or rejected")
    for col in COLUMNS[2:]:
        p.add_argument(f"--{col.replace('_', '-')}", type=measurement, default=None,
                       help="measured value or 'N/A: reason'")

    p = sub.add_parser("arm", help="record which session owns the campaign")
    p.add_argument("ws", type=Path)
    p.add_argument("--session", default=None,
                   help="the session id, default from " + " or ".join(SESSION_VARS))

    p = sub.add_parser("finish", help="is the campaign done?")
    p.add_argument("ws", type=Path)

    args = vars(ap.parse_args())
    cmd = args.pop("cmd")
    if cmd == "init":
        init(args["ws"], args["contract"])
    elif cmd == "candidate":
        candidate(args["ws"], args["sha"])
    elif cmd == "arm":
        arm(args["ws"], args["session"])
    elif cmd == "finish":
        cmd_finish(args["ws"])
    else:
        add(**args)


if __name__ == "__main__":
    main()
