---
name: setup-baseline
description: Build the starting point both diagnose and optimize measure everything else against: a worktree, a workspace, the computation graph and execution schedule, and the base commit measured for speed, memory and correctness and profiled in full. It arms nothing, so the caller decides what happens next.
---

# setup-baseline

In: `contract.md`, as init settled it.
Out: a workspace whose base row is recorded in `records/candidates.jsonl` and
`benchmark.csv`, and whose base commit carries a trace and a diagnosis.

1. Create a separate worktree and make every change there. Leave the original
   checkout alone.
2. Use `<worktree>/workspace-who-ate-my-flops/` as `<ws>`. Add
   `/workspace-who-ate-my-flops/` to the local exclude file returned by
   `git rev-parse --git-path info/exclude` from the worktree, preserving existing
   entries. Create the workspace with `python <plugin root>/tools/workspace.py
   init <ws> <contract>`. The plugin root is the directory two levels above this
   SKILL.md, and Claude Code also exposes it as `${CLAUDE_PLUGIN_ROOT}`. Every
   tool this plugin ships resolves from there, because you are standing in the
   target's worktree and never in the plugin. Read workspace.py first: it
   defines the file formats everything downstream uses, and the workspace is the
   common ground the skills coordinate through.
   Keep process records in `records/`, including any planning notes.
   Leave `contract.md`, `latest-report.md`, and `benchmark.csv` at the workspace root.
   Put helper scripts in `tools/` and run artifacts in `runs/`. Avoid adding
   ad hoc files to the workspace root.
3. Use the extract-computation-graph skill to build the job's computation graph
   and execution schedule inside the workspace: `records/computation_graph.json` for what
   is computed, and `records/execution_schedule.json` for when and where it is computed
   and what moves between devices. Both get read against the profile results
   later, to locate the performance problems and find fixes for them.
4. Record the original code's speed, memory and correctness using the scripts
   defined in `run` of contract.md, and diagnose it. The baseline covers all
   four, and all four get written down:
   - Diagnosis: run the torch-profile skill on the unmodified code, in full.
     Its traces go under `commits/<base sha>/trace/` and its report to
     `commits/<base sha>/report.md`. This is the one diagnosis of the user's
     own code: every later profile only says whether a change removed
     something from it, and the handover points back to it. `finish` refuses a
     campaign without it.
   - Speed: by default measure the warm-up cost, which is paid once, and the
     steady step time, which is paid many times over.
   - Memory: measure CPU and GPU memory, and watch how each moves across the
     warm-up phase and the steady-step phase.
   - Correctness:
     a. Read the `## Correctness Instrumentation Plan` section of contract.md for
        where the checkpoints go and which recorder it names.
     b. Use the recorder the repo already has. In imp_genai that is
        `research/core/probe`, the same code vendored under its own name.
        Install one only when the repo has none, and then install parity
        (https://github.com/OpenPerfAgent/parity): `import parity`, recorded when
        `PARITY=1` with `PARITY_OUT=<file>`, compared with `parity compare`.
     c. Read its README first, then put the planned calls in.
     d. Smoke run it 3 times, and work out the tolerance its numbers actually
        hold to. `parity derive` does this, and it needs three runs of
        unmodified code.

   Record the result with `workspace.py add --verdict base`, and carry the
   tolerance forward: every later comparison is read against it.

The questions ended with init, but setup can still fail in a way worth handing
back: a runner that does not run, a node that is busy, a tolerance that will not
hold. Say what failed and stop, rather than working around it.

Arm nothing here. What keeps a campaign turning for hours belongs to the caller,
and a diagnosis has no loop to keep turning.
