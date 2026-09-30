---
name: diagnose
description: Profile the job defined in contract.md, name the one thing it loses most time to, and propose a single fix with a measured number behind it. It stops at the first finding. optimize is this same work repeated until nothing worth doing is left.
disable-model-invocation: true
---

diagnose is a sequence of steps. Work through them one at a time and mark each
one done as you go.

It stops after one finding, one fix and the measurement that proves it. The user
chose this over `optimize` because they want an answer in an hour rather than a
branch in a day, and a second finding they did not ask for costs them that hour.

# Step 1: setup

Use the setup-baseline skill. It builds the worktree and the workspace, has
extract-computation-graph write `records/computation_graph.json` and
`records/execution_schedule.json`, then measures the base commit for speed, memory and
correctness, profiles it in full with torch-profile, and records it with
`workspace.py add --verdict base`. A workspace left in that shape is one
`optimize` can pick up later without profiling the base again.

The tolerance it derives is what step 3 reads the fix against. Without it a fix
has no number for whether the job still computes the same thing.

Do not arm anything. The registry and the Claude Code
watchdog keep a campaign turning for hours, and this is one pass. They belong to
`optimize`.

# Step 2: the one finding

setup-baseline left the traces under `commits/<base sha>/trace/`, the rendered
views beside them and the diagnosis at `commits/<base sha>/report.md`. This step
reads them and picks one.

Read all five views before naming anything. "The largest" is a claim about the
other four, so ranking them is what earns it.

Take the top finding by what it costs in wall clock: the part not already hidden
behind other work, multiplied by its repeat count from `records/computation_graph.json`.
Do not rank by kernel time. A collective that is fully hidden costs nothing, and
a 2 ms stall on every one of forty blocks outranks a 60 ms one-off.

The finding carries everything torch-profile's ③ asks for. The `code_anchor` and
the cost ceiling are the two the user reads first.

# Step 3: the one fix

Pick the fix by optimize's Tradeoffs: architecture, breadth, memory, precision.

Then prove it, on the same benchmark script and the same node:

- an interleaved timed pair against the base, so the two arms see the same
  machine
- the recorder's comparison against the baseline record, at the tolerance
  derived in step 1

Record it with `workspace.py add`, with the tradeoffs in the `--why` text.

A fix that does not hold still answers the question. The finding stands, the fix
did not, and the number says by how much. Write that down rather than going after
a second finding to rescue the run. Another fix for the same finding is in scope;
the next finding is not.

# Step 4: closing

Write `latest-report.md` for the `reader` named in contract.md, as short as you
can, carrying a `status: closed` line of its own:

- where the time goes, for the whole run and for one step
- the finding: what it is, its `code_anchor`, what it costs as a ceiling
- the fix, what it traded, and the before and after on the hardware it was
  measured on
- the correctness result, including any check that exceeded its tolerance
- what looked expensive and was not
- the rest of the ranked findings, one line each, and that `optimize` works
  through them

Write `records/mistakes.md` as you go, in the same turn you get something wrong.

`workspace.py finish` is `optimize`'s loop condition and this is not a loop, so
do not run it. Stop here. The user decides whether to run publish-pr for an
issue and a PR, or optimize to keep going.
