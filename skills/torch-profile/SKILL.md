---
name: torch-profile
description: Profile a PyTorch job, training or inference, with torch.profiler. Analyse the profile, read the surface symptoms, chase each root cause down to a file:line, and hand back a diagnosis. This skill produces no patch.
---

How is the time distributed? Where is the bottleneck? What is not the problem?
Every finding gets chased to a file:line.

# torch-profile

- Profile the script at the current commit with `torch.profiler`.
- Add the profiler hooks yourself when the script has none.
- What you produce is **a diagnosis, not a patch**. Those hooks are the only edit
  you make: nothing here changes what the job computes or how long it takes, and
  nothing here measures a fix.

```
① capture                ② views                ③ findings
   one plain run +    →     how the wall      →    ranked, anchored,
   a profiled window        actually splits        costed, handed off
```

## ① Capture

Two runs.

**First, unprofiled**: `torch.profiler` adds overhead, so take the honest clock
with the cheap APIs instead, such as `torch.cuda.Event(enable_timing=True)` and
`torch.cuda.max_memory_allocated()`. This is also the only run that sees startup,
which no profiling window covers: dataset preprocessing, checkpoint loading,
first-step compilation, and anything else paid once. Time every step, and save
both that series and those startup numbers under `commits/<sha>/`.

**Then profiled**, over about 3 steady-state steps, with the window chosen from
the step-time series the first run produced. Steps 3-4-5 is a default, not a law:
if `records/execution_schedule.json` marks a periodic phase, the window must not straddle
it, and if step time trends, the window has to sit in the regime the run actually
spends its steps in. Traces go under `commits/<sha>/trace/`, which
`workspace.py candidate <ws> <sha>` creates.

## ② Views

One profiled step can hold a million events, and nobody reads a million events.
A view is a reshaping that makes one kind of anomaly stand out and hides the
rest. `tools/view/` loads the traces into a store, then renders all five from it,
one numbered file each:

```bash
export PYTHONPATH=<plugin root>   # you are standing in the target, not here
python -m tools.view load   <ws>/commits/<sha>/trace/*.json -o <ws>/commits/<sha>/store.db
python -m tools.view render <ws>/commits/<sha>/store.db -o <ws>/commits/<sha>/views --capacity-gb <GB>
```

`load` wants `rank<N>` in each trace filename. torch.profiler's own export name
has none, so name them at capture time or the rank becomes sort order and every
per-rank row below is labelled with a rank that may not have produced it.
`tools/view/common.py` documents the schema, and each view is also a module with
a `render(con)` if you want one on its own.

1. **Runtime decomposition** (`1_runtime_decomposition.txt`). `wall = busy +
   idle`, per step and per rank, with the spread across steps and the slowest
   rank. Read it first: it forks the whole diagnosis into waiting against busy
   but inefficient.
2. **Lost time, ranked** (`2_lost_time.txt`). Every gap where nothing ran and
   every run of tiny kernels, ordered by what it costs in wall clock, each
   stamped with its offset from the step start so the same one lines up across
   steps. Each one arrives with what the machine was doing instead, which you
   then name as an entry in `records/execution_schedule.json` rather than as a symptom.
3. **Kernel table** (`3_kernel_table.txt`). Cumulative time bucketed into
   attention, GEMM, communication, element-wise and transfer, with bytes moved
   and achieved bandwidth wherever the event carries a size.
   Copies are in here on purpose: a device-to-host copy counts as GPU-busy, so it
   sits in no gap, and it disappears from every view at once if this one drops
   it.
4. **Overlap and exposure** (`4_overlap_exposure.txt`). Per collective, how much
   hid behind compute and how much was exposed. Only the exposed part lengthens
   the step, so a large communication share that is fully hidden is not a
   finding.
5. **Memory over time** (`5_memory.txt`). Allocated and reserved, the peak, and
   the headroom per GPU. Nothing else says whether a fix that spends memory is
   available at all. Pass `--capacity-gb` or the headroom column comes back
   unknown.

Read all five before you conclude anything. They are where to start rather than
the limit: the store is an ordinary database, so query it directly for whatever
this particular job makes you suspect.

## ③ Findings

Every finding carries all of these, or it is not one yet:

| | |
|---|---|
| evidence | the number, and where in the views it came from |
| `code_anchor` | out of `records/computation_graph.json`, or the `records/execution_schedule.json` entry for a wait |
| root cause | why the code does this, not what the device did |
| cost | as a **ceiling**, and against the whole run rather than one step |
| in scope | whether the reader is allowed to edit that code |
| confidence | and what would raise it |

Include the negative results. Half of what a reader needs is the list of things
that looked expensive and were not.

Write it to `commits/<sha>/report.md`. Inside `optimize`, that report is what the
loop reads to decide whether the candidate is kept.

## Output contract

Three sections, in this order, and nothing may appear in section 3 that was not
established in sections 1 and 2:

1. **Where the time went.** How this run divides into working and waiting, for
   the whole job *and* for a single step.
2. **Findings.** Ranked by how much wall clock each one actually costs, meaning
   the part not already hidden behind other work.
3. **Prescription.** Directions, each with a what-if bound stated as a
   **ceiling**, the effort it takes, whether numerics would change, and whether
   the code is the reader's to edit. **No predicted numbers for fixes you did not
   run.** Whoever runs the experiment writes the prediction.

