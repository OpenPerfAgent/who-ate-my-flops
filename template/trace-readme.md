# <repo> — <job>: torch.profiler traces, before and after

Two captures of the same step, same node, same window. Open any `trace.rank<N>.json` in https://ui.perfetto.dev.

| | `baseline/` | `optimized/` |
|---|---|---|
| code | upstream `<org/repo>` `<branch>` @ `<base sha>` (+ measurement hooks only) | `<base sha>` + the commits of <fork PR link> (shas `<...>`) + the same hooks |
| steady step (steps <a> to <b>, median, CUDA events, rank 0) | <t0> s | <t1> s (−<p>%) |
| GPU busy / idle per step (from the trace) | <busy%> / <idle ms> | <busy%> / <idle ms> |
| kernel time by class, per step per rank | <class: s, ...> | <class: s, ...> |
| peak GPU memory per rank | <m0> GB | <m1> GB |
| trace files | `trace.rank0..<n>.json`, <size> each (≈<k> kernels per step) | `trace.rank0..<n>.json`, <size> each |

This folder holds only the traces. Diagnoses, rendered views and the unprofiled timed runs are in the campaign
workspace: `<path>/ws/commits/<sha>/{report.md,views/,steptimes/}`.

## The job

<script path>, <mode/stage>, flags as in the script except <what>:

```
<command with placeholders>
```

- Model: <spec>. <what is trained>. <parallelism>.
- Input per step: <shape / tokens>. <loss>.
- <steps per run>; seeds <how>.

## Hardware and software

- <node>: <n> × <GPU> (<arch>, <mem>), driver <x>, CUDA <y>.
- <framework and versions>; <anything built from source>; <repo commit>.
- <where weights/data were read from>.

## How the traces were captured

- `torch.profiler.profile(activities=[CPU, CUDA], schedule=schedule(wait=<w1>, warmup=<w2>, active=<w3>, repeat=1), record_shapes=False, profile_memory=False, with_stack=False)`, `prof.step()` after each optimizer step, one `export_chrome_trace` per rank. Each file holds steps <c> to <d>, from the steady-state region.
- Step times above come from a separate unprofiled run (CUDA events per step), not from the profiled run.

## Reading notes

- <kernel names to look for on the baseline side and what they are>.
- <kernel names on the optimized side>.
- <rank agreement; which rank is representative>.
