# [perf] <job> on <hardware>: <t0> s to <t1> s per <unit> at <world size>, mostly from <largest cause>

<!-- Order: Summary with Speedup Result, Correctness Verification, then the repo's own .github/PULL_REQUEST_TEMPLATE.md sections in its order (skip any already covered above; nothing if the repo has no template), then Details. -->

## Summary

<What the job loses and why, with the one or two numbers that show it.> <What this PR adds.> <Headline: on the <n> <GPU> I measured, t0 to t1 (N x); other hardware not measured.>

1. **<change 1>**. <The problem.> <What the change does.> <Its limit or how to turn it off.>
2. **<change 2>**. <The problem.> <What the change does.> <Its limit or how to turn it off.>

### Speedup Result

Measured on <n> <GPU>.

| | baseline (`<base sha or main>`) | this PR | change |
|---|---|---|---|
| <headline metric> | <t0> | <t1> | **<N>x** |
| <what disappeared: slow steps, launches, stalls> | <x> | <y> | |
| peak memory | <m0> | <m1> | |
| <one-time cost: first step, compile> | <a> | <b> | |
| of which: <change 1> alone | | <t> | <k>x |
| of which: <change 2> on top | | <t1> | a further <k>x |

<!-- Several commits with their own measured effect: add the per-commit table. Mark numbers measured against a different baseline. -->

| commit | change | measured effect |
|---|---|---|
| `<sha1>` | <one line> | <t> |

## Correctness Verification

I trained the baseline and this branch on <the same data> and compared <N> values from each run: <which values>. The tolerances in the table below were set before the runs. The recording code is on the `<branch>` branch of my fork, not in this PR.

<If any exceed:> <N-M> of the <N> checks fail the tolerances set before the runs, so numerical equivalence has not been shown. The <N-M> are <list with numbers>. The other <M> are within tolerance, and <end-of-run numbers: validation loss a vs b, accuracy identical>. <Suspected cause, labelled as a guess, and that it does not make the failures acceptable by itself.>

| recorded | baseline vs this branch | tolerance |
|---|---|---|
| <end-of-run value, e.g. validation loss and accuracy> | <a vs b> | <tol> |
| <final weights> | <diff> | <tol> |
| <per-step value> | <mean, max; k steps above tol> | <tol> |
| <shapes and counts> | identical | exact |

<!-- The repo's own template sections go here, in its order, with its headings and checkboxes. Example of what a template may ask for: -->

## <template heading, e.g. Type of change>

- [x] <box that applies>

## <template heading for tests, e.g. Validation / Test Plan>

- [x] `<test command>`: <what it checks>, <pass counts with and without a GPU>
- [x] `<lint or compile check>`

## <template heading for impact, if asked>

<What stays the same by default.> <Every caller that can reach the change, not only the recipe.> <The speedup only on the hardware measured.> <The cost and how to turn it off.>

<details>
<summary>Details: hardware, model, full command, traces</summary>

**Hardware.** <GPU, arch, memory, driver, CUDA, torch; pinning; where data lives>.

**Measurement.** Against the unmodified `<base sha>`: <script as shipped>, <data size>, seed <s>, <k> interleaved runs per version, <what both versions share, or nothing set in the environment>.

**Model.** <architecture, sizes, precision, what is trained, optimizer, per-step input>.

**Data** (once): <how it was prepared>.

**The measured job**, both arms:
```
<full command>
```
<Notes on flags that need one.>

**Settings tried and rejected.** <list, with why>.

**Traces** (<profiler, window, ranks>; open in https://ui.perfetto.dev):
- baseline: <share link> (local: `commits/<base sha>/trace/`)
- optimized: <share link> (local: `commits/<head sha>/trace/`)

</details>
