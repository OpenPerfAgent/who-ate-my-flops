# Campaign contract

Define the workload you want optimized here.

---

## Required

```
reader          <who reads the reports, in their words: e.g. "MLSys engineer, could do this myself" or "algorithm engineer, new to systems">
repo            <path to the local checkout, and the branch>
activate        <the command that gets you into the environment>
run             <the command that runs the job>
node            <the machine to measure on. It has to be empty.>
```
---

## Correctness Instrumentation Plan

TBD

<!-- Name the recorder and where its checkpoints go. Prefer one the repo already
has: imp_genai ships research/core/probe. -->

---

## Everything else

TBD

<!--
Below are example questions and answers. Write a table in the same format.

| | what goes here |
|---|---|
| **Allowed changes** | Allowed changes: adjusting system configs, enabling existing acceleration features with small code edits, installing faster backends, changing Python-level workflow code, or modifying Triton/CUDA kernels. |
| **Your own tests** | The command that runs this repository's suite. It proves the *rest* of the repo still works; the row above only proves this job does. A change to something shared needs both to pass. |
| **Hands off** | Files or subsystems it must not touch · changes that are unacceptable however fast they are · whether it may trade memory for speed, and how much |
| **When to stop** | A target, such as `-15% would do` or `under 8 s a step`, and what to do once it is hit: hand back, or keep going and say so |
| **Anything else** | Data that lives somewhere awkward · a flag that has to be set · a config to copy rather than edit · a colleague who owns half the file. Prose is fine. | 
-->
