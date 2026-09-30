---
name: convert-to-benchmark-script
description: Turn the script a user really runs into a benchmark script an optimization loop can run over and over. Same shape as the real job, a few minutes long, repeatable, and leaving nothing behind. It produces the script and a table of what changed, and it does not run it.
---

# convert-to-benchmark-script

In: the script the user actually runs. 

Out: a new script that does the same job within a few minutes (mostly by reducing total steps), gives the same answer twice, and leaves nothing behind.

## Four rules

1. **It has to be the same job.** Same model size, batch size, sequence length,
   precision, parallelism, and world size etc. Shrink any of them and the bottleneck
   moves somewhere else, and then the loop spends a day optimizing the wrong
   thing.
2. **It has to be short.** Aim for one run within 8 minutes. Every candidate in
   the loop pays this cost, so a 20 minute benchmark is a 20 minute tax on every
   experiment. Short is a budget, not an instruction to cut: if the job already
   fits, leave it whole.
3. **It has to leave nothing behind.** No runs on someone's dashboard, no
   checkpoints eating a quota, no writes into the checkout.
4. **It has to give the same answer twice.** Pin every seed and every source of
   ordering. Use the script's own flag where it has one, and otherwise seed from
   your wrapper before the workload starts. `optimize`'s setup smoke-runs it 3
   times and reads the tolerance off those runs, so drift you leave in here comes
   back there as a tolerance too wide to gate anything on.

## What to turn off

Reach for the script's own flag or an environment variable every time. A deleted
line is a change to their code that nobody agreed to.

**Estimate the cost before you switch anything off.** You cannot measure it here,
so do the arithmetic on paper:

- a checkpoint: `bytes on disk / write bandwidth x saves per run`
- an eval pass: `eval batches x 1/3 of a train step / steps between evals`

Divide either by the run's total wall clock. Under a few percent, turn it off.
Above that, either measure it as a phase of its own rather than buried in
steady-state step time, or turn it off and record the exclusion, with your
estimate, in the contract's **Anything else** row. An exclusion nobody wrote down
becomes a speedup that never shows up in the real job.

| | how | why it matters |
|---|---|---|
| trackers: wandb, tensorboard, mlflow | the script's own flag, or `WANDB_MODE=disabled` | the loop makes hundreds of runs, and every one of them lands on a real dashboard |
| checkpoint writes | the script's own flag | the loop runs hundreds of times, so price the disk as well as the clock before you leave it on |
| eval or validation loops inside training | flag them off, or measure them as their own phase | otherwise every Nth steady-state step has an eval buried in it |
| uploads and notifications: hub pushes, slack, mail on finish | flag | same as trackers, and harder to take back |
| the output directories the script picks for itself: `./outputs`, `./wandb`, `./lightning_logs`, `./checkpoints` | route them all through one flag or environment variable | the workspace does not exist yet during `init`, so `optimize` points that flag at its `runs/` and nothing lands in a candidate commit |

## How much to run

**If the whole job fits the budget, run it whole.** Judge that on the real job: a
5 minute smoke config belonging to a 10 hour run is still the 10 hour run's
benchmark.

If you have to cut, keep both phases: 
- the warmup, paid once. 
- the steady step, paid many times. 

Run enough warmup steps that step time stops changing, then at least 5 to 10
steady steps after it. With fewer, the profiler has nowhere to place its capture
window, and a real speedup cannot be told apart from run-to-run noise.

### Cutting the dataset

You are allowed to use a smaller dataset where it cuts overhead. 
It moves the numbers as well as the clock, starting with dataset initialization time, so record what you cut.

## Two modes

The mode sets the step count. It does not relax anything under "the same job".

| | speed | accuracy |
|---|---|---|
| what it has to show | warmup and steady state, told apart | the quantities the contract's **What must not change** row names |
| steps | the floor above. More buys nothing | what the check needs: bit-exact settles in a few steps, a convergence check cannot be short |
| determinism | seeds fixed, `torch.use_deterministic_algorithms` left off because it costs speed | seeds fixed, data order fixed, deterministic algorithms on when the check is bit-exact |

## Hand it back

Show the user a table first: what you changed, the flag or environment variable
it maps to, and why. Anything you had to leave alone because fixing it would mean
editing their code goes in the same table, as a limitation. Then have them
confirm it.

Keep it short. Reading it costs the user attention, and attention is the scarce
thing in `init`. 