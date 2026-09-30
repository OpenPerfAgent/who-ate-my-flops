---
name: init
description: Work with the user to pin down which PyTorch job gets optimized, write contract.md, and put the correctness instrumentation in place. The optimize skill later reads the contract.md written here and optimizes on its own. Every piece of alignment with the user happens in init, all of it.
disable-model-invocation: true
---

# Set up an automatic PyTorch performance run

step 0: how much MLSys the user knows, so you can talk to them in their language.
step 1: the repo, the environment, and the script to optimize.
step 2: the instrumentation that keeps the optimization loop correct.
step 3: anything else you and the user find worth preparing.
step 4: init ends.

Write what you prepare into a contract.md, using the template at
`<plugin root>/template/contract.md`, where the plugin root is the directory two
levels above this SKILL.md, which Claude Code also exposes as
`${CLAUDE_PLUGIN_ROOT}`. It records what you and the user agreed on. The user reads
it too, so keep it short.

During init, do not change code logic, do not build a worktree, and do not run
experiments unless the user tells you to. They expect init to finish quickly.

The user has very little attention to spare. Say what you need in the most direct
and shortest wording you can.

Let the user see which step of init they are standing in.

## Step 0: how much MLSys the user knows

Before you ask anything about the job, ask how well the user knows ML systems.
Give them two examples to place themselves against:

- "I am an MLSys engineer. I could do these optimizations myself, I would rather
  not spend the time."
- "I am an algorithm engineer. I do not know systems, and I will not recognise
  most of your optimization techniques."

Anything in between is fine. Write the answer into the `reader` line of
contract.md, in their words.

Write for that reader for the rest of init, and in every report they read
later. An MLSys engineer wants the technique and the number and no explanation.
An algorithm engineer wants to hear what the change does to their training, in
the words they already use: batches, steps, memory, loss. Explain a systems term
the first time it comes up, or leave it out. If you are unsure whether they
followed, ask once; do not check after every message.

## Step 1: the repo, environment, script, and hardware

- Infer the repo, environment, script, and hardware from context first, then have
  the user confirm them. The user is the one who guarantees that the script and
  environment run and that the hardware is free.
- Ask the user for anything you cannot infer, for instance in a fresh session.
- If the checkout is dirty, talk it through with the user. What you decide has to
  fit the worktree layout used later.
- Once the user hands over the script: if it was written for real training, use
  the convert-to-benchmark-script skill to turn it into one that suits performance
  work, and have the user confirm the changes. The `run` line in the contract
  names that converted version.

## Step 2: the instrumentation that keeps the optimization loop correct

Briefly propose a few places to check correctness based on the workload, such
as the loss, model outputs, or tensor shapes. Ask the user: "What else should we
check?" Agree on the check locations before writing the instrumentation plan.

Name the recorder in the plan, and prefer one the repo already has rather than
installing a new one. imp_genai ships `research/core/probe`.

Write the plan into the contract's `## Correctness Instrumentation Plan` section.
That is the section `optimize` reads before it puts the calls in.

Do not modify the code here during the init.

## Step 3: everything else

Depending on the job, there may be more to settle. For example:

1. What changes are allowed? For example: adjusting system configs, enabling
   existing acceleration features with small code edits, installing faster
   backends, changing Python-level workflow code, or modifying Triton/CUDA kernels?
2. Test files that matter and must not break.
3. Hints from the user: is startup slow, or the steady step, or something else?

Write these into contract.md at `## Everything else` section.

They can come from your own reading of the code and then get confirmed, or the
user can propose them.

## Step 4: init ends

Ask the user to read contract.md and confirm that it describes the job they want
optimized. Note contract.md should be written concisely for human to understand (recommend [humanizer tool](https://github.com/blader/humanizer)). Once they confirm, tell them to run
the optimize skill, `/who-ate-my-flops:optimize` in Claude Code and
`$who-ate-my-flops:optimize` in Codex, and the optimization runs on its
own from there. After that point you do not go back to the user with questions.
