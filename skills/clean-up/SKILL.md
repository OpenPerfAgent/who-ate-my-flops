---
name: clean-up
description: Deliver a finished optimization run as a branch the user can merge. Cherry-pick the commits that changed the workload onto the base, hardcode the A/B flags, re-verify on the shipped code, strip the checkpoint calls in a last commit, and leave the user standing on that branch with a handover.
---

# Deliver the branch

The campaign branch is a record, not something to merge.

1. **Tag it, then cherry-pick the survivors onto the base it ran on.**
   `git tag <campaign>-full-history HEAD`. Never rewrite the campaign branch:
   `records/candidates.jsonl` references its shas, and every check below compares against
   it.
2. **Cherry-pick, do not rebase and drop.** A rejected candidate and its revert
   cancel out in the net diff, but a replay still applies each one.
3. **Ship only what changed the workload**, plus the checkpoint calls, which
   step 6 removes again. Runners and capture harnesses stay behind, and their
   commands go into the reproduce block instead. Anything in the shipped source
   that imports a file outside the checkout is a harness, whatever it is called.
4. **Delete the A/B flags before you measure**, not on both sides of the
   deletion, and as one commit of its own. Watch for an `else` that is another
   caller's entry point, and the `import os` only the flag needed.
5. **Run the recorder comparison here**, on the head with the flags hardcoded
   and the checkpoint calls still in, against the setup tolerance. This is the
   code that ships plus the calls that record it, so name it:
   `git branch <delivery>-verify HEAD`. A maintainer who wants to reproduce
   the correctness table checks that branch out.
6. **Strip the checkpoint calls in one last commit.** Pure deletion, nothing
   else in it: the recorder's import, its calls and any file that exists only
   for them. Maintainers review the speed-up, not the instrumentation, so the
   delivery branch carries none. If the repo's own recorder was used, there is
   nothing to strip and no verify branch; say so in the handover.
7. **Time what ships.** An interleaved timed pair on this head, with
   `git checkout <base> -- <src dirs>` for the before arm. If something goes
   wrong, fix it, and redo step 5 if the fix touched the model's math.

Rename the branch to what you want merged and leave it checked out.

Move stray files out of the workspace root and update any affected paths or commands.

## Handover report

Write the full handover in `latest-report.md` for the `reader` named in
contract.md. Put yourself in their position and explain the results in terms
they can understand. Include:

- the branch, its base, and that merging is theirs
- the verify branch: the delivery branch before its last commit, so the same
  code with the checkpoint calls, and the command that compares a run of it
  against the baseline record
- where the time went before, pointing at the baseline's diagnosis under
  `commits/<base sha>/report.md`, and which of its findings each shipped
  technique removed
- the speedup in the unit they plan in, with the GPU model and count
- a table of techniques and their numbers, from `benchmark.csv`
- how to reproduce it without the scaffolding
- the correctness evidence

Point at `records/candidates.jsonl` and the tag for what was rejected, rather than
transcribing it.

## Final message to the user

Give the user a short summary of the outcome, the main changes, and whether
the agreed checks passed. Highlight any unresolved issue or decision they
need to make, and link to the delivery branch and `latest-report.md`.

Keep detailed measurements, rejected candidates, and routine debugging
history in the report; do not repeat the full handover in this message.

Stop at the branch. What happens to it next is the user's to decide.
