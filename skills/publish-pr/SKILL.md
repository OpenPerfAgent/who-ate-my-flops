---
name: publish-pr
description: Write the pull request description and one issue for a finished campaign into proposed-pr-description.md and proposed-issue.md, for the user to review and post. Says what the PR may claim, what goes in each section and in what order, and how it is worded. Fills template/pr-description.md and template/issue.md; the filled examples are the .example.md files next to them. It pushes nothing and opens nothing.
---

# Draft the PR description and the issue

In: a campaign that `clean-up` has delivered (delivery branch, `benchmark.csv`, `records/candidates.jsonl`,
`commits/<sha>/{report.md,trace/}`, `latest-report.md`), and the target repo checkout.
Out: `proposed-pr-description.md` and `proposed-issue.md` next to `latest-report.md`. The user uploads traces,
pushes, opens the PR and files the issue.

The reader is a maintainer who has never heard of you, the campaign or its tools. They open the PR wanting three
answers, in this order: what was wrong, what this PR changes, and what it costs or risks. Everything else is
evidence, and evidence goes lower down or into a collapsed block. The filled example is
`template/pr-description.example.md` (https://github.com/modelscope/FunASR/pull/3705).

## 1. Decide what the PR may claim

Before writing, sort every shipped change:

- **Install or setting a user can apply alone** (a pip package the code already falls back to, an env var, a
  launcher flag, a config value): not the PR's contribution. Apply it to both arms of the measurement and name it
  in Details, with the rest of the measurement setup. The speedup the PR claims is measured on top of it.
- **A default the repo should ship for its users** (users of the recipe cannot be expected to know the setting):
  this is a contribution. Keep it as code, behind an explicit option that is off for every other caller, and turn
  it on in the recipe.
- **Source changes**: the contribution.
- **Removed code**: before claiming a removal is safe, `git blame` it and read the PR that added it. Code that
  looks useless often guards a case the benchmark does not exercise (mixed data, other devices, other launchers).
  If you cannot show the case is covered, do not ship the removal.

Every number comes from `benchmark.csv`, `records/candidates.jsonl`, `commits/<sha>/report.md`, `latest-report.md` or a
re-verification run. Nothing predicted, nothing remembered, nothing carried over from an earlier baseline without
saying so.

## 2. The description

Fill `template/pr-description.md`. The description has three parts of its own, then whatever the repo's template
asks for, then a collapsed block. Nothing else is added by default.

1. **Summary**. One paragraph: what the job loses and why, with the one or two numbers that show it; what this PR
   adds; the headline result and the hardware it was measured on. Then a numbered list, one item per change, each
   at most three sentences: the problem, what the change does, and its limit or how to turn it off. No commit shas,
   file lists or line counts here.
2. **Speedup Result** (a `###` under Summary). First line: `Measured on <n> <GPU>.` Then one table,
   `| | baseline | this PR | change |`: the headline metric (mean and median if the step time is bimodal), what
   disappeared (slow steps, launches, stalls), peak memory, the one-time cost (first step, compile), and one
   `of which:` row per change when they were measured separately. When there are several commits with their own
   measured effect, add the per-commit table `| commit | change | measured effect |`, and mark numbers that were
   measured against a different baseline. Nothing after the tables.
3. **Correctness Verification**. Written for someone who does not know the recording tool. First paragraph: what
   you ran (both versions on the same batches), which values you compared and how many, that the tolerances were
   set before the runs, and where the recording code lives (a branch of the fork, not in the PR). Do not name the
   tool as if the reader knew it; no "parity", "hooks", "arms", "gate", "checkpoint". Second paragraph: if any value
   exceeds its tolerance, the first sentence says so plainly, "<k> of the <N> checks fail the tolerances set before
   the runs, so numerical equivalence has not been shown", then lists them with their numbers, then what matched.
   A suspected cause (bf16 rounding, a different kernel) is labelled as a guess and does not turn a failure into an
   accepted result; only an explicit, justified acceptance decision does that. If nothing exceeds, lead with that.
   Then the table, `| recorded | baseline vs this PR | tolerance |`, end-of-run rows first (validation loss, final
   weights), per-step rows after, each exceedance counted inside its cell ("2 steps above 0.01"). No pass/fail
   column, no bold.
4. **The repo's own template**, after Correctness Verification. Read `.github/PULL_REQUEST_TEMPLATE.md` (and
   `CONTRIBUTING.md` for required statements) and add its sections in its order, with its headings and checkboxes.
   Skip any heading whose content is already covered by the three parts above (a "Summary", "Purpose" or "What does
   this PR do?" heading is the Summary; a "Test Result" heading is Speedup Result plus Correctness Verification).
   Fill the rest truthfully and briefly, using these rules where the heading asks for them:
   - tests or validation: the test files and what they assert in a few words, pass counts with and without a GPU,
     lint or compile checks, docs touched; the measurement setup is in Details, so do not repeat it;
   - user impact: every caller that can reach the change (an option read by a constructor affects anyone who sets
     it, not only the recipe that sets it), what stays the same by default, the cost and how to turn it off, the
     speedup only for the hardware measured;
   - notes for reviewers or risks: short bullets, each something not said above;
   - checkboxes: an unchecked box with a short reason beats a ticked box you cannot back.
   If the repo has no template, add nothing here.
5. **Details**, collapsed: `<details><summary>Details: hardware, model, full command, traces</summary>` with
   hardware and environment, the model and job, data preparation, the measurement setup (base commit, what both
   versions share, data size, seed, number of interleaved runs), the full command for both versions, settings tried
   and rejected, and the trace links (`<share link>` placeholders plus the local `commits/<sha>/trace/` folder until
   the user uploads). `</details>`.

Title on the first line as a level-one heading: `[perf] <job> on <hardware>: <t0> s to <t1> s per <unit> at
<world size>, mostly from <largest cause>`, following the repo's title rule if it has one.

Not in the description: sections the repo's template does not ask for (no default Validation, User impact or Notes
for reviewers), an "experimental PR" callout, a revision note or history of earlier versions, a "generated with" footer or session link (unless the repo requires an
AI-assistance statement), an Observations essay (profile findings go into the issue), campaign vocabulary (baseline
arms, candidates, exp names), test function names, the same number in two sections.

## 3. The issue

Fill `template/issue.md` into `proposed-issue.md`: one issue per campaign, on the observation that most needs to
reach the maintainers whether or not the PR is merged, usually the largest cost or the one invisible in the logs.
Three paragraphs: what I ran and what the profile showed as the largest cause, with the file and the number
before and after that one change; what the logs or docs said at the time and the one line that would have made the
profile unnecessary; `I opened #<PR> to record my experiment: setup, measurements, traces and some potential fixes.
A different fix may also well suit the codebase.` Title names what was observed with the job and the GPU, no
speed-up factor, no prefix of your own. If the repo has issue forms, pick a performance or question form before a
bug form and put the paragraphs under its headings.

## 4. Wording

Run the humanizer skill over both bodies. Then check:

- Problem before mechanism, result before method, what matched before what did not.
- Plain first person for the experiment ("I trained both on the same 382 batches"), no review of the repo: no
  "wrong", "ignores", "should", "must". Fixes are offered, not prescribed.
- Calm but not softened: state a failed check as failed, once, with its size, in neutral words. Softer tone
  never removes the fact; a maintainer who asked for failures to be reported will read a softened sentence as a
  retreat.
- Scope claims exactly as far as they hold: a context manager that sets process-wide flags is not "model-local";
  one B200 measured is not "sm_90 and sm_100 GPUs" or "on GPUs"; a hardware explanation for a speedup does not
  establish the same speedup on other hardware; an option read by a constructor is not "only for recipe users".
  Put the measured hardware next to every speedup number: in the Summary, the Result heading line, User impact and
  any pay-back estimate.
- Short sentences, active voice, no filler, no sales words, no em or en dashes, bold only on the numbered change
  names and the headline speedup.
- Cut any sentence a maintainer would not miss.

## 5. Report back

Paths of both files, both titles, the placeholders still in them (trace links, the fork's verify branch, the PR
number in the issue), and anything you could not verify.
