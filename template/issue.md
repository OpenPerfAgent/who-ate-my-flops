<!-- Title: <what was observed>, with the job and the GPU, e.g. "Silently slower attention kernel when training MiniMax-H3 LoRA on B200". No speed-up factor, no prefix of your own. If the repo has issue forms, pick the one that fits (performance or question before bug), keep its headings and put the paragraphs under the ones that fit. -->

I ran <the script or recipe> (`<path in repo>`, <mode or stage>, <world size>) on <GPUs>, following <the install doc>, and got <t0> s per <step / generation>. A profile showed <the single largest cause, with the file, in one sentence>. With <that one thing changed>, the same <step> took <t1> s.

<What the logs or docs said, or did not say, at the time, in one or two sentences.> <The one line that would have made the profile unnecessary: a log line, a doc note, a number in an existing warning> would have saved me the profile.

I opened #<PR> to record my experiment: setup, measurements, traces and some potential fixes. A different fix may also well suit the codebase.
