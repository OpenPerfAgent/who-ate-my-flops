<!-- Filled example of template/issue.md: https://github.com/modelscope/DiffSynth-Studio/issues/1687, filed next to PR #1680. -->

# Silently slower attention kernel when training MiniMax-H3 LoRA on B200

I ran the repo's MiniMax-H3 LoRA training example (`examples/minimax_h3/model_training/lora/MiniMax-H3-FL2VA.sh`, stage 2) on 8x B200, following the installation README, and got 6.32 s per training step. A profile showed attention running on the FA2 kernel, which is an sm80-era design and much slower on this GPU than torch SDPA's cuDNN backend. With the kernel switched, the same step took 3.80 s. Nothing in the logs said which implementation had been picked, so I only found it by profiling.

Setting `DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch` avoids this. A line in the install docs, or a one-line log of the chosen implementation at import, would have saved me the profile.

I opened #1680 to record my experiment: setup, measurements, traces and some potential fixes. A different fix may also well suit the codebase.
