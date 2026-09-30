<!-- Filled example of template/pr-description.md: https://github.com/modelscope/FunASR/pull/3705 as of 2026-09-16, reordered to this skill's section order (Summary, Correctness Verification, then FunASR's template sections, then Details); the live PR keeps the template order. -->

# [perf] Fun-ASR-Nano fine-tuning on B200: 0.244 s to 0.068 s per step at 1 GPU, mostly from cuDNN SDPA plan builds on every new batch shape

## Summary

Fine-tuning Fun-ASR-Nano with `finetune.sh` on a B200 loses most of each step to two things: torch routes the Qwen3 attention to cuDNN, which recompiles its execution plan for every new batch shape (about 1 s, on one step in six), and the rest of the step is launch-bound (8,080 kernels for 37 ms of GPU work). This PR adds two opt-in switches under `llm_conf`, both turned on in `finetune.sh`: `sdpa_backends` runs the LLM forward with the flash, efficient or math attention kernels instead of cuDNN, and `torch_compile` compiles the decoder stack. On the one B200 I measured, step time goes from 0.244 s to 0.068 s (3.6x); I have not measured other GPUs. Here are the two opt-in switches:

1. **`llm_conf.sdpa_backends`**. On the B200 (sm_100) I measured, torch sends the Qwen3 attention to cuDNN because it carries a padding mask, and cuDNN rebuilds its execution plan for every new batch shape, about 1 s each, on one step in six. Setting it to `[flash, efficient, math]` runs the decoder forward under `sdpa_kernel` with those backends, which need no per-shape plans. The flags are process-wide while the forward runs and are restored on return, so leave it unset when several models share one process.
2. **`llm_conf.torch_compile`**. Once the plan builds are gone, the step is launch-bound: 8,080 kernels for 37 ms of GPU work, 2,900 of them in the decoder. Setting it wraps the decoder stack in `torch.compile(dynamic=True)`, which halves the launch count. Inputs on the CPU and single-sequence batches take the eager path, and the first step pays a one-time compile.

### Speedup Result

Measured on one B200.

| | baseline (`main`) | this PR, `finetune.sh` overrides on | change |
|---|---|---|---|
| step time, mean of steps 2 to 382 | 0.2429 s (0.2404, 0.2453) | 0.0680 s (0.0685, 0.0676) | **3.57x**, −72.0 % |
| step time, median | 0.0960 s (0.095, 0.097) | 0.0645 s (0.065, 0.064) | 1.49x |
| steps over 0.5 s (cuDNN plan builds, 1.0 to 1.13 s each) | 59 of 381 | 0 | |
| kernel launches per step, steps without a plan build | 8,080 | 4,530 | |
| peak allocated memory | 7.39 GB | 6.82 GB | |
| step 1 | 2.2 s | 64 s (warm Inductor cache), 157 s (cold) | one-time compile |
| training phase, 382 steps | 93.8 / 95.7 s | 89.7 / 90.7 s including the compile | |
| of which: `sdpa_backends` alone (two runs, same day and setup) | | 0.0996 s mean (0.1019, 0.0974), 0.0965 s median, 0 slow steps | 2.44x on the mean |
| of which: `torch_compile` on top | | 0.0680 s mean, 0.0645 s median | a further 1.46x |

## Correctness Verification

I trained the baseline and this branch on the same 382 batches and compared 6,128 values from each run: per-step loss, logit statistics and correct-token counts, parameter and gradient norms at seven steps, the final validation loss, and every batch shape. The tolerances in the table below were set before the runs. The recording code is on the `perf/fun-asr-nano-training-speed-verify` branch of my fork, not in this PR.

23 of the 6,128 checks fail the tolerances set before the runs, so numerical equivalence has not been shown. The 23 are two per-step losses at 0.0128 and 0.0115 against a 0.01 limit, two gradient norms at 2 % against 1 %, and a one-token change in the correct-token count on 11 steps. The other 6,105 are within tolerance, and the final validation loss is 0.101834 on the baseline and 0.101967 on this branch with identical accuracy. My guess is bf16 rounding, since the flash and memory-efficient kernels and the compiled decoder round differently from eager cuDNN, but I have not verified that, and it does not make the failures acceptable by itself.

| recorded | baseline vs this branch | tolerance |
|---|---|---|
| validation loss and accuracy, 200 dev utterances | loss 0.101834 vs 0.101967; accuracy identical | 0.001 |
| trainable-parameter norm, 7 steps | 3.4e-9 relative | 1e-4 |
| logits mean and std, every step | 0.16 % mean, 0.67 % max | 1 % |
| per-step loss, 382 steps | 0.0014 mean, 0.0128 max; 2 steps above 0.01 | 0.01 |
| correct-token count, 382 steps | 3 of 34,886 tokens net; 11 steps above 1 % | 1 % per step |
| gradient norm before clipping, 7 steps | 0.8 % or less on 5 steps, 2.0 % on 2 | 1 % |
| batch shapes and frame counts | identical | exact |

## Type of change

- [ ] Bug fix
- [ ] Documentation
- [ ] Example or demo
- [ ] Runtime or deployment
- [ ] Benchmark or evaluation
- [x] Model/training change

## Validation

- [x] `python -m pytest tests/test_fun_asr_nano_train_opts.py`: 4 cases x 2 model copies: 8 passed with a CUDA device, 6 passed / 2 skipped (the CUDA-only case) with `CUDA_VISIBLE_DEVICES=""`
- [x] `python -m compileall funasr examples tests`
- [x] `tests/test_fun_asr_nano_lora_injection.py`, `test_fun_asr_nano_vllm_dtype.py`, `test_fun_asr_nano_autocast_device.py` still pass (14)
- [x] Docs: `examples/industrial_data_pretraining/fun_asr_nano/docs/finetune.md`, new "Training-speed options" section
- [ ] Runtime/deployment command tested (training-only change)

## User impact

With the defaults the model is built exactly as before, and the new tests check that for both model copies. The two keys are read by both model constructors, so any caller that sets them gets the switches; `finetune.sh` is the recipe that sets them. The speedup above was measured on one B200 only. The cost is a one-time compile on the first step, about a minute with a warm Inductor cache and 2.5 minutes cold on that B200. Removing either key turns that switch off.

## Notes for reviewers

- The review comments of 2026-09-13 and 2026-09-15 are addressed: both switches are opt-in and off by default, the attention setting is described as process-wide while the forward runs, and the tests cover the defaults and the enabled path for both model copies.
- Changing batch shapes caused no recompiles in 382 steps; single-sequence batches run eagerly.
- On that B200 the compile pays for itself after about 2,000 steps with a warm cache and 5,000 cold, well within the recipe's 50 epochs.
- `TORCH_CUDNN_SDPA_DEPRIORITIZED=1` gives the same result as `sdpa_backends` without editing the recipe. The profile is in #3704.
- Not tested: DeepSpeed (`use_deepspeed=true`) and more than one GPU.
- Left out of this PR: the frozen encoder runs with dropout on every step, and AdamW keeps the LLM's optimizer state in bf16.

<details>
<summary>Details: hardware, model, full command, traces</summary>

**Hardware.** 1 × NVIDIA B200 (sm_100, 183 GB) on an 8-GPU node, driver 580.126.20, CUDA 12.8 (the torch build), cuDNN 9.19, torch 2.11.0+cu128; the process pinned with `numactl --cpunodebind=0 --membind=0` to the NUMA node of GPU 0. Weights: the ModelScope snapshot of `FunAudioLLM/Fun-ASR-Nano-2512` on local disk; AISHELL-1 wavs on node-local disk.

**Measurement.** Against the unmodified `486b4b7ce`: `finetune.sh` as shipped (on this branch, with its two new `++llm_conf.*` lines), 3600 AISHELL-1 utterances = 382 steps, seed 1234, two interleaved runs per version, nothing set in the environment.

**Model.** SenseVoiceEncoderSmall (70 SANM layers, 221 M, frozen, fp32 with TF32 matmuls) + Transformer adaptor (12.6 M, frozen) + Qwen3-0.6B (28 layers, 596 M, bf16, trained) + a CTC decoder (39 M, trainable, not called in training); 868.86 M parameters, 635.12 M trainable. AdamW lr 2e-4, 2500 warm-up steps, grad clip 5. Each step is a token-budget batch of 6000 (`speech_length + text_length`) with at most 10 utterances: on average 9.6 utterances, 640 LLM tokens and 722 fbank frames, with a new `(batch, padded length)` pair on about 1 step in 6.

**Data** (once): the recipe's `tools/scp2jsonl.py` on an AISHELL-1 `wav.scp` and `text` pair produces `train.jsonl` and `val.jsonl` (`docs/finetune.md:15-53`); 3600 training utterances (the first 3600 of a seeded sample of 36,000) and 200 dev utterances.

**The measured job**, both arms; the checkout under test goes first on `PYTHONPATH` so `import funasr` resolves to it. The two `++llm_conf.torch_compile=true ++llm_conf.sdpa_backends=[flash,efficient,math]` lines are passed on this branch only (they are the two lines `finetune.sh` adds; the baseline checkout does not know the keys):
```
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
export TORCHINDUCTOR_CACHE_DIR=<persistent dir> TRITON_CACHE_DIR=<persistent dir>
cd examples/industrial_data_pretraining/fun_asr_nano
numactl --cpunodebind=0 --membind=0 \
torchrun --nnodes 1 --nproc_per_node 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 26669 \
  $(which funasr-train-ds) \
  hydra.run.dir=<out>/hydra \
  ++seed=1234 ++model=<model dir> ++trust_remote_code=true \
  ++train_data_set_list=<data>/train.jsonl ++valid_data_set_list=<data>/val.jsonl \
  ++dataset_conf.data_split_num=1 ++dataset_conf.batch_sampler=BatchSampler ++dataset_conf.batch_size=6000 \
  ++dataset_conf.sort_size=1024 ++dataset_conf.batch_type=token ++dataset_conf.num_workers=4 \
  ++train_conf.max_epoch=1 ++train_conf.log_interval=1 ++train_conf.resume=true \
  ++train_conf.validate_interval=2000 ++train_conf.save_checkpoint_interval=2000 \
  ++train_conf.effective_save_name_excludes=None ++train_conf.keep_nbest_models=20 ++train_conf.avg_nbest_model=10 \
  ++train_conf.use_deepspeed=false ++train_conf.deepspeed_config=deepspeed_conf/ds_stage1.json \
  ++train_conf.find_unused_parameters=true \
  ++optim_conf.lr=0.0002 ++audio_encoder_conf.freeze=true ++audio_adaptor_conf.freeze=true ++llm_conf.freeze=false \
  ++output_dir=<out> \
  ++llm_conf.torch_compile=true ++llm_conf.sdpa_backends=[flash,efficient,math]   # this branch only
```
`trust_remote_code=true` loads the recipe's own `model.py` from the working directory, which is why the change is mirrored into `funasr/models/fun_asr_nano/model.py`. `++train_conf.find_unused_parameters=true` is a no-op on one GPU; it is there because the multi-GPU form of the script needs it (the CTC decoder is trainable but unused). The trainer has no max-steps flag, so the run length is the dataset. `++seed=1234` seeds torch, numpy and random; the batch sampler shuffles with `manual_seed(epoch)`, so the batch order is fixed.

**Correctness run.** On the verify branch, the same command with `PARITY=1 PARITY_OUT=<dir>/code.json`; for the baseline record, check out `486b4b7ce`, cherry-pick the hooks commit, run into `<dir>/base.json`; `parity compare <dir>/base.rank0.json <dir>/code.rank0.json` prints every checkpoint against its tolerance. Also swept on the baseline and rejected as noise: `OMP_NUM_THREADS` 8 and 1, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `++cudnn_benchmark=true`, `PYTORCH_NVML_BASED_CUDA_CHECK=1`; excluded because they change the data or the numerics: `++dataset_conf.num_workers=8`, `++optim_conf.fused=true`. Earlier revisions of this PR: compile alone measured 1.38x (0.0985 to 0.0713 s) against a baseline with `TORCH_CUDNN_SDPA_DEPRIORITIZED=1` set; the process-wide switch plus compile-on-by-default measured 0.2438 to 0.0680 s (3.59x) on the same setup as this table.

**Traces** (torch.profiler, rank 0, steps 4 to 9 of the unmodified script and steps 9 to 14 of the optimized arm, previous revision, same computation; open in https://ui.perfetto.dev):
- baseline: https://drive.google.com/open?id=14roiLxfbM3zmsxTgmcO2QKoiZnS4B_n_
- optimized: https://drive.google.com/open?id=1Md6OxOjDtvy0rfCtyWPrkOWYT7U-STqd

</details>















