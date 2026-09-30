<h1><img src="assets/comet.png" alt="" width="48" height="48" align="absmiddle"> who-ate-my-flops</h1>

A small plugin for Claude Code and Codex to diagnose and optimize end-to-end performance in PyTorch training and inference workloads.

[![Release v0.1.0](https://img.shields.io/badge/Release-v0.1.0-91b7ab?style=flat-square)](https://github.com/OpenPerfAgent/who-ate-my-flops/releases/tag/v0.1.0)
[![Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-91b7ab?style=flat-square)](LICENSE)
[![Claude Code](https://img.shields.io/badge/Claude_Code-supported-cb9274?style=flat-square)](#claude-code)
[![Codex](https://img.shields.io/badge/Codex-supported-91b7ab?style=flat-square)](#codex)

**[Blog](https://openperfagent.github.io/who-ate-my-flops/) · [Installation](#installation) · [Quick Start](#quick-start) · [Roadmap](#roadmap)**

🔥 Optimizations developed with the plugin have been merged into FunASR, FastVideo, gsplat, Ultralytics, Unsloth, and SGLang, with measured speedups of up to 3.6× on the tested workloads.

### See how it works in 1 minute

https://github.com/user-attachments/assets/be65b2d6-da4f-4b3a-a68a-c32a47485200

## What It Does

Provides a harness for end-to-end PyTorch job performance optimization, with:

- Workload context and profiling analysis to guide the agent’s reasoning
- Correctness checks to evaluate changes
- Guidance for the agent to ask questions that uncover users’ goals and constraints.

Optimizations can range from configuration changes to Python code and GPU kernels. The plugin works on individual training or inference jobs with a repeatable launch command, e.g. jobs submitted through Slurm.

## Installation

### Ask your agent

Tell Claude Code or Codex:

```text
Install the who-ate-my-flops plugin from
https://github.com/OpenPerfAgent/who-ate-my-flops
for this client. Follow the installation instructions in its README.
```

### Manual installation

#### Claude Code

Run these commands in Claude Code:

```text
/plugin marketplace add OpenPerfAgent/who-ate-my-flops
/plugin install who-ate-my-flops@openperfagent
```

#### Codex

Run these commands in your terminal:

```bash
codex plugin marketplace add OpenPerfAgent/who-ate-my-flops
codex plugin add who-ate-my-flops@openperfagent
```

## Quick Start

Have your code repository, a command that launches the training or inference job, and access to an idle GPU environment ready. Start Claude Code or Codex in the repository you want to optimize.

Before running `init`, enable **Auto mode** in Claude Code or **Approve for me** in Codex so the agent can work with fewer interruptions.

### 1. Set up the workload

Run `init` and give the agent your launch command and GPU environment. It will ask about your optimization goal, constraints, and correctness requirements, and record them in `contract.md`.

| Claude Code | Codex |
|---|---|
| `/who-ate-my-flops:init` | `$who-ate-my-flops:init` |

### 2. Diagnose or optimize

Choose `diagnose` for one investigation and a measured fix, or `optimize` to let the agent work through multiple improvements.

| | Claude Code | Codex |
|---|---|---|
| Diagnose | `/who-ate-my-flops:diagnose` | `$who-ate-my-flops:diagnose` |
| Optimize | `/who-ate-my-flops:optimize` | `$who-ate-my-flops:optimize` |

### 3. Wait and review the results

Wait for the agent to finish, then start with `latest-report.md` to review the results.
Reports and experiment records are in `workspace-who-ate-my-flops/`.

```text
workspace-who-ate-my-flops/
├── latest-report.md   # Results, correctness checks, and reproduction commands
├── benchmark.csv      # Baseline and optimization measurements
├── contract.md        # Agreed goals and constraints
├── records/           # Process records and planning notes
├── tools/             # Helper scripts
├── runs/              # Run outputs
└── commits/           # Profiling traces and diagnoses by commit
```

Questions, bugs, or feedback? [Open an issue](https://github.com/OpenPerfAgent/who-ate-my-flops/issues).

## Roadmap

- [ ] Add skills for NVIDIA Nsight Systems.
- [ ] Further improvement of the alignment with user intent.
- [ ] Enable recursive delegation to kernel optimization agents when needed.

## Citation

You can cite our [blog post](https://openperfagent.github.io/who-ate-my-flops/) with:

```bibtex
@misc{zhao2026whoatemyflops,
  title        = {Beyond kernels: Letting agents optimize the whole {AI} job},
  author       = {Zhao, Hexu and Xi, Haocheng and Wang, Yichuan and Yin, Shaofeng and Lv, Zhaoyang and Feng, Haiwen and Li, Xiuyu and Panda, Aurojit and Li, Jinyang},
  year         = {2026},
  url          = {https://openperfagent.github.io/who-ate-my-flops/}
}
```

## License

Copyright © 2026 Impossible, Inc.

Licensed under the [Apache License, Version 2.0](LICENSE).
