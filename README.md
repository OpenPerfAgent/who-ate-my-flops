# who-ate-my-flops

A small plugin for Claude Code and Codex to diagnose and optimize end-to-end performance in PyTorch training and inference workloads.

[Read the blog](https://openperfagent.github.io/who-ate-my-flops/) for the design and results.

## What It Does

Provides a harness for performance optimization, with:

- Workload context and profiling analysis to guide the agent’s reasoning
- Correctness checks to evaluate changes

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

### Watch a usage example

https://github.com/user-attachments/assets/be65b2d6-da4f-4b3a-a68a-c32a47485200

## Roadmap

- [ ] Add skills for NVIDIA Nsight Systems.
- [ ] Further improvement of the alignment with user intent.
- [ ] Enable recursive delegation to kernel optimization agents when needed.

## License

Copyright © 2026 Impossible, Inc.

Licensed under the [Apache License, Version 2.0](LICENSE).
