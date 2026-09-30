---
name: extract-computation-graph
description: Read a PyTorch job's code into two files in the workspace’s records/ directory. records/computation_graph.json says what is computed: the module tree down to op level, with a code anchor and a repeat count on every entry. records/execution_schedule.json says when and where it is computed and what moves between devices: phases, the data path onto the GPU, host to device copies and their sync points, the collectives between GPUs, and which overlaps are intended. It reads code and runs nothing. Use it to build either file, and to rebuild one when the code has moved under it.
---

# extract-computation-graph

In: the job's code, at a known commit.
Out: `records/computation_graph.json` and `records/execution_schedule.json` in the workspace’s records/ directory.

This runs nothing. It records what you expect the job to do. The profile records
what it actually did, and you read one against the other.

## Two files, and what each is for

| | `records/computation_graph.json` | `records/execution_schedule.json` |
|---|---|---|
| says | **what** is computed | **when and where** it is computed, and what moves between devices |
| lets you attribute | a measured kernel back to the module, layer or op that issued it, times its repeat count | a measured idle gap back to the copy, collective or sync point behind it |
| lets you check | whether the cost sits where the structure says the work is | whether what the code meant to overlap actually did |

The fields below are examples. Add what this job needs.

## `records/computation_graph.json`

| field | why it is there |
|---|---|
| `job_type`: training, inference serving, or batch | each has a different skeleton. Do not assume training |
| `commit` | every `file:line` below means nothing without the sha it was read at |
| `entrypoint` | the step function, as a `file:line` |
| `modules`, down to op level (model -> modules -> layers -> ops), each naming its enclosing parent and its execution order | findings land on ops, not on files |
| `shapes` on the boundaries you can read | shape decides which kernel gets picked and how much memory the tensor takes |
| `code_anchor` on every entry, written `model/dit.py:DiTBlock.forward:214` | this is where a root cause has to land later. Symbol before line, so it survives an edit above it |
| `repeat` on every unit that repeats: the unit and its count | a cost inside a unit multiplies by N. Rank by `per-unit cost x N`, never by one instance: forty blocks turn a millisecond into forty |

## `records/execution_schedule.json`

| field | what to record |
|---|---|
| `phases` | startup, data, forward, backward, optimizer, periodic eval or checkpoint, teardown. Each with a `kind` of one-off, repeated or periodic, and who it `runs_on` |
| `data_path` | every hop from disk to device: worker processes, collation, pinned staging, the copy itself. Mark which hops are asynchronous and which block the step |
| `host_device` | the H2D and D2H copies, whether their source is pinned or pageable, and every sync point: `.item()`, `.cpu()`, `.numpy()`, `.tolist()`, `torch.cuda.synchronize()`, printing a tensor, any Python branch that reads one |
| `device_device` | the collectives, each tagged with the parallelism it serves (data, tensor, pipeline, expert, sequence) and whether the call blocks or was issued with `async_op=True` |
| `compute` | forward, backward, which blocks recompute under activation checkpointing, gradient accumulation, the optimizer step, and the `torch.compile` regions with the graph breaks between them |
| `overlap` | what is supposed to run at the same time as what, and on which stream |

## Plan the read first

The plan has to descend: model, modules, layers, ops etc. A plan that stops at module
level produces a graph that stops there too.

## Do not

- Run the job. This reads code and nothing else.
- Edit the code. Read it where it is.
- Fill in a field you did not verify. A wrong `repeat` count is worse than a
  missing one, because everything downstream multiplies by it.
