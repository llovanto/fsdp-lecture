---
title: FSDP
transition: slide-left
---

# FSDP (Fully Sharded Data Parallel)

---

# Plan

- Collective ops intro
- DDP recap
- Mixed precision recap
- FSDP basics
- Prefetching
- Sharding strategies
- HSDP
- Resharding to a smaller world size
- CPU offloading
- FSDP implementations

---

# Collective ops intro - `torch.distributed`

- Process groups: 1 GPU - 1 process
- Distributed training requires inter-process communication ops:
  - `AllReduce`
  - `AllGather`
  - `ReduceScatter`
  - ...
- Backends implement these ops: NCCL for GPUs, GLOO for CPUs, etc.

---

# Collective ops intro - `AllReduce`

<img border="rounded" src="./assets/all_reduce.png" alt="" width="700">

---

# Collective ops intro - `AllGather`

<img border="rounded" src="./assets/all_gather.png" alt="" width="700">

---

# Collective ops intro - `ReduceScatter`

<img border="rounded" src="./assets/reduce_scatter.png" alt="" width="600">

---

# Collective ops intro

<img border="rounded" src="./assets/comm_equation.webp" alt="" width="800">

---

# DDP recap

<img border="rounded" src="./assets/ddp.png" alt="" width="700">

---

# Mixed precision recap

<img border="rounded" src="./assets/float_types.webp" alt="">

---

# Mixed precision recap

<img border="rounded" src="./assets/mixed_precision.png" alt="">

---
layout: center
---

# Fully Sharded Data Parallel

---

# Fully SHARDED Data Parallel

Huge models don't fit into one GPU.

Compared to DDP:

- master weights, grads and optimizer states are sharded (**less memory overhead**)
- weights are unsharded before forward / backward and resharded after
- grads are reduced (and sharded) after backward
- per-parameter communications are grouped for better network utilization

<br>

<img border="rounded" src="./assets/tensor_sharding.jpg" alt="" width="300">

---

# FSDP - overview

<img border="rounded" src="./assets/fsdp.png" alt="">

---

# FSDP - overview

<img border="rounded" src="./assets/fsdp_multi.png" alt="">

---

# FSDP - memory consumption (LLama 70B on 128 GPUs)

Assuming per-transformer-block sharding.

Params per block (80 blocks):

```plain
3 (w1, w2, v1) * 8192 (hidden-size) * 28672 (intermediate-size)
  + (2 (q-proj + out-proj) + 2 * 1/8 (kw-proj)) * 8192 * 8192
  + ...
  ~= 0.85B
```

---

# FSDP - memory consumption (LLama 70B on 128 GPUs)

Sharded states (in `fp32`):

```plain
weights = 70B / 128 * 4 bytes ~= 2.2 GB
grads = weights
optim_states = 2 * weights (for AdamW)
```

Unsharded states (in `fp16`/`bf16`):

```plain
weights = 2 * 0.85B * 2 bytes = 3.4 GB
grads = weights
```

Total: `(3.4 GB * 2 + 2.2 GB * (1 + 1 + 2)) = 15.6 GB = 20 % of an 80GB GPU`

Not too bad! (compared to DDP with `4 * 70B * 4 bytes = 1.1TB`)

---

# FSDP - memory consumption (LLama 70B on 1024 GPUs)

Sharded states (in `fp32`):

```plain
weights = 70B / 1024 * 4 bytes ~= 0.275 GB
grads = weights
optim_states = 2 * weights (for AdamW)
```

Unsharded states (in `fp16` or`bf16`):

```plain
weights = 2 * 0.85B * 2 bytes = 3.4 GB
grads = weights
```

Total: `(3.4 GB * 2 + 0.275 GB * (1 + 2 + 2)) = 7.9 GB = 10 % of an 80GB GPU`

Great!

---

# Prefetching - DDP

<br>
<br>
<br>
<br>

<img border="rounded" src="./assets/ddp_overlap.png" alt="">

---

# Prefetching - FSDP

<img border="rounded" src="./assets/fsdp.png" alt="">

---

# Prefetching - FSDP

<br>
<br>
<br>
<br>

<img border="rounded" src="./assets/fsdp_overlap.png" alt="">

---

# FSDP - communication overhead

<img border="rounded" src="./assets/fsdp_comm1.png" alt="">
---

# FSDP - communication overhead

<img border="rounded" src="./assets/fsdp_comm2.png" alt="" width="700">

---

# Sharding strategies - ZeRO

<img border="rounded" src="./assets/zero_memory.png" alt="">

---

# HSDP - Hybrid sharding

<img border="rounded" src="./assets/hsdp.avif" alt="" width="650">

---

# HSDP - Hybrid sharding

<img border="rounded" src="./assets/hsdp_fsdp.png" alt="">

---

# HSDP - Hybrid sharding (motivation)

<img border="rounded" src="./assets/topology.png" alt="">

---

# Resharding to a smaller world size

In case there is enough memory post-forward resharding can be done to a smaller
world size:

- less backward all-gather comms
- higher memory consumption

<br>

<img border="rounded" src="./assets/fsdp.png" alt="" width="600">

---

# CPU offloading

Master weights, grads and optmizer states can be offloaded to CPU and only
  loaded to GPU before unsharding

- less memory consumption
- some overhead on CPU-GPU transfers and CPU optimizer step

<br>

<img border="rounded" src="./assets/fsdp_workflow.png" alt="" width="600">

---

# FSDP implementations - FSDP1

FlatParameter, complex, bloated, buggy

<img border="rounded" src="./assets/fsdp1.png" alt="" width="600">

---

# FSDP implementations

## FSDP2

Per parameter sharding, clean, performant

## YaFSDP

FSDP2 rewrite with flat sharding (and so less overhead)

---

# Recap

FSDP is a data parallelism strategy with parameter sharding which:

- reduces memory consumption
- (ideally) has no performance overhead
- is used in (almost) every large-scale distributed training recipe
