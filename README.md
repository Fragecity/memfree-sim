# Fused Differentiable Kinematics Kernel v1

`memfree-sim` is a minimal research prototype for one narrow problem: batched forward kinematics of a fixed 7-DOF serial arm in large-scale robot RL, where the memory wall is driven by repeated chains of tiny `4x4` SE(3) transforms.

The repository contains:

- A PyTorch reference implementation that materializes per-joint local/global transforms.
- A Triton fused forward kernel that keeps the chain in registers/SRAM and only writes the final end-effector pose.
- A custom `torch.autograd.Function` with an analytic backward pass that returns gradients with respect to joint angles without emitting an explicit Jacobian tensor.
- CUDA benchmarks for forward time, backward time, end-to-end throughput, and peak memory.
- Unit tests for local transforms, FK correctness, gradients, and edge cases.

## Environment

The intended runtime is `WSL2 Ubuntu` or native Linux with CUDA. This repository is designed around:

- Python `3.11` or `3.12`
- PyTorch with CUDA
- Triton on Linux

This project is managed with `uv`. The repository pins the local interpreter via [`.python-version`](C:/Users/15246/Projects/memfree-sim/.python-version) and uses the `dev` dependency group by default.

Recommended setup inside WSL:

```bash
uv sync
```

Do not install a separate Linux NVIDIA driver inside WSL; use the Windows host driver CUDA mapping.

Useful commands:

```bash
uv run pytest -q
uv run python -m benchmarks.fk_bench --batch-sizes 4096 8192 16384 32768
```

## Package Layout

```text
src/memfree_sim/
  arm_spec.py      # 7-DOF arm constants
  kinematics.py    # reference FK + analytic joint transforms
  triton_fk.py     # fused Triton forward + custom backward
tests/
benchmarks/
```

## Public API

```python
import torch

from memfree_sim import ArmSpec, fk_reference, fused_fk, FusedFKFunction

arm_spec = ArmSpec.default(device="cuda")
q = torch.randn(8192, 7, device="cuda", dtype=torch.float32, requires_grad=True)

pose_ref = fk_reference(q, arm_spec)
pose_fused = fused_fk(q, arm_spec)
pose_direct = FusedFKFunction.apply(q, arm_spec)
```

`fk_reference(q, arm_spec, return_intermediates=True)` returns:

```python
(pose, {"local_transforms": ..., "global_transforms": ...})
```

## Benchmarks

Run:

```bash
uv run python -m benchmarks.fk_bench --batch-sizes 4096 8192 16384 32768
```

The benchmark writes JSON and CSV results into `benchmarks/results/`. Reported fields:

- `batch_size`
- `impl`
- `forward_ms`
- `backward_ms`
- `envs_per_sec`
- `peak_mem_mb`

`envs_per_sec` is computed from end-to-end forward + backward time.

## Scope

This v1 intentionally does not cover:

- JAX
- NeRF or graphics workloads
- Multi-branch kinematic trees
- Explicit Jacobian output tensors
- Mixed precision

Those are follow-on extensions after the serial-chain CUDA path is validated.
