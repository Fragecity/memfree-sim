# Agent Guide

This repository is a focused prototype for fused differentiable forward kinematics of a fixed 7-DOF serial arm. Keep the project centered on the existing robot RL simulation story; do not broaden scope to NeRF, generic physics engines, or multi-branch kinematic trees unless the user explicitly asks for that expansion.

## Technical Constraints

- Runtime target is Linux or WSL2 with CUDA.
- Triton is Linux-only in practice for this project; Windows-host development may be limited to static edits and skipped tests.
- The current implementation target is CUDA `float32` only.
- Public API surface should stay small: `ArmSpec`, `fk_reference`, `fused_fk`, `FusedFKFunction`.

## Project Layout

- `src/memfree_sim/arm_spec.py`: fixed 7-DOF arm constants and device/dtype transfer logic.
- `src/memfree_sim/kinematics.py`: reference FK, local transforms, and analytic backward math.
- `src/memfree_sim/triton_fk.py`: Triton fused forward kernel and custom autograd wrapper.
- `tests/`: correctness and edge-case coverage.
- `benchmarks/`: performance measurement and result export.

## Environment And Commands

Use `uv` for environment management.

```bash
uv sync
uv run pytest -q
uv run python -m benchmarks.fk_bench --batch-sizes 4096 8192 16384 32768
```

If CUDA/Triton is unavailable on the current machine, avoid pretending benchmarks were validated. State that limitation explicitly.

## Editing Expectations

- Preserve the memory-wall framing: avoid reintroducing large intermediate tensors into the fused path.
- Keep benchmark outputs in `benchmarks/results/`.
- Prefer adding targeted tests with every non-trivial kernel or autograd change.
- If runtime assumptions change, update both `README.md` and this file in the same edit.
