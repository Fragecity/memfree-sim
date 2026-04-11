from __future__ import annotations

import torch
import triton
import triton.language as tl

from .arm_spec import ArmSpec
from .kinematics import analytic_fk_backward


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8, num_stages=2),
    ],
    key=["batch_size"],
)
@triton.jit
def _fk_forward_kernel(
    q_ptr,
    a_ptr,
    d_ptr,
    alpha_ptr,
    theta_offset_ptr,
    out_ptr,
    batch_size,
    q_stride0,
    out_stride0,
    out_stride1,
    out_stride2,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    env_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = env_offsets < batch_size
    zero_vec = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    one_vec = tl.full((BLOCK_SIZE,), 1.0, tl.float32)

    r00 = one_vec
    r01 = zero_vec
    r02 = zero_vec
    r10 = zero_vec
    r11 = one_vec
    r12 = zero_vec
    r20 = zero_vec
    r21 = zero_vec
    r22 = one_vec
    tx = zero_vec
    ty = zero_vec
    tz = zero_vec

    for joint_idx in range(7):
        theta = tl.load(
            q_ptr + env_offsets * q_stride0 + joint_idx, mask=mask, other=0.0
        )
        theta = theta + tl.load(theta_offset_ptr + joint_idx)
        alpha = tl.load(alpha_ptr + joint_idx)
        a = tl.load(a_ptr + joint_idx)
        d = tl.load(d_ptr + joint_idx)

        ct = tl.cos(theta)
        st = tl.sin(theta)
        ca = tl.cos(alpha)
        sa = tl.sin(alpha)

        l00 = ct
        l01 = -st * ca
        l02 = st * sa
        l10 = st
        l11 = ct * ca
        l12 = -ct * sa
        l20 = zero_vec
        l21 = zero_vec + sa
        l22 = zero_vec + ca
        lx = a * ct
        ly = a * st
        lz = zero_vec + d

        nr00 = r00 * l00 + r01 * l10 + r02 * l20
        nr01 = r00 * l01 + r01 * l11 + r02 * l21
        nr02 = r00 * l02 + r01 * l12 + r02 * l22
        nr10 = r10 * l00 + r11 * l10 + r12 * l20
        nr11 = r10 * l01 + r11 * l11 + r12 * l21
        nr12 = r10 * l02 + r11 * l12 + r12 * l22
        nr20 = r20 * l00 + r21 * l10 + r22 * l20
        nr21 = r20 * l01 + r21 * l11 + r22 * l21
        nr22 = r20 * l02 + r21 * l12 + r22 * l22

        ntx = r00 * lx + r01 * ly + r02 * lz + tx
        nty = r10 * lx + r11 * ly + r12 * lz + ty
        ntz = r20 * lx + r21 * ly + r22 * lz + tz

        r00 = nr00
        r01 = nr01
        r02 = nr02
        r10 = nr10
        r11 = nr11
        r12 = nr12
        r20 = nr20
        r21 = nr21
        r22 = nr22
        tx = ntx
        ty = nty
        tz = ntz

    tl.store(
        out_ptr + env_offsets * out_stride0 + 0 * out_stride1 + 0 * out_stride2,
        r00,
        mask=mask,
    )
    tl.store(
        out_ptr + env_offsets * out_stride0 + 0 * out_stride1 + 1 * out_stride2,
        r01,
        mask=mask,
    )
    tl.store(
        out_ptr + env_offsets * out_stride0 + 0 * out_stride1 + 2 * out_stride2,
        r02,
        mask=mask,
    )
    tl.store(
        out_ptr + env_offsets * out_stride0 + 0 * out_stride1 + 3 * out_stride2,
        tx,
        mask=mask,
    )
    tl.store(
        out_ptr + env_offsets * out_stride0 + 1 * out_stride1 + 0 * out_stride2,
        r10,
        mask=mask,
    )
    tl.store(
        out_ptr + env_offsets * out_stride0 + 1 * out_stride1 + 1 * out_stride2,
        r11,
        mask=mask,
    )
    tl.store(
        out_ptr + env_offsets * out_stride0 + 1 * out_stride1 + 2 * out_stride2,
        r12,
        mask=mask,
    )
    tl.store(
        out_ptr + env_offsets * out_stride0 + 1 * out_stride1 + 3 * out_stride2,
        ty,
        mask=mask,
    )
    tl.store(
        out_ptr + env_offsets * out_stride0 + 2 * out_stride1 + 0 * out_stride2,
        r20,
        mask=mask,
    )
    tl.store(
        out_ptr + env_offsets * out_stride0 + 2 * out_stride1 + 1 * out_stride2,
        r21,
        mask=mask,
    )
    tl.store(
        out_ptr + env_offsets * out_stride0 + 2 * out_stride1 + 2 * out_stride2,
        r22,
        mask=mask,
    )
    tl.store(
        out_ptr + env_offsets * out_stride0 + 2 * out_stride1 + 3 * out_stride2,
        tz,
        mask=mask,
    )
    tl.store(
        out_ptr + env_offsets * out_stride0 + 3 * out_stride1 + 0 * out_stride2,
        zero_vec,
        mask=mask,
    )
    tl.store(
        out_ptr + env_offsets * out_stride0 + 3 * out_stride1 + 1 * out_stride2,
        zero_vec,
        mask=mask,
    )
    tl.store(
        out_ptr + env_offsets * out_stride0 + 3 * out_stride1 + 2 * out_stride2,
        zero_vec,
        mask=mask,
    )
    tl.store(
        out_ptr + env_offsets * out_stride0 + 3 * out_stride1 + 3 * out_stride2,
        one_vec,
        mask=mask,
    )


def _prepare_inputs(q: torch.Tensor, arm_spec: ArmSpec) -> tuple[torch.Tensor, ArmSpec]:
    if not q.is_cuda:
        raise ValueError("fused_fk requires a CUDA tensor.")
    if q.dtype != torch.float32:
        raise TypeError(f"fused_fk only supports torch.float32, got {q.dtype}.")
    if q.ndim != 2 or q.shape[1] != 7:
        raise ValueError(f"Expected q with shape [B, 7], got {tuple(q.shape)}.")
    q = q.contiguous()
    arm_spec = arm_spec.to(device=q.device, dtype=q.dtype)
    return q, arm_spec


def _launch_fused_fk(q: torch.Tensor, arm_spec: ArmSpec) -> torch.Tensor:
    batch_size = q.shape[0]
    out = torch.empty((batch_size, 4, 4), device=q.device, dtype=q.dtype)
    grid = lambda meta: (triton.cdiv(batch_size, meta["BLOCK_SIZE"]),)
    _fk_forward_kernel[grid](
        q,
        arm_spec.a,
        arm_spec.d,
        arm_spec.alpha,
        arm_spec.theta_offset,
        out,
        batch_size,
        q.stride(0),
        out.stride(0),
        out.stride(1),
        out.stride(2),
    )
    return out


class FusedFKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, arm_spec: ArmSpec) -> torch.Tensor:
        q, arm_spec = _prepare_inputs(q, arm_spec)
        out = _launch_fused_fk(q, arm_spec)
        ctx.arm_spec = arm_spec
        ctx.save_for_backward(q)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (q,) = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_q = analytic_fk_backward(q, grad_output, ctx.arm_spec)
        return grad_q, None


def fused_fk(q: torch.Tensor, arm_spec: ArmSpec) -> torch.Tensor:
    return FusedFKFunction.apply(q, arm_spec)
