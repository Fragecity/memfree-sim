from __future__ import annotations

from typing import Any

import torch

from .arm_spec import ArmSpec


def _validate_q(q: torch.Tensor, arm_spec: ArmSpec) -> torch.Tensor:
    if q.ndim != 2 or q.shape[1] != arm_spec.num_joints:
        raise ValueError(f"Expected q with shape [B, {arm_spec.num_joints}], got {tuple(q.shape)}.")
    if q.dtype != torch.float32:
        raise TypeError(f"Expected q.dtype=torch.float32, got {q.dtype}.")
    return q


def _batch_identity(batch: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    eye = torch.eye(4, device=device, dtype=dtype)
    return eye.unsqueeze(0).expand(batch, -1, -1).clone()


def joint_transform(theta: torch.Tensor, alpha: torch.Tensor, a: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    if theta.ndim != 1:
        raise ValueError(f"Expected theta with shape [B], got {tuple(theta.shape)}.")

    ct = torch.cos(theta)
    st = torch.sin(theta)
    ca = torch.cos(alpha)
    sa = torch.sin(alpha)

    zeros = torch.zeros_like(theta)
    ones = torch.ones_like(theta)

    row0 = torch.stack((ct, -st * ca, st * sa, a * ct), dim=-1)
    row1 = torch.stack((st, ct * ca, -ct * sa, a * st), dim=-1)
    row2 = torch.stack((zeros, sa.expand_as(theta), ca.expand_as(theta), d.expand_as(theta)), dim=-1)
    row3 = torch.stack((zeros, zeros, zeros, ones), dim=-1)
    return torch.stack((row0, row1, row2, row3), dim=1)


def joint_transform_derivative(theta: torch.Tensor, alpha: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    if theta.ndim != 1:
        raise ValueError(f"Expected theta with shape [B], got {tuple(theta.shape)}.")

    ct = torch.cos(theta)
    st = torch.sin(theta)
    ca = torch.cos(alpha)
    sa = torch.sin(alpha)

    zeros = torch.zeros_like(theta)

    row0 = torch.stack((-st, -ct * ca, ct * sa, -a * st), dim=-1)
    row1 = torch.stack((ct, -st * ca, st * sa, a * ct), dim=-1)
    row2 = torch.stack((zeros, zeros, zeros, zeros), dim=-1)
    row3 = torch.stack((zeros, zeros, zeros, zeros), dim=-1)
    return torch.stack((row0, row1, row2, row3), dim=1)


def build_local_transforms(q: torch.Tensor, arm_spec: ArmSpec) -> torch.Tensor:
    q = _validate_q(q, arm_spec)
    arm_spec = arm_spec.to(device=q.device, dtype=q.dtype)

    locals_per_joint = []
    for joint_idx in range(arm_spec.num_joints):
        theta = q[:, joint_idx] + arm_spec.theta_offset[joint_idx]
        local = joint_transform(theta, arm_spec.alpha[joint_idx], arm_spec.a[joint_idx], arm_spec.d[joint_idx])
        locals_per_joint.append(local)
    return torch.stack(locals_per_joint, dim=1)


def fk_reference(
    q: torch.Tensor,
    arm_spec: ArmSpec,
    return_intermediates: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    q = _validate_q(q, arm_spec)
    arm_spec = arm_spec.to(device=q.device, dtype=q.dtype)
    batch = q.shape[0]

    local_transforms = build_local_transforms(q, arm_spec)
    running = _batch_identity(batch, device=q.device, dtype=q.dtype)
    global_transforms = []
    for joint_idx in range(arm_spec.num_joints):
        running = torch.matmul(running, local_transforms[:, joint_idx])
        global_transforms.append(running)

    end_effector = running
    if not return_intermediates:
        return end_effector

    intermediates: dict[str, torch.Tensor] = {
        "local_transforms": local_transforms,
        "global_transforms": torch.stack(global_transforms, dim=1),
    }
    return end_effector, intermediates


def analytic_fk_backward(q: torch.Tensor, grad_output: torch.Tensor, arm_spec: ArmSpec) -> torch.Tensor:
    q = _validate_q(q, arm_spec)
    if grad_output.shape != (q.shape[0], 4, 4):
        raise ValueError(f"Expected grad_output with shape [B, 4, 4], got {tuple(grad_output.shape)}.")

    arm_spec = arm_spec.to(device=q.device, dtype=q.dtype)
    local_transforms = build_local_transforms(q, arm_spec)

    batch = q.shape[0]
    prefixes = [_batch_identity(batch, device=q.device, dtype=q.dtype)]
    running = prefixes[0]
    for joint_idx in range(arm_spec.num_joints):
        running = torch.matmul(running, local_transforms[:, joint_idx])
        prefixes.append(running)

    suffix = _batch_identity(batch, device=q.device, dtype=q.dtype)
    grad_q = torch.empty_like(q)
    for joint_idx in reversed(range(arm_spec.num_joints)):
        d_local = joint_transform_derivative(
            q[:, joint_idx] + arm_spec.theta_offset[joint_idx],
            arm_spec.alpha[joint_idx],
            arm_spec.a[joint_idx],
        )
        left = prefixes[joint_idx].transpose(-1, -2)
        right = suffix.transpose(-1, -2)
        grad_local = torch.matmul(torch.matmul(left, grad_output), right)
        grad_q[:, joint_idx] = (grad_local * d_local).sum(dim=(-1, -2))
        suffix = torch.matmul(local_transforms[:, joint_idx], suffix)

    return grad_q


def summarize_intermediates(q: torch.Tensor, arm_spec: ArmSpec) -> dict[str, Any]:
    pose, intermediates = fk_reference(q, arm_spec, return_intermediates=True)
    return {
        "pose": pose,
        "local_transforms": intermediates["local_transforms"],
        "global_transforms": intermediates["global_transforms"],
    }

