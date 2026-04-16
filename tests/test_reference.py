from __future__ import annotations

import torch

from memfree_sim.arm_spec import ArmSpec
from memfree_sim.kinematics import build_local_transforms, fk_reference, joint_transform


def test_single_joint_transform_matches_closed_form() -> None:
    spec = ArmSpec.default(dtype=torch.float32)
    theta = torch.linspace(-1.0, 1.0, steps=11, dtype=torch.float32)
    joint_idx = 3

    actual = joint_transform(
        theta + spec.theta_offset[joint_idx],
        spec.alpha[joint_idx],
        spec.a[joint_idx],
        spec.d[joint_idx],
    )

    ct = torch.cos(theta)
    st = torch.sin(theta)
    ca = torch.cos(spec.alpha[joint_idx])
    sa = torch.sin(spec.alpha[joint_idx])

    expected = torch.empty_like(actual)
    expected[:, 0, 0] = ct
    expected[:, 0, 1] = -st * ca
    expected[:, 0, 2] = st * sa
    expected[:, 0, 3] = spec.a[joint_idx] * ct
    expected[:, 1, 0] = st
    expected[:, 1, 1] = ct * ca
    expected[:, 1, 2] = -ct * sa
    expected[:, 1, 3] = spec.a[joint_idx] * st
    expected[:, 2, 0] = 0.0
    expected[:, 2, 1] = sa
    expected[:, 2, 2] = ca
    expected[:, 2, 3] = spec.d[joint_idx]
    expected[:, 3, 0] = 0.0
    expected[:, 3, 1] = 0.0
    expected[:, 3, 2] = 0.0
    expected[:, 3, 3] = 1.0

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_fk_reference_returns_consistent_intermediates() -> None:
    spec = ArmSpec.default(dtype=torch.float32)
    q = torch.randn(8, 7, dtype=torch.float32)

    pose, intermediates = fk_reference(q, spec, return_intermediates=True)

    local_transforms = build_local_transforms(q, spec)
    running = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(q.shape[0], 1, 1)
    expected_globals = []
    for joint_idx in range(7):
        running = running @ local_transforms[:, joint_idx]
        expected_globals.append(running)

    torch.testing.assert_close(intermediates["local_transforms"], local_transforms, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        intermediates["global_transforms"],
        torch.stack(expected_globals, dim=1),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(pose, expected_globals[-1], rtol=1e-6, atol=1e-6)
