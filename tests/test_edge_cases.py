from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from memfree_sim import ArmSpec, fk_reference


def test_reference_handles_batch_size_one_and_zero_angles() -> None:
    spec = ArmSpec.default(dtype=torch.float32)
    q = torch.zeros(1, 7, dtype=torch.float32)

    pose, intermediates = fk_reference(q, spec, return_intermediates=True)

    assert pose.shape == (1, 4, 4)
    assert intermediates["local_transforms"].shape == (1, 7, 4, 4)
    assert intermediates["global_transforms"].shape == (1, 7, 4, 4)
    torch.testing.assert_close(pose[:, 3, :], torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32))


def test_reference_handles_joint_limit_boundary_values() -> None:
    spec = ArmSpec.default(dtype=torch.float32)
    assert spec.joint_limits is not None
    lower = torch.tensor([limit[0] for limit in spec.joint_limits], dtype=torch.float32)
    upper = torch.tensor([limit[1] for limit in spec.joint_limits], dtype=torch.float32)
    q = torch.stack((lower, upper), dim=0)

    pose = fk_reference(q, spec)

    assert pose.shape == (2, 4, 4)
    assert torch.isfinite(pose).all()


try:
    import triton  # noqa: F401

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

from memfree_sim import fused_fk


@pytest.mark.skipif(not HAS_TRITON, reason="Triton is not installed.")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton edge-case tests.")
def test_fused_fk_accepts_non_contiguous_input() -> None:
    spec = ArmSpec.default(device="cuda", dtype=torch.float32)
    q_base = torch.randn(64, 14, device="cuda", dtype=torch.float32)
    q = q_base[:, ::2]
    assert not q.is_contiguous()

    actual = fused_fk(q, spec)
    expected = fk_reference(q.contiguous(), spec)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
