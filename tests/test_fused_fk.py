from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for Triton FK tests.", allow_module_level=True)

from memfree_sim import ArmSpec, fk_reference, fused_fk


@pytest.mark.parametrize("batch_size", [32, 2048])
def test_fused_fk_matches_reference_forward(batch_size: int) -> None:
    spec = ArmSpec.default(device="cuda", dtype=torch.float32)
    q = torch.randn(batch_size, 7, device="cuda", dtype=torch.float32)

    actual = fused_fk(q, spec)
    expected = fk_reference(q, spec)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("batch_size", [16, 1024])
def test_fused_fk_matches_reference_backward(batch_size: int) -> None:
    spec = ArmSpec.default(device="cuda", dtype=torch.float32)
    q_ref = torch.randn(batch_size, 7, device="cuda", dtype=torch.float32, requires_grad=True)
    q_fused = q_ref.detach().clone().requires_grad_(True)

    out_ref = fk_reference(q_ref, spec)
    loss_ref = out_ref[:, :3, :].square().sum()
    loss_ref.backward()

    out_fused = fused_fk(q_fused, spec)
    loss_fused = out_fused[:, :3, :].square().sum()
    loss_fused.backward()

    torch.testing.assert_close(q_fused.grad, q_ref.grad, rtol=1e-4, atol=1e-5)

