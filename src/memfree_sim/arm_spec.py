from __future__ import annotations

from dataclasses import dataclass
from math import pi

import torch


@dataclass(frozen=True)
class ArmSpec:
    a: torch.Tensor
    d: torch.Tensor
    alpha: torch.Tensor
    theta_offset: torch.Tensor
    joint_limits: tuple[tuple[float, float], ...] | None = None
    name: str = "seven_dof_serial_arm"

    def __post_init__(self) -> None:
        sizes = {tensor.numel() for tensor in (self.a, self.d, self.alpha, self.theta_offset)}
        if sizes != {7}:
            raise ValueError("ArmSpec expects exactly 7 joints.")

    @property
    def num_joints(self) -> int:
        return 7

    @property
    def device(self) -> torch.device:
        return self.a.device

    @property
    def dtype(self) -> torch.dtype:
        return self.a.dtype

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "ArmSpec":
        return ArmSpec(
            a=self.a.to(device=device, dtype=dtype),
            d=self.d.to(device=device, dtype=dtype),
            alpha=self.alpha.to(device=device, dtype=dtype),
            theta_offset=self.theta_offset.to(device=device, dtype=dtype),
            joint_limits=self.joint_limits,
            name=self.name,
        )

    @classmethod
    def default(
        cls,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "ArmSpec":
        # Approximate 7-DOF arm geometry with standard DH parameters.
        a = torch.tensor([0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088], device=device, dtype=dtype)
        d = torch.tensor([0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.107], device=device, dtype=dtype)
        alpha = torch.tensor(
            [-pi / 2.0, pi / 2.0, pi / 2.0, pi / 2.0, -pi / 2.0, pi / 2.0, 0.0],
            device=device,
            dtype=dtype,
        )
        theta_offset = torch.zeros(7, device=device, dtype=dtype)
        joint_limits = tuple((-2.9, 2.9) for _ in range(7))
        return cls(a=a, d=d, alpha=alpha, theta_offset=theta_offset, joint_limits=joint_limits)

