from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from memfree_sim import ArmSpec, fk_reference, fused_fk


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure numerical parity between reference and fused FK.")
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for fk_parity.")

    torch.manual_seed(args.seed)
    spec = ArmSpec.default(device="cuda", dtype=torch.float32)
    q_ref = torch.randn(args.batch_size, 7, device="cuda", dtype=torch.float32, requires_grad=True)
    q_fused = q_ref.detach().clone().requires_grad_(True)

    out_ref = fk_reference(q_ref, spec)
    out_fused = fused_fk(q_fused, spec)
    forward_abs = (out_fused - out_ref).abs()

    loss_ref = out_ref[:, :3, :].square().sum()
    loss_fused = out_fused[:, :3, :].square().sum()
    loss_ref.backward()
    loss_fused.backward()
    grad_abs = (q_fused.grad - q_ref.grad).abs()

    payload = {
        "batch_size": args.batch_size,
        "seed": args.seed,
        "forward_max_abs_err": float(forward_abs.max().item()),
        "forward_mean_abs_err": float(forward_abs.mean().item()),
        "grad_max_abs_err": float(grad_abs.max().item()),
        "grad_mean_abs_err": float(grad_abs.mean().item()),
    }

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
