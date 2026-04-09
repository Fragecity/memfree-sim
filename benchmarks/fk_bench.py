from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from memfree_sim import ArmSpec, fk_reference, fused_fk


@dataclass
class BenchResult:
    batch_size: int
    impl: str
    forward_ms: float
    backward_ms: float
    envs_per_sec: float
    peak_mem_mb: float


def _time_cuda_callable(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end))


def _run_impl(fn, batch_size: int, arm_spec: ArmSpec, warmup: int, repeat: int) -> BenchResult:
    for _ in range(warmup):
        q = torch.randn(batch_size, 7, device="cuda", dtype=torch.float32, requires_grad=True)
        out = fn(q, arm_spec)
        loss = out[:, :3, :].square().mean()
        loss.backward()
        torch.cuda.synchronize()

    forward_times = []
    backward_times = []
    peak_bytes = 0

    for _ in range(repeat):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        q = torch.randn(batch_size, 7, device="cuda", dtype=torch.float32, requires_grad=True)
        out_holder: dict[str, torch.Tensor] = {}

        def _forward() -> None:
            out_holder["out"] = fn(q, arm_spec)

        forward_ms = _time_cuda_callable(_forward)
        loss = out_holder["out"][:, :3, :].square().mean()
        backward_ms = _time_cuda_callable(loss.backward)

        peak_bytes = max(peak_bytes, torch.cuda.max_memory_allocated())
        forward_times.append(forward_ms)
        backward_times.append(backward_ms)

    avg_forward = sum(forward_times) / len(forward_times)
    avg_backward = sum(backward_times) / len(backward_times)
    total_ms = avg_forward + avg_backward
    return BenchResult(
        batch_size=batch_size,
        impl=fn.__name__,
        forward_ms=avg_forward,
        backward_ms=avg_backward,
        envs_per_sec=(batch_size * 1000.0) / total_ms,
        peak_mem_mb=peak_bytes / (1024.0 * 1024.0),
    )


def _default_outputs() -> tuple[Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"fk_bench_{stamp}.json", out_dir / f"fk_bench_{stamp}.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark reference vs fused FK kernels.")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[4096, 8192, 16384, 32768])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--csv-out", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for fk_bench.")

    arm_spec = ArmSpec.default(device="cuda", dtype=torch.float32)
    implementations = [
        ("reference", fk_reference),
        ("fused", fused_fk),
    ]

    results: list[BenchResult] = []
    disabled_impls: set[str] = set()
    for batch_size in args.batch_sizes:
        for impl_name, impl in implementations:
            if impl_name in disabled_impls:
                continue
            try:
                result = _run_impl(impl, batch_size, arm_spec, args.warmup, args.repeat)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    print(f"[{impl_name}] batch_size={batch_size}: OOM, stopping further larger batches for this impl.")
                    torch.cuda.empty_cache()
                    disabled_impls.add(impl_name)
                    continue
                raise
            result.impl = impl_name
            results.append(result)
            print(
                f"{impl_name:>9} batch={batch_size:>6} "
                f"forward={result.forward_ms:>8.3f}ms "
                f"backward={result.backward_ms:>8.3f}ms "
                f"peak={result.peak_mem_mb:>8.2f}MB "
                f"throughput={result.envs_per_sec:>10.1f} env/s"
            )

    json_out, csv_out = _default_outputs()
    if args.json_out is not None:
        json_out = args.json_out
    if args.csv_out is not None:
        csv_out = args.csv_out

    json_out.parent.mkdir(parents=True, exist_ok=True)
    csv_out.parent.mkdir(parents=True, exist_ok=True)

    with json_out.open("w", encoding="utf-8") as fh:
        json.dump([asdict(result) for result in results], fh, indent=2)

    with csv_out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["batch_size", "impl", "forward_ms", "backward_ms", "envs_per_sec", "peak_mem_mb"],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))

    print(f"Wrote {json_out}")
    print(f"Wrote {csv_out}")


if __name__ == "__main__":
    main()
