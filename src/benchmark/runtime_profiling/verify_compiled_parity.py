"""Compiled-vs-eager parity gate for the generators.

Runs the same seed through the optimized (fused-MLP / torch.compile'd on
CUDA) and eager builds from `benchmark.runtime_profiling.simulator_builders`
and reports the worst step-feature delta between them. On cpu the compile
and fusion branches are both dead (torch.compile of the phase methods and
the fused-MLP kernel are cuda-only), so the two builds are the same program
and this is a no-op smoke test; on cuda it is a real numerical-parity check.

Run on a GPU:

    python -m benchmark.runtime_profiling.verify_compiled_parity

--tol defaults to None: report-only, exit 0 regardless of the measured
delta. An uncalibrated numeric default would be indistinguishable from a
threshold someone quietly loosened, so this gates only when the caller
passes an explicit --tol backed by a measured null.
"""
import argparse
import sys


def compare_compiled_vs_eager(preset: str, device: str, n_events: int,
                              n_steps: int, seed: int) -> dict:
    """Run the same seed through the compiled and eager builds and report the
    step-feature delta. On cpu both builds are identical by construction (the
    compile branches are cuda-only), so this is a no-op smoke there and a real
    comparison on cuda."""
    import numpy as np
    import torch

    from benchmark.runtime_profiling.simulator_builders import (
        build_eager_simulator, build_optimized_simulator)

    frames = []
    for build in (build_optimized_simulator, build_eager_simulator):
        torch.manual_seed(seed)
        simulator, _ = build(preset, device)
        simulator.run(E_0=1e8, n_events=n_events, n_steps=n_steps,
                      save_steps_states=True, E_kill_eV=0.0)
        frames.append(simulator.steps_features_df.to_numpy(dtype=np.float64))

    got, ref = frames
    n = min(len(got), len(ref))
    d = np.abs(got[:n] - ref[:n])
    d = d[~np.isnan(d).any(axis=1)]
    return {"preset": preset, "device": device,
            "max_abs_dz": float(d.max()) if d.size else 0.0,
            "mean_abs_dz": float(d.mean()) if d.size else 0.0}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="phin_gan")
    # None means "not passed" -> resolve to common.config.DEVICE in main().
    # Do not default this to "cpu": that would make an explicit `--device
    # cpu` indistinguishable from the unset default, silently promoting a
    # requested CPU-only smoke run to CUDA on a cluster node.
    p.add_argument("--device", default=None, choices=["cpu", "cuda"])
    p.add_argument("--n-events", type=int, default=64)
    p.add_argument("--n-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=20260728 + 4)
    # None means report-only: print the stats and exit 0 regardless of the
    # measured delta. There is no measured null behind any non-None value
    # yet, so this module never gates unless a caller explicitly opts in.
    p.add_argument("--tol", type=float, default=None,
                   help="max |compiled - eager| step-feature tolerance; "
                        "omit to report only (no gating)")
    return p


def main() -> None:
    from common.config import DEVICE

    args = _build_parser().parse_args()
    device = args.device if args.device is not None else DEVICE

    print(f"Compiled-vs-eager parity gate ({args.preset}, {device}):")
    result = compare_compiled_vs_eager(args.preset, device, args.n_events,
                                       args.n_steps, args.seed)
    print(f"  max_abs_dz:  {result['max_abs_dz']:.3e}")
    print(f"  mean_abs_dz: {result['mean_abs_dz']:.3e}")

    if args.tol is None:
        print("Report-only (--tol not set): not gating on the measured delta.")
        return

    if result["max_abs_dz"] > args.tol:
        sys.exit(f"PARITY GATE FAILED: max_abs_dz={result['max_abs_dz']:.3e} "
                 f"exceeds tol={args.tol:.3e} — compiled/fused build diverges "
                 f"from eager; do not trust optimized physics output until "
                 f"resolved.")
    print("Parity gate passed: compiled/fused matches eager within tolerance.")


if __name__ == "__main__":
    main()
