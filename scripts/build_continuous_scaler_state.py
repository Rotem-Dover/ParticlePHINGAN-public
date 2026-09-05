#!/usr/bin/env python
"""Build ContinuousYScaler's portable tensor state from the analytic mean/std lookup.

`physics.g4h_ionisation.generators.continuous_generator.build_analytic_mean_std_table`
builds the 5000x5000x2 continuous mean/std lookup analytically and writes it
to `paths.PATH_ANALYTIC_MEAN_STD_TABLE` (under `paths.TABLES_DIR`).
`ContinuousYScaler.load()` does not read that pickle directly -- it restores
its lookup from a `.state.pt` snapshot under `paths.SCALERS_DIR`. This
script converts the pickle into that portable-state format.

Usage:
    PYTHONPATH=src python scripts/build_continuous_scaler_state.py [--force]

Writes exactly one file:
    <SCALERS_DIR>/ContinuousYScaler.state.pt
"""
import argparse
import pickle
import sys

import torch

import common.paths as paths

STEPS = 5_000
STATE_FILENAME = "ContinuousYScaler.state.pt"


def build_state() -> dict[str, torch.Tensor]:
    """Load the analytic pickle and shape it into the state dict `load()` reads."""
    src = paths.PATH_ANALYTIC_MEAN_STD_TABLE
    if not src.exists():
        raise FileNotFoundError(
            f"Analytic mean/std lookup table not found at {src}. Build it first with: "
            f"PYTHONPATH=src python -m physics.g4h_ionisation.generators."
            f"continuous_generator.build_analytic_mean_std_table"
        )
    with open(src, "rb") as f:
        kin_table, step_table, lookup_table = pickle.load(f)

    assert tuple(kin_table.shape) == (STEPS,), f"kin_table {tuple(kin_table.shape)}"
    assert tuple(step_table.shape) == (STEPS,), f"step_table {tuple(step_table.shape)}"
    assert tuple(lookup_table.shape) == (STEPS, STEPS, 2), f"lookup_table {tuple(lookup_table.shape)}"

    kin_table = kin_table.to(torch.float32).cpu()
    step_table = step_table.to(torch.float32).cpu()
    lookup_table = lookup_table.to(torch.float32).cpu()

    # `load()` re-derives all six scalars from the three tables at load time;
    # they are stored here too so the state file is self-describing.
    return {
        "kin_table": kin_table,
        "step_table": step_table,
        "lookup_table": lookup_table,
        "_kin_delta": (kin_table[-1] - kin_table[0]) / (kin_table.shape[0] - 1),
        "_steps_delta": (step_table[-1] - step_table[0]) / (step_table.shape[0] - 1),
        "_kin_table_start": kin_table[0].clone(),
        "_step_table_start": step_table[0].clone(),
    }


def _tensors_identical(a: torch.Tensor, b: torch.Tensor) -> bool:
    """NaN-aware equality: NaN positions must coincide, finite values must match."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    na, nb = torch.isnan(a), torch.isnan(b)
    if not torch.equal(na, nb):
        return False
    return bool(torch.equal(a[~na], b[~nb]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="overwrite an existing state file")
    args = ap.parse_args()

    out_dir = paths.SCALERS_DIR
    out = out_dir / STATE_FILENAME
    if out.exists() and not args.force:
        print(f"REFUSING: {out} already exists. Pass --force to overwrite.")
        return 1

    state = build_state()
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, out)

    reloaded = torch.load(out, map_location="cpu", weights_only=True)
    assert set(reloaded) == set(state), f"key mismatch: {set(reloaded)} vs {set(state)}"
    for k in state:
        assert _tensors_identical(state[k], reloaded[k]), f"round-trip mismatch on {k!r}"

    lut = reloaded["lookup_table"]
    nan_mu = int(torch.isnan(lut[..., 0]).sum())
    nan_sigma = int(torch.isnan(lut[..., 1]).sum())
    cells = lut.shape[0] * lut.shape[1]
    step = reloaded["step_table"]
    kin = reloaded["kin_table"]

    print(f"wrote {out}")
    print(f"  keys        : {sorted(reloaded)}")
    print(f"  lookup      : {tuple(lut.shape)} {lut.dtype}")
    print(f"  NaN mu      : {nan_mu} / {cells}  ({100.0 * nan_mu / cells:.2f}%)")
    print(f"  NaN sigma   : {nan_sigma} / {cells}  ({100.0 * nan_sigma / cells:.2f}%)")
    print(f"  step grid   : log10 {float(step[0]):.4f} .. {float(step[-1]):.4f}"
          f"   [analytic mean/std lookup grid]")
    print(f"  kin grid    : log10 {float(kin[0]):.4f} .. {float(kin[-1]):.4f}")
    if nan_mu == 0:
        print("  WARNING: zero NaN -- the analytic lookup should carry a NaN region "
              "over the deterministic regime; this looks like a different table.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
