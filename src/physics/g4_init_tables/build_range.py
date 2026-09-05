"""
Build the CSDA range table from a dE/dx table.

Port of G4LossTableBuilder::BuildRangeTable
(geant4-11.2.0/source/processes/electromagnetic/utils/src/G4LossTableBuilder.cc):

  range(E_0)            = 2 · E_0 / dE/dx(E_0)            # constant-loss extrapolation
  range(E_j) = range(E_{j-1})
             + Σ_{k=0}^{n-1} de / dE/dx(E_j - (k+0.5)·de)
  where de = (E_j - E_{j-1}) / n, n = 100.

The integrator queries dE/dx through a G4PhysicsVector (so we get the same
spline behaviour Geant4 uses internally).

Scaled kinetic energy / inverse-range:
  scaled_E(R) is just the inverse of E -> range(E), i.e. the table swapped.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from physics.g4_init_tables.interpolator import G4PhysicsVector

_N_SUBSTEPS = 100  # G4 constant


def build_range_table(dedx_vector: G4PhysicsVector) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (E_eV, range_cm) on the (possibly truncated) grid of the dE/dx vector.

    Mirrors G4LossTableBuilder::BuildRangeTable's leading-zero protection: bins
    whose dE/dx is non-positive at the low-E head are dropped before integration.
    """
    bv = dedx_vector.bin
    dv = dedx_vector.data
    bin0 = int((dv > 0).to(torch.int64).argmax())
    if not (dv[bin0:] > 0).all():
        raise ValueError("dE/dx becomes non-positive again above the leading zero block")
    bv = bv[bin0:].clone()
    dv = dv[bin0:].clone()
    n_points = bv.numel()
    if n_points < 3:
        raise ValueError("too few positive dE/dx points to integrate")

    range_t = torch.zeros_like(bv)
    range_t[0] = 2.0 * bv[0] / dv[0]
    inv = 1.0 / _N_SUBSTEPS

    for j in range(1, n_points):
        E_lo = bv[j - 1]
        E_hi = bv[j]
        de = (E_hi - E_lo) * inv
        # midpoints: E_hi - (k+0.5)·de for k = 0..n-1
        ks = torch.arange(_N_SUBSTEPS, dtype=bv.dtype, device=bv.device)
        mids = E_hi - (ks + 0.5) * de
        dedx_mid = dedx_vector(mids)
        seg = (de / dedx_mid).sum()
        range_t[j] = range_t[j - 1] + seg
    return bv.clone(), range_t


def build_scaled_kin_energy(e_grid: torch.Tensor, range_cm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (range_cm, E_eV) — i.e. the inverse range table for direct lookup."""
    return range_cm.clone(), e_grid.clone()


def save_range(e_grid: torch.Tensor, range_cm: torch.Tensor, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "range.npz"
    np.savez_compressed(path, energy_eV=e_grid.cpu().numpy(), range_cm=range_cm.cpu().numpy())
    return path


def save_scaled_kin_energy(range_cm: torch.Tensor, e_grid: torch.Tensor, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "scaled_kin_energy.npz"
    np.savez_compressed(path, range_cm=range_cm.cpu().numpy(), energy_eV=e_grid.cpu().numpy())
    return path
