"""
G4-style log-spaced energy grid (G4PhysicsLogVector). GEANT4's default EM
parameters (G4EmParameters) use 100 eV .. 100 TeV with 7 bins per decade,
i.e. 84 bins / 85 nodes.
"""
from __future__ import annotations

import math

import torch

import common.units as U
from physics.g4_init_tables._g4math import g4_exp_scalar, g4_log_scalar

# PhysListEmStandard (StepsGenerator/src/PhysListEmStandard.cc) overrides the
# G4EmParameters defaults: MinEnergy=10 eV, MaxEnergy=10 TeV, 10 bins/decade.
# These values *must* match the simulation that produced our reference dumps
# (see StepsGenerator/PhDMacro.mac, /run/setCut 1 um, addPhysics local).
G4_E_MIN_DEFAULT = 10.0                   # eV  (PhysListEmStandard MinEnergy)
G4_E_MAX_DEFAULT = 10.0 * U.GeV2eV * 1e3  # 10 TeV in eV
G4_BINS_PER_DECADE = 10


def n_nodes(e_min: float = G4_E_MIN_DEFAULT,
            e_max: float = G4_E_MAX_DEFAULT,
            bins_per_decade: int = G4_BINS_PER_DECADE) -> int:
    n_bins = int(round(bins_per_decade * math.log10(e_max / e_min)))
    return n_bins + 1


def log_energy_grid(e_min: float = G4_E_MIN_DEFAULT,
                    e_max: float = G4_E_MAX_DEFAULT,
                    n: int | None = None,
                    bins_per_decade: int = G4_BINS_PER_DECADE) -> torch.Tensor:
    """Logarithmically spaced energies in eV, matching G4PhysicsLogVector bin
    centres.

    Mirrors G4PhysicsLogVector::G4PhysicsLogVector exactly (lines 47-75 of
    G4PhysicsLogVector.cc): endpoints are stored verbatim (no log/exp
    round-trip on emin/emax), and intermediate nodes are
    `emin * G4Exp(i / invdBin)` with `invdBin = (n-1)/G4Log(emax/emin)`.

    We use the Padé-approximation G4Log/G4Exp (defined in `_g4math.py`) rather
    than libm's `math.exp` / `math.log`, because G4 populates its own
    runtime tables with `G4Exp` — using libm here drifts the interior nodes
    by ±1 ULP at a handful of energies and breaks bit-level parity even
    after the correct grid bounds are matched. Endpoints stay verbatim.
    """
    if n is None:
        n = n_nodes(e_min, e_max, bins_per_decade)
    grid = torch.empty(n, dtype=torch.float64)
    grid[0] = e_min
    grid[-1] = e_max
    inv_dbin = (n - 1) / g4_log_scalar(e_max / e_min)
    for i in range(1, n - 1):
        grid[i] = e_min * g4_exp_scalar(i / inv_dbin)
    return grid
