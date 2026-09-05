"""
Build the ionisation lambda(E) table — the inverse macroscopic cross section
for delta-ray production above the cut Tc. Port of
G4VEnergyLossProcess::BuildLambdaTable / G4EmTableUtil::BuildLambdaTable for
the high-E model (G4BetheBlochModel).

  σ_vol(E) = N_e · ∫_{Tc}^{Tmax} (dσ/dT) dT      [1/cm]
  λ(E)     = 1 / σ_vol(E)                         [cm]

The integral is the analytic Berger-Seltzer expression already encoded in
G4BraggModel / G4BetheBlochModel::ComputeCrossSectionPerElectron — we reuse
the Bethe-Bloch port for it.

Grid: G4 anchors the lambda table's lower edge at the *threshold energy*
E_thr where T_max(E_p) = Tc (below this, there is no phase space for
delta production). The grid is log-uniform from E_thr to e_max using the
configured bins-per-decade.

Only the proton path is implemented: the electron/positron model
(G4MollerBhabhaModel) is not, so `build_lambda_table` refuses any
non-proton particle rather than silently falling through.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

import physics.definitions.physical_constants as C
from physics.g4_init_tables.grid import G4_BINS_PER_DECADE, log_energy_grid
from physics.g4_init_tables.models.bethe_bloch import (
    compute_cross_section_per_volume_MeV_internal as _bb_xs_MeV,
)
from physics.definitions.material import Material
from physics.definitions.particle import Particle


def _delta_ray_threshold_MeV(particle: Particle, cut_MeV: float) -> float:
    """E_p (MeV) such that the maximum-allowed delta-ray energy equals cut.

    Mirrors G4hIonisation::MinPrimaryEnergy with MeV-magnitude operands —
    matches G4's CLHEP-internal arithmetic to ULP. Hadrons only.
    """
    m_e_MeV = C.electron_mass_eV * 1e-6
    M_MeV = particle.mass_eV * 1e-6
    ratio = m_e_MeV / M_MeV
    x = 0.5 * cut_MeV / m_e_MeV
    gam = x * ratio + math.sqrt((1.0 + x) * (1.0 + x * ratio * ratio))
    return M_MeV * (gam - 1.0)


def build_lambda_table(particle: Particle, material: Material, cut_eV: float,
                       e_min: float, e_max: float, n_nodes: int | None = None
                       ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (e_grid_eV, sigma_per_cm, lambda_cm).

    Only the proton path is implemented: the electron/positron model
    (G4MollerBhabhaModel) is not.

    For hadrons, the grid is generated in **MeV-magnitudes** (matching G4's
    CLHEP-internal arithmetic) and only converted to eV at the very end
    for storage. Generating the grid directly in MeV via G4Exp on
    MeV-magnitude operands is the only way to make the grid energies match
    G4's bit-for-bit; the post-hoc `eV * 1e-6` conversion costs 1 ULP at a
    handful of nodes (e.g., 737378.4537270541 vs the literal
    0.7373784537270541), which is enough to shift the cross section at
    those nodes by a fraction of a ULP relative and amplify through the
    spline into the steep threshold-adjacent bin.
    """
    if particle.name.lower() != "proton":
        raise NotImplementedError(
            f"only proton lambda tables are supported; got {particle.name!r}. "
            "The electron models (Moller/Bhabha) are not implemented.")

    # Hadron: keep the whole arithmetic chain in MeV-magnitudes (G4
    # CLHEP-internal). Generate the threshold AND the grid in MeV from
    # the start so σ at every node matches G4 bit-for-bit.
    cut_MeV = cut_eV * 1e-6
    e_thr_MeV = _delta_ray_threshold_MeV(particle, cut_MeV)
    e_max_MeV = e_max * 1e-6
    if e_thr_MeV >= e_max_MeV:
        raise ValueError(f"delta-ray threshold {e_thr_MeV:.3e} MeV >= e_max")
    if n_nodes is None:
        n_bins = max(int(round(G4_BINS_PER_DECADE * math.log10(e_max_MeV / e_thr_MeV))), 5)
        n_nodes_actual = n_bins + 1
    else:
        n_nodes_actual = n_nodes
    e_grid_MeV = log_energy_grid(e_min=e_thr_MeV, e_max=e_max_MeV, n=n_nodes_actual)
    sigma_vol = _bb_xs_MeV(particle, material, e_grid_MeV, cut_MeV=cut_MeV)
    # Convert grid to eV for storage. The *1e6 conversion may cost 1 ULP
    # at a few nodes vs a directly-eV-generated grid, but downstream
    # code reads this stored grid as-is and remains consistent.
    e_grid = e_grid_MeV * 1e6
    # By definition σ(E_thr)=0 — but the analytic formula evaluated at the
    # numerically inverted threshold may leave a sub-ULP residue, so force
    # the first node exactly. G4 stores 0 (not +inf or epsilon) here; see
    # G4LossTableBuilder::BuildLambdaTable with startFromNull=true.
    sigma_vol[0] = 0.0
    lam = torch.where(sigma_vol > 0, 1.0 / sigma_vol, torch.zeros_like(sigma_vol))
    return e_grid, sigma_vol, lam


def save_lambda(e_grid: torch.Tensor, sigma_per_cm: torch.Tensor,
                lam_cm: torch.Tensor, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "lambda_ion.npz"
    np.savez_compressed(path,
                        energy_eV=e_grid.cpu().numpy(),
                        lambda_cm=lam_cm.cpu().numpy(),
                        sigma_per_cm=sigma_per_cm.cpu().numpy())
    return path
