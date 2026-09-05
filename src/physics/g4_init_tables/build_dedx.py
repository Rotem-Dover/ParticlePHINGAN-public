"""
Build G4-style dE/dx table by evaluating G4BetheBlochModel on a log-spaced
energy grid (port of G4VEnergyLossProcess::BuildDEDXTable for the high-E
EM model). The low-E Bragg model is composed in below eth.

Output: (E_eV, dE/dx_eV_per_cm) saved as .npy and the G4PhysicsVector
spline serialised via numpy.savez_compressed for downstream consumers.

Only the proton dE/dx path (Bragg + Bethe-Bloch) is implemented; the
electron/positron model (G4MollerBhabhaModel) is not, so `build_dedx_table`
refuses any non-proton particle rather than silently falling through.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from physics.g4_init_tables.grid import log_energy_grid
from physics.g4_init_tables.interpolator import G4PhysicsVector
from physics.g4_init_tables.models.bethe_bloch import compute_dedx_per_volume as _bb_dedx
from physics.g4_init_tables.models.bragg import compute_dedx_per_volume as _bragg_dedx
from physics.definitions.material import Material
from physics.definitions.particle import Particle

import physics.definitions.physical_constants as C


def _build_hadron_dedx(particle: Particle, material: Material, e_grid: torch.Tensor,
                       cut_eV: float) -> torch.Tensor:
    """
    Compose G4BraggModel (E < eth) and G4BetheBlochModel (E >= eth) as
    G4hIonisation does. eth = 2 MeV * mass / m_proton.

    G4 smooths the model boundary in G4EmModelManager::FillDEDXVector by
    rescaling the high-E model so dEdx is continuous at eth:

        del  = (dedx_low(eth) / dedx_high(eth) - 1) * eth
        dedx_high(E) ← (1 + del/E) * dedx_high(E)        for E >= eth

    The correction decays as 1/E so it vanishes at very high E; without it,
    the two models disagree at eth and dE/dx has a discontinuous step there.
    """
    eth = 2e6 * particle.mass_eV / C.proton_mass_eV  # G4hIonisation eth
    bragg = _bragg_dedx(particle, material, e_grid, cut_eV)
    bethe = _bb_dedx(particle, material, e_grid, cut_eV)

    eth_t = torch.tensor([eth], dtype=e_grid.dtype, device=e_grid.device)
    bragg_eth = float(_bragg_dedx(particle, material, eth_t, cut_eV).item())
    bethe_eth = float(_bb_dedx(particle, material, eth_t, cut_eV).item())
    del_factor = (bragg_eth / bethe_eth - 1.0) * eth if bethe_eth > 0.0 else 0.0
    bethe_corrected = (1.0 + del_factor / e_grid) * bethe

    return torch.where(e_grid < eth, bragg, bethe_corrected)


def build_dedx_table(particle: Particle, material: Material, cut_eV: float,
                     e_min: float, e_max: float, n_nodes: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build the dE/dx table for the hadron (Bragg + Bethe-Bloch) path (see
    `_build_hadron_dedx` for the boundary smoothing).

    Builds proton tables only: the electron/positron path
    (G4MollerBhabhaModel) is not implemented.
    """
    if particle.name.lower() != "proton":
        raise NotImplementedError(
            f"only proton dE/dx tables are supported; got {particle.name!r}. "
            "The electron models (Moller/Bhabha) are not implemented.")
    e_grid = log_energy_grid(e_min=e_min, e_max=e_max, n=n_nodes)
    dedx = _build_hadron_dedx(particle, material, e_grid, cut_eV)
    return e_grid, dedx


def save_dedx(e_grid: torch.Tensor, dedx: torch.Tensor, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "dedx.npz"
    np.savez_compressed(path, energy_eV=e_grid.cpu().numpy(), dedx_eV_per_cm=dedx.cpu().numpy())
    return path


def make_g4_vector(e_grid: torch.Tensor, dedx: torch.Tensor) -> G4PhysicsVector:
    return G4PhysicsVector(e_grid, dedx, log_grid=True, use_spline=True)
