"""
Port of G4EmCorrections high-order corrections to the Bethe-Bloch dE/dx:
Barkas (Z³), Bloch (Z⁴), Mott (Zα). These are the `HighOrderCorrections`
contribution added by G4BetheBlochModel::ComputeDEDXPerVolume for non-ion
hadrons (proton, etc.).

Reference: geant4-11.2.0/source/processes/electromagnetic/utils/src/
G4EmCorrections.cc:
  HighOrderCorrections (line 172)
  BarkasCorrection     (line 635)
  BlochCorrection      (line 685)
  MottCorrection       (line 710)
  Initialise           (line 884)  — defines sBarkasCorr / sThetaK / sThetaL

Shell correction is implemented separately, in `shell_correction.py`, and
applied by `bethe_bloch.compute_dedx_per_volume` alongside these corrections.
"""
from __future__ import annotations

import math

import torch

import physics.definitions.physical_constants as C
from physics.definitions.material import Material
from physics.definitions.particle import Particle
from physics.g4_init_tables.interpolator import G4PhysicsVector
from physics.g4_init_tables.models.bethe_bloch import TWOPI_MC2_RCL2, _kinematics

# G4 source: G4EmCorrections::Initialise, fTable[47][2] (W, value).
_BARKAS_TABLE = torch.tensor([
    [0.02, 21.5], [0.03, 20.0], [0.04, 18.0], [0.05, 15.6], [0.06, 15.0],
    [0.07, 14.0], [0.08, 13.5], [0.09, 13.0], [0.10, 12.2], [0.20, 9.25],
    [0.30, 7.0],  [0.40, 6.0],  [0.50, 4.5],  [0.60, 3.5],  [0.70, 3.0],
    [0.80, 2.5],  [0.90, 2.0],  [1.00, 1.7],  [1.20, 1.2],  [1.30, 1.0],
    [1.40, 0.86], [1.50, 0.7],  [1.60, 0.61], [1.70, 0.52], [1.80, 0.5],
    [1.90, 0.43], [2.00, 0.42], [2.10, 0.3],  [2.40, 0.2],  [3.00, 0.13],
    [3.08, 0.10], [3.10, 0.09], [3.30, 0.08], [3.50, 0.07], [3.80, 0.06],
    [4.00, 0.051],[4.10, 0.04], [4.80, 0.03], [5.00, 0.024],[5.10, 0.02],
    [6.00, 0.013],[6.50, 0.01], [7.00, 0.009],[7.10, 0.008],[8.00, 0.006],
    [9.00, 0.0032],[10.0, 0.0025],
], dtype=torch.float64)

_W_MAX_BARKAS = 10.0
_BARKAS_INTERP = G4PhysicsVector(_BARKAS_TABLE[:, 0], _BARKAS_TABLE[:, 1],
                                 log_grid=False, use_spline=False)


# b-parameter (Barkas screening) per Z, from G4EmCorrections::BarkasCorrection.
def _barkas_b(Z: int, material_name: str) -> float:
    if Z == 1:
        return 0.6 if material_name == "G4_lH2" else 1.8
    if Z == 2:
        return 0.6
    if Z <= 10:
        return 1.8
    if Z <= 17:
        return 1.4
    if Z == 18:
        return 1.8
    if Z <= 25:
        return 1.4
    if Z <= 50:
        return 1.35
    return 1.3


def barkas_correction(particle: Particle, material: Material,
                      kin_E_eV: torch.Tensor) -> torch.Tensor:
    """
    Per-volume Barkas Z³ correction in eV/cm to add to dE/dx (already
    multiplied by 2π·m_e·r_e²·z²·n_e/β², matching G4EmCorrections::
    HighOrderCorrections's `sum *= ...`).
    """
    _, _, beta2, _bg2 = _kinematics(particle, kin_E_eV)
    beta = torch.sqrt(beta2)
    # G4 SetupKinematics: ba2 = beta2 / alpha²
    alpha2 = C.alpha ** 2  # CLHEP/G4 derives fine_structure_const from SI, not literal
    ba2 = beta2 / alpha2
    Z = int(round(material.Z))
    if Z == 47:
        BarkasTerm = 0.006812 * torch.pow(beta, -0.9)
    elif Z >= 64:
        BarkasTerm = 0.002833 * torch.pow(beta, -1.2)
    else:
        b = _barkas_b(Z, material.name)
        X = ba2 / Z
        W = b / torch.sqrt(X)
        val = _BARKAS_INTERP(W)
        # G4 caps the table at W=10 by scaling val *= Wmax/W
        val = torch.where(W > _W_MAX_BARKAS, val * (_W_MAX_BARKAS / W), val)
        BarkasTerm = val / (torch.sqrt(Z * X) * X)

    # G4: BarkasTerm *= 1.29 * charge / N_atoms_per_volume; then multiplied
    # per-atom density in HighOrderCorrections via atomDensity[i] is per-element.
    # For a mono-element material, atomDensity = N_atoms_per_volume, so the
    # /N_atoms_per_volume cancels.
    BarkasTerm = BarkasTerm * 1.29 * particle.charge
    # HighOrderCorrections then does: sum *= electron_density · q² · 2π·m_e·r_e² / β²
    coeff = material.electron_density * (particle.charge ** 2) * TWOPI_MC2_RCL2 / beta2
    # The HighOrderCorrections multiplier is 2·(Barkas + Bloch) + Mott; this
    # function returns 2·Barkas·coeff. (Caller sums Bloch and Mott.)
    return 2.0 * BarkasTerm * coeff


def bloch_correction(particle: Particle, kin_E_eV: torch.Tensor,
                     n_terms_max: int = 200) -> torch.Tensor:
    """
    Z⁴ Bloch correction, matching G4EmCorrections::BlochCorrection
    (G4EmCorrections.cc:685-705) exactly — including its `del > 0.01*term`
    early-exit. Summing the series to full convergence instead of truncating
    at that criterion gives a numerically different, more negative Bloch
    term. The truncation is by design in G4; reproducing it exactly is
    required for parity, not an approximation to relax.
    """
    _, _, beta2, _ = _kinematics(particle, kin_E_eV)
    alpha2 = C.alpha ** 2  # CLHEP/G4 derives fine_structure_const from SI, not literal
    ba2 = beta2 / alpha2
    q2 = particle.charge ** 2
    y2 = q2 / ba2

    # term_1 = 1/(1+y2), then iterate j=2,3,... until del <= 0.01*term.
    term = 1.0 / (1.0 + y2)
    active = torch.ones_like(term, dtype=torch.bool)
    j = 1.0
    for _ in range(n_terms_max):
        j += 1.0
        delta = 1.0 / (j * (j * j + y2))
        # G4 loop body always adds the term then checks `del > 0.01*term` to
        # decide whether to continue. Once `delta <= 0.01*term` the loop
        # exits without including any *further* terms beyond the current one.
        term = torch.where(active, term + delta, term)
        active = active & (delta > 0.01 * term)
        if not bool(active.any()):
            break
    return -y2 * term


def mott_correction(particle: Particle, kin_E_eV: torch.Tensor) -> torch.Tensor:
    """G4EmCorrections::MottCorrection = π · α · β · z."""
    _, _, beta2, _ = _kinematics(particle, kin_E_eV)
    beta = torch.sqrt(beta2)
    return math.pi * C.alpha * beta * particle.charge


def high_order_corrections(particle: Particle, material: Material,
                           kin_E_eV: torch.Tensor) -> torch.Tensor:
    """
    G4EmCorrections::HighOrderCorrections for a non-ion hadron, in eV/cm.
    `sum = 2·(Barkas + Bloch) + Mott`, then multiplied by 2π·m_e·r_e²·z²·n_e/β².
    """
    _, _, beta2, _ = _kinematics(particle, kin_E_eV)
    barkas_2x = barkas_correction(particle, material, kin_E_eV)  # already 2·Barkas·coeff
    bloch = bloch_correction(particle, kin_E_eV)
    mott = mott_correction(particle, kin_E_eV)
    coeff = material.electron_density * (particle.charge ** 2) * TWOPI_MC2_RCL2 / beta2
    return barkas_2x + (2.0 * bloch + mott) * coeff
