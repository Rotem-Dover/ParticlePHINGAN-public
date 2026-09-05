"""
Port of G4BetheBlochModel::ComputeDEDXPerVolume and CrossSectionPerVolume
for hadrons (proton). All inputs/outputs in eV and cm.

Reference: geant4-11.2.0/source/processes/electromagnetic/standard/src/
G4BetheBlochModel.cc

Note: ICRU90 stopping data is not implemented; Barkas/HighOrderCorrections
and shell correction are implemented separately (em_corrections.py,
shell_correction.py). Density correction uses the Sternheimer
parameterisation with per-material parameters.
"""
from __future__ import annotations

import math

import numpy as np
import torch

import physics.definitions.physical_constants as C
from physics.g4_init_tables._g4math import g4_log, g4_log_scalar
from physics.definitions.material import Material
from physics.definitions.particle import Particle

# 2π · m_e c² · r_e²  expressed in [eV · cm²]
TWOPI_MC2_RCL2 = 2.0 * math.pi * C.electron_mass_eV * C.electron_radius_cm ** 2
# G4 uses `2 * G4Log(10.)`, not libm; the ULP-level difference propagates
# through the density correction and shifts dE/dx in the Bethe-Bloch
# regime.
TWO_LN10 = 2.0 * g4_log_scalar(10.0)


class SternheimerParams:
    """Density-effect parameterisation parameters (PDG / G4 NIST)."""
    def __init__(self, x0: float, x1: float, m: float, a: float, C_: float, delta0: float = 0.0):
        self.x0 = x0; self.x1 = x1; self.m = m; self.a = a; self.C = C_; self.delta0 = delta0

    def __call__(self, bg2: torch.Tensor) -> torch.Tensor:
        # G4: x = G4Log(bg2)/twoln10 == log10(β·γ) — uses G4Log not libm.
        x = g4_log(bg2) / TWO_LN10
        delta = torch.zeros_like(bg2)
        # x >= x1
        m1 = x >= self.x1
        delta[m1] = TWO_LN10 * x[m1] - self.C
        # x0 <= x < x1
        m2 = (x >= self.x0) & ~m1
        delta[m2] = TWO_LN10 * x[m2] - self.C + self.a * (self.x1 - x[m2]) ** self.m
        # x < x0 (for conductors, δ0 · 10^(2(x-x0)))
        m3 = x < self.x0
        if self.delta0 > 0:
            delta[m3] = self.delta0 * torch.pow(torch.tensor(10.0, dtype=bg2.dtype, device=bg2.device), 2.0 * (x[m3] - self.x0))
        return delta


# Sternheimer parameters — matched to G4 NIST values exactly
# (G4DensityEffectData.cc:149 for Al; same source for Si, Fe, Be), extracted
# by scripts/extract_g4_material_constants.py.
_STERNHEIMER = {
    "Aluminum": SternheimerParams(x0=0.1708, x1=3.0127, m=3.6345, a=0.08024, C_=4.2395, delta0=0.12),
    "Silicon":  SternheimerParams(x0=0.2014, x1=2.8715, m=3.2546, a=0.14921, C_=4.4355, delta0=0.14),
    "Iron": SternheimerParams(x0=-0.0012, x1=3.1531, m=2.9632, a=0.1468, C_=4.2911, delta0=0.12),
    "Beryllium": SternheimerParams(x0=0.0392, x1=1.6922, m=2.4339, a=0.80392, C_=2.7847, delta0=0.14),
}


def _kinematics(particle: Particle, kin_E_eV: torch.Tensor):
    """Return (tau, gamma, beta2, bg2) tensors."""
    mass = particle.mass_eV
    tau = kin_E_eV / mass
    gam = tau + 1.0
    bg2 = tau * (tau + 2.0)
    beta2 = bg2 / (gam * gam)
    return tau, gam, beta2, bg2


def t_max(particle: Particle, kin_E_eV: torch.Tensor) -> torch.Tensor:
    """Max single-collision energy transfer (G4VEmModel::MaxSecondaryEnergy).

    The factor order matches G4BetheBlochModel::MaxSecondaryEnergy
    (line 493-495 of G4BetheBlochModel.cc) bit-for-bit:
        `2*m_e * tau * (tau+2) / (1 + 2*(tau+1)*ratio + ratio²)`
    NOT the algebraically-equivalent `2*m_e * bg2 / D` (where bg2 was
    already computed). The two associations differ by a ULP, which
    propagates through the ionisation lambda table's spline and amplifies
    into the steep threshold-adjacent bin.
    """
    r = particle.electron_mass_ratio
    mass = particle.mass_eV
    tau = kin_E_eV / mass
    return (2.0 * C.electron_mass_eV * tau * (tau + 2.0)) \
           / (1.0 + 2.0 * (tau + 1.0) * r + r * r)


def compute_dedx_per_volume(
    particle: Particle, material: Material, kin_E_eV: torch.Tensor, cut_eV: float,
    corrections: bool = True,
) -> torch.Tensor:
    """G4BetheBlochModel::ComputeDEDXPerVolume in eV/cm.

    `corrections=True` adds the Barkas/Bloch/Mott high-order corrections
    (em_corrections.py) AND the K/L/M/N shell correction (shell_correction.py).
    These are correlated: G4 always applies both together, and HOC alone
    (without shell) leaves a systematic ~+1% bias at 1-10 MeV.
    """
    tmax = t_max(particle, kin_E_eV)
    cut = torch.minimum(torch.full_like(kin_E_eV, cut_eV), tmax)

    _, gam, beta2, bg2 = _kinematics(particle, kin_E_eV)
    xc = cut / tmax

    eexc = material.I  # eV
    eexc2 = eexc * eexc
    eDensity = material.electron_density  # 1/cm³

    # general Bethe-Bloch (G4Log to match G4BetheBlochModel.cc:188)
    dedx = g4_log(2.0 * C.electron_mass_eV * bg2 * cut / eexc2) - (1.0 + xc) * beta2

    # spin-1/2 correction (proton): + (0.5 * cut/(E + m))²
    if particle.spin > 0.0:
        del_ = 0.5 * cut / (kin_E_eV + particle.mass_eV)
        dedx = dedx + del_ * del_

    # density correction
    # Hard lookup on purpose: a missing entry would silently skip the
    # density correction and produce wrong dE/dx with no complaint, so an
    # unknown material name raises KeyError immediately.
    stern = _STERNHEIMER[material.name]
    dedx = dedx - stern(bg2)

    # G4 subtracts 2·ShellCorrection inside the bracket (before the
    # twopi_mc2_rcl2·z²·n_e/β² scaling).
    if corrections:
        from physics.g4_init_tables.models.shell_correction import shell_correction
        dedx = dedx - 2.0 * shell_correction(particle, material, kin_E_eV)

    dedx = dedx * TWOPI_MC2_RCL2 * (particle.charge ** 2) * eDensity / beta2

    if corrections:
        from physics.g4_init_tables.models.em_corrections import (
            high_order_corrections as _hoc,
        )
        dedx = dedx + _hoc(particle, material, kin_E_eV)
    return torch.clamp(dedx, min=0.0)


def compute_cross_section_per_volume_MeV_internal(
    particle: Particle, material: Material, kin_E_MeV: torch.Tensor,
    cut_MeV: float, max_E_MeV: float | None = None,
) -> torch.Tensor:
    """G4-bit-exact σ_per_cm. Inputs in MeV-magnitudes (matching G4 CLHEP).

    Same physics as `compute_cross_section_per_volume` but evaluates the
    kinematics in MeV-magnitude operands — beta2/tmax differ by 1 ULP from
    the eV-magnitude formulation, and the difference amplifies through the
    lambda-ion spline near the production threshold. Called from
    build_lambda_ion, where the grid is generated in MeV directly, so the
    cross section is computed in the same magnitude as its own grid.
    """
    m_e_MeV = C.electron_mass_eV * 1e-6  # equals literal 0.510998910 exactly
    mass_MeV = particle.mass_eV * 1e-6   # equals literal exactly when mass_eV is integer-valued
    ratio = m_e_MeV / mass_MeV
    tau = kin_E_MeV / mass_MeV
    tmax_MeV = (2.0 * m_e_MeV * tau * (tau + 2.0)) \
               / (1.0 + 2.0 * (tau + 1.0) * ratio + ratio * ratio)

    if max_E_MeV is None:
        max_MeV = tmax_MeV
    else:
        max_MeV = torch.minimum(tmax_MeV, torch.full_like(tmax_MeV, max_E_MeV))
    cut_t = torch.full_like(kin_E_MeV, cut_MeV)

    totE = kin_E_MeV + mass_MeV
    energy2 = totE * totE
    beta2 = kin_E_MeV * (kin_E_MeV + 2.0 * mass_MeV) / energy2

    cross = (max_MeV - cut_t) / (cut_t * max_MeV) - beta2 * g4_log(max_MeV / cut_t) / tmax_MeV
    if particle.spin > 0.0:
        cross = cross + 0.5 * (max_MeV - cut_t) / energy2

    valid = cut_t < max_MeV
    cross = torch.where(valid, cross, torch.zeros_like(cross))
    cross = cross * (TWOPI_MC2_RCL2 * (particle.charge ** 2) / beta2)
    # `cross` has units (1/MeV) * (eV·cm²) = (1e-6 cm²/MeV) * MeV = 1e-6 cm²
    # Multiplied by electron_density (1/cm³) → 1e-6 / cm. Scale by 1e-6
    # to recover σ in 1/cm.
    return cross * material.electron_density * 1e-6


def compute_cross_section_per_volume(
    particle: Particle, material: Material, kin_E_eV: torch.Tensor,
    cut_eV: float, max_E_eV: float | None = None,
) -> torch.Tensor:
    """
    G4BetheBlochModel::CrossSectionPerVolume (above the production cut Tc).
    Returns σ_vol in 1/cm.  λ(E) = 1/σ_vol(E).

    The arithmetic order mirrors G4BetheBlochModel::ComputeCrossSectionPerElectron
    (G4BetheBlochModel.cc:184-212). G4 evaluates these expressions in
    MeV-magnitude operands (CLHEP-internal, MeV=1), while this function
    works in eV. The two paths differ by a ULP in beta2 and tmax — a
    precision floor that persists unless values are stored as MeV literals
    rather than converted from eV — and that ULP amplifies through the
    lambda-ion spline in the bin between the production threshold and the
    first node. `compute_cross_section_per_volume_MeV_internal` avoids the
    conversion by working in MeV throughout.
    """
    mass = particle.mass_eV
    tmax = t_max(particle, kin_E_eV)
    if max_E_eV is None:
        max_energy = tmax
    else:
        max_energy = torch.minimum(tmax, torch.full_like(tmax, max_E_eV))
    cut = torch.full_like(kin_E_eV, cut_eV)

    energy = kin_E_eV + mass
    energy2 = energy * energy
    beta2 = kin_E_eV * (kin_E_eV + 2.0 * mass) / energy2

    cross = (max_energy - cut) / (cut * max_energy) - beta2 * g4_log(max_energy / cut) / tmax
    if particle.spin > 0.0:
        cross = cross + 0.5 * (max_energy - cut) / energy2

    valid = cut < max_energy
    cross = torch.where(valid, cross, torch.zeros_like(cross))
    cross = cross * (TWOPI_MC2_RCL2 * (particle.charge ** 2) / beta2)
    return cross * material.electron_density  # 1/cm
