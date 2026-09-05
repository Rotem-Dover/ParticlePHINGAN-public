"""Canonical relativistic kinematics for charged particles.

Provides _kinematics_jit: a @torch.jit.script function callable from other
JIT-compiled kernels. Uses the G4-faithful tau*(tau+2) numerator for t_max
(matches G4BetheBlochModel::MaxSecondaryEnergy bit-for-bit factor order).
"""
from __future__ import annotations
from typing import Tuple

import torch


@torch.jit.script
def _kinematics_jit(
    kin_eV: torch.Tensor,
    mass_eV: float,
    electron_mass_eV: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (tau, gamma, gamma_sq, beta_sq, t_max) for a heavy charged particle.

    t_max uses the G4BetheBlochModel::MaxSecondaryEnergy factor order:
        2*m_e * tau*(tau+2) / (1 + 2*gamma*r + r²)
    NOT the algebraically-equivalent beta_sq*gamma_sq numerator (differs by
    ~ULP, amplifies through the lambda-ion spline near threshold).

    Args:
        kin_eV: kinetic energy tensor (eV), any shape.
        mass_eV: particle rest mass (eV), Python float.
        electron_mass_eV: electron rest mass (eV), Python float.

    Returns:
        tau       = E_kin / m
        gamma     = tau + 1
        gamma_sq  = gamma²
        beta_sq   = 1 - 1/gamma²
        t_max     = 2 m_e tau(tau+2) / (1 + 2 gamma r + r²)   [heavy particle]
    """
    mass_ratio = electron_mass_eV / mass_eV
    tau = kin_eV / mass_eV
    gamma = tau + 1.0
    gamma_sq = gamma * gamma
    beta_sq = 1.0 - 1.0 / gamma_sq
    t_max = (2.0 * electron_mass_eV * tau * (tau + 2.0)) / (
        1.0 + 2.0 * gamma * mass_ratio + mass_ratio * mass_ratio
    )
    return tau, gamma, gamma_sq, beta_sq, t_max
