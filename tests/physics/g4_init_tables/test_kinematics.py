"""
Parity and correctness tests for the canonical _kinematics_jit helper.
"""
from __future__ import annotations
import torch
from physics.definitions.particle import proton
from physics.definitions.physical_constants import electron_mass_eV
from physics.g4_init_tables.models.bethe_bloch import t_max as bb_t_max


def _energies():
    """Log-spaced energies from 1 MeV to 1 GeV (eV)."""
    return torch.logspace(6, 9, steps=100, dtype=torch.float64)


def test_t_max_matches_bethe_bloch():
    """_kinematics_jit t_max must match bethe_bloch.t_max within float64 tolerance."""
    from physics.g4_init_tables.models.kinematics import _kinematics_jit
    E = _energies()
    _, _, _, _, tmax_jit = _kinematics_jit(E, proton.mass_eV, electron_mass_eV)
    tmax_ref = bb_t_max(proton, E)
    torch.testing.assert_close(tmax_jit, tmax_ref, rtol=1e-12, atol=0.0)


def test_old_formula_agrees_new_formula():
    """beta_sq*gamma_sq and tau*(tau+2) agree to within float64 machine epsilon."""
    from physics.g4_init_tables.models.kinematics import _kinematics_jit
    E = _energies()
    tau, gamma, gamma_sq, beta_sq, tmax_new = _kinematics_jit(
        E, proton.mass_eV, electron_mass_eV
    )
    # Old formula used in the JIT functions being replaced:
    mass_ratio = electron_mass_eV / proton.mass_eV
    tmax_old = (2.0 * electron_mass_eV * beta_sq * gamma_sq) / (
        1.0 + 2.0 * gamma * mass_ratio + mass_ratio * mass_ratio
    )
    torch.testing.assert_close(tmax_new, tmax_old, rtol=1e-13, atol=0.0)


def test_kinematics_100mev_proton():
    """Known physics: 100 MeV proton kinematics match expected values."""
    from physics.g4_init_tables.models.kinematics import _kinematics_jit
    E = torch.tensor([100e6], dtype=torch.float64)  # 100 MeV in eV
    tau, gamma, gamma_sq, beta_sq, tmax = _kinematics_jit(
        E, proton.mass_eV, electron_mass_eV
    )
    expected_tau = 100e6 / proton.mass_eV
    assert abs(tau.item() - expected_tau) < 1e-10
    assert abs(gamma.item() - (expected_tau + 1.0)) < 1e-10
    assert abs(gamma_sq.item() - (expected_tau + 1.0) ** 2) < 1e-10
    assert beta_sq.item() > 0.0
    assert beta_sq.item() < 1.0
    # t_max should be in the range ~0.1-1 MeV for 100 MeV proton
    assert 1e5 < tmax.item() < 1e6


def test_gamma_sq_and_beta_sq_consistent():
    """gamma_sq * (1 - beta_sq) == 1 everywhere (Lorentz identity)."""
    from physics.g4_init_tables.models.kinematics import _kinematics_jit
    E = _energies()
    _, _, gamma_sq, beta_sq, _ = _kinematics_jit(E, proton.mass_eV, electron_mass_eV)
    identity = gamma_sq * (1.0 - beta_sq)
    torch.testing.assert_close(identity, torch.ones_like(identity), rtol=1e-13, atol=0.0)


def test_electron_kinematics():
    """_kinematics_jit works for electron mass too (moller_bhabha use case)."""
    from physics.g4_init_tables.models.kinematics import _kinematics_jit
    E = torch.logspace(4, 7, steps=50, dtype=torch.float64)  # 10 keV – 10 MeV
    tau, gamma, gamma_sq, beta_sq, _ = _kinematics_jit(
        E, electron_mass_eV, electron_mass_eV
    )
    assert torch.all(tau > 0)
    assert torch.all(gamma > 1.0)
    assert torch.all((beta_sq > 0.0) & (beta_sq < 1.0))
