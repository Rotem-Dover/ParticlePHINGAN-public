"""
Computes the macroscopic cross-section Sigma(E) for delta-ray production above
the production threshold Tc, from the locally computed lambda_ion init table,
and provides the corresponding mean free path.
"""
import logging

import torch

from physics.g4_init_tables.runtime import get_init_table_set
from common.run_context import PARTICLE as proton, TARGET_MATERIAL as material
from physics.definitions.particle import Particle
from physics.definitions.material import Material

log = logging.getLogger(__name__)


class MacroscopicCrossSectionTable:
    """
    Macroscopic cross-section Sigma(E) for delta-ray production, from the
    locally computed lambda_ion init table. Uses sigma_ion_biased — G4's
    cached preStepLambda (lambdaFactor-biased above the sigma peak) — so the
    analytical step-length PDF stays consistent with the
    PhysicsLengthGenerator MC sampler, which reads the same quantity.
    """
    def __init__(self):
        self._tables = get_init_table_set()

    def __call__(self, kinetic_energy: float) -> float:
        val = float(self._tables.sigma_ion_biased(
            torch.tensor([float(kinetic_energy)], dtype=torch.float64))[0])
        return val if val > 0.0 else 0.0


# Module-level singleton — loaded on first use
_g4_cross_section = MacroscopicCrossSectionTable()


def macroscopic_cross_section(kinetic_energy: float,
                              particle: Particle = proton,
                              mat: Material = material) -> float:
    """
    Macroscopic cross-section Sigma(E) for delta-ray production above Tc.

    Uses values interpolated from the preStepLambda tables.

    Returns
    -------
    float
        Macroscopic cross-section in 1/cm. Returns 0 if T_max <= Tc.
    """
    kinetic_energy = kinetic_energy
    t_max = particle.t_max(kinetic_energy)
    if t_max <= mat.Tc:
        return 0.0

    return _g4_cross_section(kinetic_energy)


def mean_free_path(kinetic_energy: float,
                   particle: Particle = proton,
                   mat: Material = material) -> float:
    """
    Mean free path lambda = 1/Sigma(E) for delta-ray production.

    Returns
    -------
    float
        Mean free path in cm. Returns float('inf') if Sigma = 0.
    """
    sigma = macroscopic_cross_section(kinetic_energy, particle, mat)
    if sigma <= 0:
        return float('inf')
    return 1.0 / sigma
