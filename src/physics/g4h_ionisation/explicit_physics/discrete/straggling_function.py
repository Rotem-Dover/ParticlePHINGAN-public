"""
Implements the discrete energy-loss straggling model for secondary electron (delta-ray)
production. Computes the straggling function parameters from particle/material properties
and provides the analytical PDF for the energy transferred to individual
knocked-out electrons above the production threshold Tc. (MC sampling lives in
the vectorized torch kernels: `discrete/_kernels.py`.)
"""
import numpy as np

from numpy.typing import NDArray
from typing       import Callable
from enum         import Enum, auto

from common.run_context import PARTICLE as proton, TARGET_MATERIAL as material
from physics.definitions.particle import Particle
from physics.definitions.material import Material
from common.utils import PointInPhaseSpace

from physics.g4h_ionisation.explicit_physics.discrete._ionization_straggling_function import secondary_straggling_function, moller_straggling_function


# TODO: This has not yet been compared to the GEANT4 samples.


class StragglingType(Enum):
    SecondaryGeneration = auto()
    # Add here more straggling types if needed


class DiscreteLossParams:
    def __init__(
            self, particle: Particle = proton, material: Material = material
    ):
        self.particle = particle
        self.material = material

        self.__primary_energy: float = None

    def set_state(self, point_in_phase_space: PointInPhaseSpace):
        """
        Set the state of the continuous loss model.
        Expecting step_length in cm and primary_energy in eV.
        """
        self.__primary_energy = point_in_phase_space.primary_energy

        # Verify that valid inputs are provided
        assert self.t_max >= self.material.Tc, f"t_max must be greater than Tc. Got {self.t_max} and {self.material.Tc}."

    @property
    def primary_energy(self) -> float:
        return self.__primary_energy

    @property
    def beta(self) -> float:
        return self.particle.beta(self.__primary_energy)

    @property
    def w(self) -> float:
        return self.t_max / self.material.Tc - 1

    @property
    def t_max(self) -> float:
        return self.particle.t_max(self.__primary_energy)

    @property
    def w3(self) -> float:
        return self.t_max

class DiscreteStragglingModel(DiscreteLossParams):
    """
    Straggling function of the discrete energy loss model.
    TODO: Verify this implementation fits GEANT4 data
    """
    def __init__(
            self, particle: Particle = proton, material: Material = material
    ):
        DiscreteLossParams.__init__(self, particle, material)

    @property
    def min_ion_energy_transfer(self) -> float:
        """
        Minimum energy loss in a single ionization collision.
        """
        return self.w3 / (1 + self.w)

    @property
    def max_ion_energy_transfer(self) -> float:
        """
        Maximum energy loss in a single ionization collision.
        """
        return self.w3

    @property
    def is_moller(self) -> bool:
        """
        Electron primaries scatter off identical atomic electrons, so the
        delta-ray spectrum is Møller (G4MollerBhabhaModel), not the
        heavy-particle Rutherford tail. Gated on the same condition as the
        Møller branch of the MC sampler (`physics_mc._IS_ELECTRON`).
        """
        return self.particle.name == "electron"

    @property
    def regime_label(self) -> str:
        return "Moller z(x)/x^2" if self.is_moller else "Rutherford 1/E^2"

    @property
    def straggling_function(self) -> Callable[[NDArray[float]], NDArray[float]]:
        """
        Total energy loss straggling function.
        """
        return self._ionization_straggling_function

    @property
    def straggling_cdf(self) -> Callable[[NDArray[float]], NDArray[float]]:
        """
        Compute the cumulative distribution function (CDF) of the straggling function.
        """
        return lambda x: np.cumsum(self.straggling_function(x)) * (x[1] - x[0])

    @property
    def _ionization_straggling_function(self) -> Callable[[NDArray[float]], NDArray[float]]:
        """
        Straggling function of energy loss by generation of secondary electrons through ionization.
        """
        if self.is_moller:
            return lambda x: moller_straggling_function(
                x, self.primary_energy, self.material.Tc,
                self.particle.gamma(self.primary_energy),
            )
        return lambda x: secondary_straggling_function(x, self.w3, self.w, self.beta)


