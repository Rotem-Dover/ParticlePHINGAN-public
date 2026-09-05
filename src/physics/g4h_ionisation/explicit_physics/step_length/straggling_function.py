"""
Implements the step-length distribution model following GEANT4's G4hIonisation process.
The step length is min(L_discrete, L_eloss), where L_discrete is an exponential variate
from the macroscopic cross-section and L_eloss is the deterministic energy-loss limiter.
Provides parameter computation and analytical PDFs. (MC sampling lives in the
vectorized torch generator: `generators/length_generator/physics_mc.py`.)
"""
import math
import logging

import numpy as np
from numpy.typing import NDArray
from typing import Callable
from enum import Enum, auto

import common.units as U
from common.utils import PointInPhaseSpace

from physics.definitions.physical_constants import electron_mass_eV, electron_radius_cm
from common.run_context import PARTICLE as proton, TARGET_MATERIAL as material
from physics.definitions.particle import Particle
from physics.definitions.material import Material
from physics.g4h_ionisation.explicit_physics.continuous._tables import Geant4Tables
from physics.g4h_ionisation.explicit_physics.step_length._cross_section import macroscopic_cross_section
from physics.g4h_ionisation.explicit_physics.step_length._step_length_pdf import truncated_exponential_pdf, deterministic_pdf


log = logging.getLogger(__name__)


class StepLengthType(Enum):
    Deterministic = auto()          # T_max <= Tc: no discrete interactions, step = L_eloss
    TruncatedExponential = auto()   # General case: L = min(Exp(Sigma), L_eloss)


# G4VEnergyLossProcess step-function parameters (SetStepFunction defaults)
DR_OVER_RANGE = 0.1       # fraction of remaining range per step
FINAL_RANGE_CM = 0.01     # 100 um = 0.01 cm


class StepLengthParams(Geant4Tables):
    """
    Parameters for step length determination following G4hIonisation.

    With MSC off and no transportation, the step length is:
        L = min(L_discrete, L_eloss)

    where:
        L_discrete = -ln(U) / Sigma(E)   (PostStep, stochastic)
        L_eloss = f(R, finalRange, dR)    (AlongStep, deterministic)

    The macroscopic cross-section Sigma(E) is computed analytically from the
    Rutherford-like differential cross-section for heavy-particle ionization
    integrated from Tc to T_max.
    """
    def __init__(self,
                 particle: Particle = proton,
                 mat: Material = material,
                 dR_over_range: float = DR_OVER_RANGE,
                 final_range_cm: float = FINAL_RANGE_CM):
        self.particle = particle
        self.material = mat
        self.dR_over_range = dR_over_range
        self.final_range_cm = final_range_cm

        Geant4Tables.__init__(self)

        self.__primary_energy: float = None
        self.step_length_type: StepLengthType = None

    def set_state(self, point_in_phase_space: PointInPhaseSpace) -> None:
        """
        Set the state from kinetic energy. Step length depends only on E.
        Expecting primary_energy in eV.
        """
        self.__primary_energy = point_in_phase_space.primary_energy
        self.step_length_type = self._determine_type()

    @property
    def primary_energy(self) -> float:
        return self.__primary_energy

    @property
    def beta(self) -> float:
        return self.particle.beta(self.__primary_energy)

    @property
    def t_max(self) -> float:
        return self.particle.t_max(self.__primary_energy)

    @property
    def sigma(self) -> float:
        """Macroscopic cross-section for delta-ray production in 1/cm."""
        return macroscopic_cross_section(self.__primary_energy, self.particle, self.material)

    @property
    def mean_free_path_cm(self) -> float:
        """Mean free path lambda = 1/Sigma in cm."""
        s = self.sigma
        if s <= 0:
            return float('inf')
        return 1.0 / s

    @property
    def remaining_range_cm(self) -> float:
        """Remaining CSDA range R(E) in cm."""
        if self.__primary_energy <= 0:
            return 0.0
        R_mm = self.energy_to_range(self.__primary_energy)  # mm
        return R_mm * U.mm2cm

    @property
    def L_eloss(self) -> float:
        """
        Deterministic energy-loss step limit from AlongStep (G4VEnergyLossProcess).

        Implements the step-function formula from G4VEnergyLossProcess::AlongStepGPIL
        (lines 610-614 of G4VEnergyLossProcess.cc):

        If R > finalRange:
            x = R * dRoverRange + finalRange * (1 - dRoverRange) * (2 - finalRange/R)
        Else:
            x = R  (step = full remaining range)

        Parameters are set by SetStepFunction(dRoverRange=0.1, finalRange=100um).
        """
        R = self.remaining_range_cm
        f = self.final_range_cm

        if R <= f:
            return R

        dR = self.dR_over_range
        return R * dR + f * (1 - dR) * (2 - f / R)

    def _determine_type(self) -> StepLengthType:
        if self.t_max <= self.material.Tc:
            return StepLengthType.Deterministic
        return StepLengthType.TruncatedExponential

class StepLengthModel(StepLengthParams):
    """
    Step length distribution model following G4hIonisation (MSC off, no transportation).

    Phase-space regions for 0-100 MeV protons in aluminum:

    1. High energy (100 -> ~5 MeV):
       - L_eloss ~ 0.1*R (mm-scale steps)
       - Sigma moderate, lambda >> L_eloss
       - Distribution ~ delta(L_eloss): energy-loss limiter almost always wins
       - PDF: Sigma*exp(-Sigma*L) for L < L_eloss, point mass exp(-Sigma*L_eloss) at L_eloss

    2. Medium energy (~5 -> ~0.5 MeV):
       - L_eloss shrinks as range decreases
       - Sigma can become comparable to 1/L_eloss
       - Truncated exponential with visible exponential tail before cutoff

    3. Low energy (~0.5 MeV -> T_max = Tc threshold):
       - R approaches finalRange (100 um)
       - L_eloss transitions from formula to R
       - Sigma -> 0 as T_max -> Tc
       - Distribution narrows to delta(L_eloss)

    4. Very low energy (T_max < Tc):
       - No delta-ray production possible (Sigma = 0)
       - Purely deterministic: L = L_eloss = R
    """
    def __init__(self,
                 particle: Particle = proton,
                 mat: Material = material,
                 dR_over_range: float = DR_OVER_RANGE,
                 final_range_cm: float = FINAL_RANGE_CM):
        StepLengthParams.__init__(self, particle, mat, dR_over_range, final_range_cm)

    @property
    def straggling_function(self) -> Callable[[NDArray[float]], NDArray[float]]:
        """
        PDF of the step length distribution.

        Returns a callable f(x) -> pdf(x) where x is step length in cm.
        """
        if self.step_length_type == StepLengthType.Deterministic:
            L_el = self.L_eloss
            return lambda x: deterministic_pdf(x, L_el)

        sigma = self.sigma
        L_el = self.L_eloss
        return lambda x: truncated_exponential_pdf(x, sigma, L_el)

    @property
    def straggling_cdf(self) -> Callable[[NDArray[float]], NDArray[float]]:
        """
        CDF of the step length distribution.
        """
        return lambda x: np.cumsum(self.straggling_function(x)) * (x[1] - x[0])

    @property
    def min_step_length(self) -> float:
        """Minimum meaningful step length (essentially 0)."""
        return 0.0

    @property
    def max_step_length(self) -> float:
        """Maximum possible step length = L_eloss (the deterministic cap)."""
        return self.L_eloss


