"""
Implements the continuous energy-loss straggling model following GEANT4's modified-Urban
fluctuation scheme. Computes excitation and ionization loss parameters from particle/material
properties, selects between low- and high-collision-frequency regimes, and produces the
combined analytical PDF by convolving excitation and ionization contributions.
"""
import numpy as np
import math
import logging

from numpy.typing import NDArray
from typing       import Callable
from enum         import Enum, auto

import common.units as U
from common.utils import PointInPhaseSpace, convolve_2_pdfs

from physics.definitions.physical_constants                                            import electron_mass_eV, electron_radius_cm
from common.run_context                                                    import PARTICLE as proton, TARGET_MATERIAL as material
from physics.definitions.particle                                                      import Particle
from physics.definitions.material                                                      import Material
from physics.g4h_ionisation.explicit_physics.continuous._utils                              import truncated_gaussian
from physics.g4h_ionisation.explicit_physics.continuous._tables                             import Geant4Tables
from physics.g4h_ionisation.explicit_physics.continuous._excitation_straggling_function     import few_excitation_collisions_pdf, many_excitation_collisions_pdf
from physics.g4h_ionisation.explicit_physics.continuous._ionization_straggling_function     import few_ionization_collisions_pdf, many_ionization_collisions_pdf


# TODO: This has not yet been compared to the GEANT4 samples.


class StragglingType(Enum):
    DeterministicBetheBloch = auto()
    ThickLimit              = auto()
    LowColFreqExc           = auto()
    HighColFreqExc          = auto()
    LowColFreqIon           = auto()
    HighColFreqIon          = auto()


class GeantConstants:
    def __init__(self):
        self.__E0:                    float = 10.   # eV (Outer electron ionization threshold, a universal approx constant that parametrizes the 1/E^2 Rutherford tail of the energy loss distribution)
        self.__n_threshold:           int   = 8     # number of ionization events threshold for model selection
        self.__ionization_rate:       float = 0.56  # rate (fraction) of the ionization part of the total loss (excitation+ionization)
        self.__linear_loss_limit:     float = 0.01
        self.__min_kin_energy:        float = 0.1 * U.keV2eV
        self.__second_exc_level_frac: float = 0
        self.__first_exc_level_frac:  float = 1 - self.__second_exc_level_frac
        self.__min_energy_loss:       float = 10.  # eV
        self.__n_min_bohr:            int   = 10
        self.__A0:                    float = 42  # TODO: Change to a more meaningful name. Is it float or int?
        self.__FW:                    float = 4   # TODO: Change to a more meaningful name. Is it float or int

    @property
    def min_energy_loss(self) -> float:
        return self.__min_energy_loss

    @property
    def n_min_bohr(self) -> int:
        return self.__n_min_bohr

    @property
    def e0(self) -> float:
        return self.__E0

    @property
    def nmaxCont(self) -> int:
        return self.__n_threshold

    @property
    def ionization_rate(self) -> float:
        return self.__ionization_rate

    @property
    def linear_loss_limit(self) -> float:
        return self.__linear_loss_limit

    @property
    def min_kin_energy(self) -> float:
        return self.__min_kin_energy

    @property
    def first_exc_level_frac(self) -> float:
        return self.__first_exc_level_frac

    @property
    def second_exc_level_frac(self) -> float:
        return self.__second_exc_level_frac

    @property
    def A0(self) -> float:
        return self.__A0

    @property
    def FW(self) -> float:
        return self.__FW


class ContinuousLossParams(GeantConstants, Geant4Tables):
    def __init__(self, particle: Particle = proton, material: Material = material):
        self.particle = particle
        self.material = material

        # Tables come from the locally computed init tables (InitTableSet)
        # for the active (particle, material) set in common.run_context.
        GeantConstants.__init__(self)
        Geant4Tables.__init__(self)

        self.__step_length:    float = None
        self.__primary_energy: float = None

    def set_state(self, point_in_phase_space: PointInPhaseSpace) -> None:
        """
        Set the state of the continuous loss model.
        Expecting step_length in cm and primary_energy in eV.
        """
        self._reset_state()
        self.straggling_type = []

        step_length, primary_energy = point_in_phase_space.step_length, point_in_phase_space.primary_energy

        self.__step_length    = step_length
        self.__primary_energy = primary_energy

        self.set_fluct_params()

    def _reset_state(self):
        # Reset all attributes that may be set by set_fluct_params/set_glandz_params
        attrs_to_delte = []
        for attr in self.__dict__:
            if attr == "particle" or attr == "material":
                continue
            if attr.startswith("__") or attr.startswith("_"):
                continue
            attrs_to_delte.append(attr)
        for attr in attrs_to_delte:
            delattr(self, attr)

    @property
    def step_length(self) -> float:
        return self.__step_length

    @property
    def primary_energy(self) -> float:
        return self.__primary_energy

    @property
    def beta(self) -> float:
        return self.particle.beta(self.__primary_energy)

    @property
    def gamma(self) -> float:
        return self.particle.gamma(self.__primary_energy)

    @property
    def t_max(self) -> float:
        return self.particle.t_max(self.__primary_energy)

    @property
    def t_up(self) -> float:
        t_max = self.t_max
        t_cut = self.material.Tc
        if t_max < t_cut:
            return t_max
        return t_cut

    @property
    def ipotFluct(self) -> float:
        return self.material.I

    def set_fluct_params(self) -> None:
        """
        Parameter/regime computation of G4UniversalFluctuation::SampleFluctuations.
        Sampling itself lives in the vectorized torch kernels
        (`continuous/_kernels.py`), pinned to this scalar math by
        tests/physics/g4h_ionisation/explicit_physics/test_glandz_params_parity.py.
        """
        self.straggling_type = []

        # short step
        self.average_loss = self.__step_length * self.energy_to_dE_dx(self.__primary_energy)  # eV/cm * cm = eV

        # long step
        if self.average_loss > self.primary_energy * self.linear_loss_limit:
            self.x = self.f_range - self.step_length * U.cm2mm
            self.de = self.primary_energy - self.scaled_kin_energy_for_loss(self.x)
            logging.info("### G4VEnergyLossProcess::AlongStepDoIt: "
                             f"\n\tx = {self.x * U.mm2m}\n"
                             f"\tfRange = {self.f_range * U.mm2m}\n"
                             f"\tlength = {self.step_length * U.cm2m}\n"
                             f"\treduceFactor = {1}\n"
                             f"\tpreStepKinEnergy = {self.__primary_energy * U.eV2MeV}\n"
                             f"\tScaledKinEnergyForLoss = {self.scaled_kin_energy_for_loss(self.x) * U.eV2MeV}\n"
                             f"\tmassRatio = {1}")
            if self.de > 0:
                self.average_loss = self.de

        if self.average_loss > self.primary_energy:
            return

        logging.debug("### G4UniversalFluctuation::SampleFluctuations:")
        logging.debug(f"\tkinetic_energy = {self.__primary_energy}")
        logging.debug(f"\tlength = {self.__step_length * U.cm2m}")
        logging.debug(f"\ttcut = {self.material.Tc}")
        logging.debug(f"\ttmax = {self.t_max}")
        logging.debug(f"\taverageLoss = {self.average_loss}")

        if self.average_loss < self.min_energy_loss:  # TODO: Find what is minLoss
            logging.debug(f"output = {self.average_loss}")
            self.straggling_type.append(StragglingType.DeterministicBetheBloch)
            return

        self.mean_loss = self.average_loss
        logging.debug(f"\tmean_loss = {self.mean_loss}")
        logging.debug(f"\tbeta = {self.beta}")
        logging.debug(f"\tgamma = {self.gamma}")
        logging.debug(f"\tbeta2 = {self.beta**2}")
        logging.debug(f"\tgamma2 = {self.gamma**2}")
        self.siga = 0.

        # Gaussian regime
        # for heavy particles only and conditions
        # for Gauusian fluct. has been changed
        if (self.particle.mass_eV > electron_mass_eV)\
                and (self.mean_loss >= self.n_min_bohr * self.material.Tc)\
                and (self.t_max <= 2. * self.material.Tc):  # TODO: Verify this condition

            self.charge_square = 1.
            twopi_mc2_rcl2 = 2 * math.pi * electron_mass_eV * electron_radius_cm ** 2
            self.siga = math.sqrt((self.t_max / self.beta**2 - 0.5 * self.material.Tc) * twopi_mc2_rcl2 *
                                  self.step_length * self.charge_square * self.material.electron_density)
            logging.debug(f"\tsiga = {self.siga}")

            self.sn = self.mean_loss / self.siga
            logging.debug(f"\tsn = {self.sn}")

            # Thick target
            if self.sn >= 2.:
                self.sample_thick_gauss = True
                self.straggling_type.append(StragglingType.ThickLimit)

            else:
                self.neff = self.sn**2
                logging.debug(f"\tneff = {self.neff}")
                self.sample_thick_gamma = True

            return

        logging.debug(f"\te0 = {self.e0}")
        # very small step or low-density material
        if self.material.Tc <= self.e0:
            logging.debug(f"\toutput = {self.mean_loss}")
            raise ValueError
            # return

        self.scaling = min((1 + 500 / self.material.Tc, 1.5))
        self.mean_loss /= self.scaling
        logging.debug(f"\tscaling = {self.scaling}")
        logging.debug(f"\tmean_loss = {self.mean_loss}")

        self.set_glandz_params()

    def set_glandz_params(self) -> None:
        """
        Parameter computation of G4UniversalFluctuation::SampleGlandz (the
        sampling itself lives in `continuous/_kernels.py`).
        """
        self.a1, self.a3 = 0, 0
        self.e1 = self.ipotFluct
        logging.debug(f"\te1 = {self.e1}")

        if self.material.Tc > self.e1:
            self.a1 = self.mean_loss * (1 - self.ionization_rate) / self.e1
            logging.debug(f"\ta1 = {self.a1}")
            if self.a1 < self.A0:
                self.fwnow = 0.1 + (self.FW - 0.1) * math.sqrt(self.a1 / self.A0)
                self.a1 /= self.fwnow
                self.e1 *= self.fwnow
                logging.debug(f"\tfwnow = {self.fwnow}")
            else:
                self.a1 /= self.FW
                self.e1 *= self.FW
            logging.debug(f"\te1 = {self.e1}")
            logging.debug(f"\ta1 = {self.a1}")
        self.w1 = self.material.Tc/self.e0
        logging.debug(f"\tw1 = {self.w1}")
        self.a3 = self.ionization_rate * self.mean_loss * (self.material.Tc - self.e0) \
                  / (self.e0 * self.material.Tc * math.log(self.w1))
        logging.debug(f"\ta3 = {self.a3}")
        if self.a1 <= 0.:
            self.a3 /= self.ionization_rate
            logging.debug(f"\ta3 = {self.a3}")

        self.emean_exc, self.sig2e_exc = 0., 0.

        if self.a1 > 0:
            if self.a1 > self.nmaxCont:
                self.emean_exc += self.a1 * self.e1     # eav from AddExcitation is emean
                self.sig2e_exc += self.a1 * self.e1**2  # esig2 from AddExcitation is sig2e

                logging.debug(f"\teav = {self.emean_exc}")
                logging.debug(f"\tesig2 = {self.sig2e_exc}")
            else:
                self.straggling_type.append(StragglingType.LowColFreqExc)

        self.sample_gauss_exc = self.sig2e_exc > 0.
        if self.sample_gauss_exc:
            self.straggling_type.append(StragglingType.HighColFreqExc)

        # ionisation
        if self.a3 > 0:
            self.emean_ion, self.sig2e_ion = 0., 0.
            self.p3 = self.a3
            self.alfa = 1.
            logging.debug(f"\tp3 = {self.p3}")
            logging.debug(f"\talfa = {self.alfa}")
            if self.a3 > self.nmaxCont:
                self.alfa = self.w1 * (self.nmaxCont + self.a3) / (self.w1 * self.nmaxCont + self.a3)
                self.alfa1 = self.alfa * math.log(self.alfa) / (self.alfa - 1.)
                self.namean = self.a3 * self.w1 * (self.alfa - 1.) / ((self.w1 - 1.) * self.alfa)
                self.emean_ion += self.namean * self.e0 * self.alfa1
                self.sig2e_ion += self.e0**2 * self.namean * (self.alfa - self.alfa1**2)
                self.p3 = self.a3 - self.namean
                logging.debug(f"\talfa = {self.alfa}")
                logging.debug(f"\talfa1 = {self.alfa1}")
                logging.debug(f"\tnamean = {self.namean}")
                logging.debug(f"\temean = {self.emean_ion}")
                logging.debug(f"\tsig2e = {self.sig2e_ion}")
                logging.debug(f"\tp3 = {self.p3}")

            # alfa meaning: ``Since we moved the small energy losses to the Gaussian part, the remaining collisions in the tail effectively start at a slightly higher energy than E0''.
            self.w3 = self.alfa * self.e0
            logging.debug(f"\tw3 = {self.w3}")

            if self.material.Tc > self.w3:
                self.w = (self.material.Tc - self.w3) / self.material.Tc
                logging.debug(f"\tw = {self.w}")
                self.straggling_type.append(StragglingType.LowColFreqIon)

            self.sample_guass_ion = self.sig2e_ion > 0.
            if self.sample_guass_ion:
                self.straggling_type.append(StragglingType.HighColFreqIon)

    @property
    def f_range(self):
        if self.__primary_energy == 0:
            return 0

        f_range = self.energy_to_range(self.__primary_energy)  # mm
        if f_range < 0:
            return 0

        if self.__primary_energy < self.min_kin_energy:
            f_range *= math.sqrt(self.__primary_energy / self.min_kin_energy)

        return f_range  # mm

    def scaled_kin_energy_for_loss(self, R: float) -> float:
        return self.range_to_scaled_energy(R)


class ContinuousStragglingModel(ContinuousLossParams):
    """
    Straggling function of the continuous energy loss model.
    TODO: Verify this implementation fits GEANT4 data
    """
    def __init__(self, particle: Particle = proton, material: Material = material):
        ContinuousLossParams.__init__(self, particle, material)

    @property
    def straggling_function(self) -> Callable[[NDArray[float]], NDArray[float]]:
        """
        Combine the excitation and ionization straggling functions.
        """
        if StragglingType.DeterministicBetheBloch in self.straggling_type:
            def pdf(x: NDArray[float]) -> NDArray[float]:
                idx = np.argmin(np.abs(x - self.average_loss))
                bin_width = x[idx] - x[idx - 1]
                output = np.zeros_like(x)
                output[idx] = 1. / bin_width
                return output
            return pdf

        if StragglingType.ThickLimit in self.straggling_type:
            # truncated Gaussian
            return lambda x: truncated_gaussian(x, self.mean_loss, self.siga)

        ion_stragg_func = self._ionization_straggling_function
        exc_stragg_func = self._excitation_straggling_function
        return lambda x: convolve_2_pdfs(x, ion_stragg_func(x), exc_stragg_func(x))

    @property
    def straggling_cdf(self) -> Callable[[NDArray[float]], NDArray[float]]:
        """
        Compute the cumulative distribution function (CDF) of the straggling function.
        """
        return lambda x: np.cumsum(self.straggling_function(x)) * (x[1] - x[0])

    @property
    def _excitation_straggling_function(self) -> Callable[[NDArray[float]], NDArray[float]]:
        """
        Choose the excitation straggling function model, based on primary energy and step length,
        and compute it.
        """
        if StragglingType.LowColFreqExc in self.straggling_type and StragglingType.HighColFreqExc in self.straggling_type:
            raise NotImplementedError
        if StragglingType.HighColFreqExc in self.straggling_type:
            pdf = lambda x: many_excitation_collisions_pdf(x, self.a1, self.e1)
        elif StragglingType.LowColFreqExc in self.straggling_type:
            pdf = lambda x: few_excitation_collisions_pdf(x, self.a1, self.e1)
        else:
            raise RuntimeError("Excitation straggling function not defined")

        # Tackle the Glandz scaling factor
        return lambda x: pdf(x/self.scaling)/np.abs(self.scaling)

    @property
    def _ionization_straggling_function(self) -> Callable[[NDArray[float]], NDArray[float]]:
        """
        Choose the ionization straggling function model, based on primary energy and step length,
        and compute it.
        """
        # TODO: The current implementation of ionization collisions finds x by itself, but it should be given as an argument.
        #  Also, need to find the optimum f_max and N.
        if StragglingType.LowColFreqIon in self.straggling_type and StragglingType.HighColFreqIon not in self.straggling_type:
            pdf = lambda x: few_ionization_collisions_pdf(x, self.w3, self.w, self.p3, f_max=20, N=2_000_000)
        elif StragglingType.HighColFreqIon in self.straggling_type:
            if StragglingType.LowColFreqIon not in self.straggling_type:
                raise NotImplementedError
            pdf = lambda x: many_ionization_collisions_pdf(x, self.emean_ion, self.sig2e_ion, self.w3, self.w, self.p3, f_max=5, N=2_000_000)
        else:
            raise RuntimeError("Ionization straggling function not defined")

        # Tackle the Glandz scaling factor
        return lambda x: pdf(x / self.scaling) / np.abs(self.scaling)


