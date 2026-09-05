"""
Defines the Particle dataclass representing a charged projectile (e.g. proton) with
its mass, charge, spin, and methods for relativistic kinematics (gamma, beta) and
maximum single-collision energy transfer (T_max).
"""
from dataclasses import dataclass

import physics.definitions.physical_constants as C


@dataclass
class Particle:
    name:            str    # "proton", "alpha"
    mass_eV:         float  # mass of particle in eV
    mass_amu:        float  # mass of particle in atomic mass units
    charge:          float  # charge of particle
    spin:            float  # spin of particle
    lepton_number:   float  # lepton number of particle. 0 for non-leptons, ±1 for leptons
    magnetic_moment: float  # magnetic moment of particle in units of nuclear magneton

    def __post_init__(self):
        # Compute the m_e/M ratio in MeV-magnitude operands so it matches G4
        # (which uses `electron_mass_c2 / mass` internally in MeV units, e.g.
        # 0.510998910 / 938.272013). The eV-magnitude division
        # 510998.910 / 938272013.0 differs by 1 ULP, and that ULP propagates
        # through the ionisation lambda table and amplifies through its
        # spline into the steep threshold-adjacent bin.
        import common.units as U
        self.electron_mass_ratio = (C.electron_mass_eV * U.eV2MeV) / (self.mass_eV * U.eV2MeV)

    def gamma(self, energy: float) -> float:
        return energy / self.mass_eV + 1

    def beta(self, energy: float) -> float:
        return (1 - 1 / self.gamma(energy) ** 2) ** 0.5

    def t_max(self, energy: float) -> float:
        """
        Maximum energy transfer in a single collision.

        Leptons (geant4-11.2.0 G4MollerBhabhaModel::MaxSecondaryEnergy):
          e- (Møller): the two outgoing electrons are identical, so by
          convention the faster one is the primary => T_max = T/2.
          e+ (Bhabha): T_max = T.
        Heavy particles: the standard kinematic limit
          (G4BetheBlochModel::MaxSecondaryEnergy).
        """
        if self.lepton_number != 0:
            return 0.5 * energy if self.charge < 0 else energy
        g = self.gamma(energy)
        b = self.beta(energy)
        return ((2*C.electron_mass_eV)*((b*g)**2))/(1 + 2*g*self.electron_mass_ratio + self.electron_mass_ratio**2)


proton = Particle(
    name="proton", mass_eV=C.proton_mass_eV, mass_amu=C.proton_mass_amu, charge=+1., spin=0.5,
    lepton_number=0, magnetic_moment=C.proton_mag_mom
)

electron = Particle(
    name="electron", mass_eV=C.electron_mass_eV, mass_amu=C.electron_mass_amu, charge=-1., spin=0.5,
    lepton_number=+1, magnetic_moment=C.electron_mag_mom
)
