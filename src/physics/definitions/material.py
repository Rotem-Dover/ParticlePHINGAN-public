"""
Defines the Material dataclass holding properties of a target material (atomic number,
density, mean excitation energy, production threshold, etc.) and derived quantities
such as electron density. Pre-configured instances are provided for silicon, aluminum,
iron and beryllium.
"""
from dataclasses import dataclass
from physics.definitions.physical_constants import avogadro


@dataclass
class Material:
    name: str    # name of material
    Tc:   float  # production threshold for delta ray in eV
    I:    float  # mean excitation energy in eV
    A:    float  # atomic mass in g/mole
    Z:    float  # atomic number
    rho:  float  # density in g/cm3
    radiation_length_cm: float = 0.0  # radiation length X0 in cm

    @property
    def number_of_atoms_per_volume(self):
        return avogadro * self.rho / self.A

    @property
    def electron_density(self):
        return self.number_of_atoms_per_volume * self.Z


silicon  = Material(name="Silicon",  Tc=990, I=173., A=28.0855, Z=14, rho=2.329, radiation_length_cm=9.370)
aluminum = Material(name="Aluminum", Tc=1010.5686025013306, I=166., A=26.9815, Z=13, rho=2.699, radiation_length_cm=8.897)

# Every float of the two materials below is pasted verbatim from a Geant4
# 11.2.0 InitTablesDump MATPROPS printout captured at a 1 um production cut
# (measurements/g4_material_constants_dump.txt; the dump tool's cut argument is
# in nanometres, so a 1000.0 argument means 1 um), and
# tests/physics/test_material_constants_match_g4_dump.py re-parses that file to
# pin them. Tc is the electron production threshold Geant4 derives from the
# range cut in each material.
#
# Aluminum above is NOT from that printout and is not pinned to it: its Tc, rho
# and radiation length differ from the dump's G4_Al line at the ~1e-13 relative
# level, floating-point noise between Geant4 builds rather than a physics
# difference. What pins aluminum is downstream and covers all three materials:
# tests/physics/g4_init_tables/test_vs_g4_reference.py rebuilds the four
# initialisation tables from these constants and compares them, node by node at
# roughly 1 ULP, against binaries dumped out of Geant4's own G4PhysicsVector
# objects. Silicon has no MATPROPS line in the dump and no reference tables, and
# is unused by the runtime.
iron      = Material(name="Iron",      Tc=5591.6392187789997, I=286, A=55.845110798, Z=26, rho=7.8739999999999988, radiation_length_cm=1.7574934650970921)
beryllium = Material(name="Beryllium", Tc=990, I=63.700000000000003, A=9.0121800000000007, Z=4, rho=1.8480000000000003, radiation_length_cm=35.27597513557388)