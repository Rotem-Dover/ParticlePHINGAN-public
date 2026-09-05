"""
t_max kinematics per particle species.

Reference: geant4-11.2.0 —
  G4MollerBhabhaModel::MaxSecondaryEnergy: e- => T/2 (identical outgoing
  electrons; the faster one is the primary by convention), e+ => T.
  G4BetheBlochModel::MaxSecondaryEnergy: heavy-particle kinematic limit.
"""
import physics.definitions.physical_constants as C
from physics.definitions.particle import Particle, electron, proton


class TestTMax:
    def test_electron_t_max_is_half_kinetic_energy(self):
        for ke in (1.0e6, 1.0e8):
            assert electron.t_max(ke) == 0.5 * ke

    def test_positive_lepton_t_max_is_full_kinetic_energy(self):
        positron = Particle(
            name="positron", mass_eV=C.electron_mass_eV,
            mass_amu=electron.mass_amu, charge=+1.0, spin=0.5,
            lepton_number=-1, magnetic_moment=electron.magnetic_moment,
        )
        for ke in (1.0e6, 1.0e8):
            assert positron.t_max(ke) == ke

    def test_proton_t_max_matches_the_heavy_particle_formula(self):
        # Reference values from the heavy-particle kinematic limit formula;
        # the lepton branch must leave the hadron path byte-for-byte
        # identical to it.
        reference_values = {
            1.0e6: 2177.2542707827442,
            1.0e7: 21876.680987296877,
            1.0e8: 229179.44241579832,
        }
        for ke, expected in reference_values.items():
            assert proton.t_max(ke) == expected
